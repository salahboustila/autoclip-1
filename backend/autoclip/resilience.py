"""Keep the web server up, and leave a record when it goes down.

The server runs in a console window, which on Windows fails in two ways that
leave no trace:

* **QuickEdit.** It's on by default, and one click in the window starts a text
  selection. While a selection is active every write to the console blocks, and
  uvicorn writes a line per request — so the next request freezes the event
  loop, and the server looks down until someone presses Esc in that window.
* **No log file.** Everything went to the console, so a crash's traceback left
  with the window. The app's own INFO logs never appeared at all, because
  nothing configured the root logger.

`prepare_server` deals with both before uvicorn starts. `keep_awake` covers the
jobs themselves, which a laptop idling into sleep would otherwise stall.
"""

from __future__ import annotations

import contextlib
import copy
import faulthandler
import logging
import logging.config
import os
import platform
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

import uvicorn.config

from . import __version__, paths

log = logging.getLogger(__name__)

#: Configured to write only to the log file. The hooks that use it chain on to
#: the default hooks, which print the same traceback to the console.
crash_log = logging.getLogger("autoclip.crash")

LOG_FILE_NAME = "server.log"
CRASH_FILE_NAME = "crash.log"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3

STD_INPUT_HANDLE = -10
STD_OUTPUT_HANDLE = -11
STD_ERROR_HANDLE = -12
ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
ENABLE_QUICK_EDIT_MODE = 0x0040
ENABLE_EXTENDED_FLAGS = 0x0080
ES_SYSTEM_REQUIRED = 0x00000001
ES_CONTINUOUS = 0x80000000

#: The CTRL_*_EVENT values Windows sends a console's processes, as log wording.
CONSOLE_EVENTS = {
    0: "Ctrl+C was pressed",
    1: "Ctrl+Break was pressed",
    2: "the console window was closed",
    5: "the user logged off",
    6: "Windows is shutting down",
}

#: Open for the life of the process: faulthandler writes to it while the
#: interpreter is dying, so it must never be closed.
_crash_file: TextIO | None = None

#: Windows calls this ctypes callback, so it has to outlive its registration.
_console_handler: Any = None


def prepare_server() -> dict[str, Any]:
    """Set up the console, the log file and crash capture. Returns uvicorn's log config."""
    directory = paths.logs_dir()
    directory.mkdir(parents=True, exist_ok=True)

    config = log_config(directory / LOG_FILE_NAME)
    if not enable_ansi_colors():
        # A console that can't render uvicorn's colour codes would print them raw.
        for name in ("default", "access"):
            config["formatters"][name]["use_colors"] = False
    logging.config.dictConfig(config)
    install_crash_logging(directory)
    quick_edit_off = disable_quick_edit()
    log_console_events()

    log.info(
        "AutoClip %s server starting (pid %d, Python %s%s). Logs in %s",
        __version__,
        os.getpid(),
        platform.python_version(),
        ", QuickEdit off" if quick_edit_off else "",
        directory,
    )
    return config


def log_config(log_file: Path) -> dict[str, Any]:
    """uvicorn's logging config, plus a rotating log file and the app's own loggers.

    The file handler is listed first so a record reaches disk before the console
    handler, which is the one that can block.
    """
    config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    config["formatters"]["file"] = {"format": "%(asctime)s %(levelname)-8s %(name)s: %(message)s"}
    config["filters"] = {"connection_reset": {"()": ConnectionResetFilter}}
    config["handlers"]["file"] = {
        "class": "logging.handlers.RotatingFileHandler",
        "formatter": "file",
        "filename": str(log_file),
        "maxBytes": LOG_MAX_BYTES,
        "backupCount": LOG_BACKUP_COUNT,
        "encoding": "utf-8",
    }

    loggers = config["loggers"]
    for name in ("uvicorn", "uvicorn.access"):
        loggers[name]["handlers"] = ["file", *loggers[name]["handlers"]]
    loggers["autoclip.crash"] = {"handlers": ["file"], "level": "INFO", "propagate": False}
    loggers["asyncio"] = {"filters": ["connection_reset"]}
    # HTTP clients log every provider request at INFO.
    for name in ("httpx", "httpcore", "google_genai"):
        loggers[name] = {"level": "WARNING"}

    config["root"] = {"handlers": ["file", "default"], "level": "INFO"}
    return config


class ConnectionResetFilter(logging.Filter):
    """Drop asyncio's traceback for a browser that hung up mid-response.

    Windows' proactor event loop logs ``ConnectionResetError`` [WinError 10054]
    from ``_call_connection_lost`` whenever a client drops a connection. It's
    harmless, but in a log read after an outage it looks like the cause.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info[1] if record.exc_info else None
        return not (
            isinstance(exc, ConnectionResetError) and "_call_connection_lost" in record.getMessage()
        )


def install_crash_logging(directory: Path) -> None:
    """Record whatever would take the process down.

    A native crash — in CTranslate2 or MediaPipe, say — never reaches Python's
    exception handling, so faulthandler writes those tracebacks to ``crash.log``.
    Uncaught Python exceptions, in any thread, go to ``server.log``.
    """
    global _crash_file
    if _crash_file is None:
        _crash_file = (directory / CRASH_FILE_NAME).open("a", encoding="utf-8")
        faulthandler.enable(file=_crash_file, all_threads=True)

    previous_hook = sys.excepthook
    previous_thread_hook = threading.excepthook

    def excepthook(
        exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None
    ) -> None:
        if not issubclass(exc_type, KeyboardInterrupt):
            crash_log.critical(
                "Unhandled exception; the server is exiting.", exc_info=(exc_type, exc, tb)
            )
        previous_hook(exc_type, exc, tb)

    def thread_excepthook(args: threading.ExceptHookArgs) -> None:
        # SystemExit ends a thread quietly by design; the default hook ignores it too.
        if args.exc_type is not SystemExit:
            crash_log.error(
                "Unhandled exception in thread %s.",
                args.thread.name if args.thread else "unknown",
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
        previous_thread_hook(args)

    sys.excepthook = excepthook
    threading.excepthook = thread_excepthook


def without_quick_edit(mode: int) -> int:
    """A console input mode with QuickEdit cleared and everything else kept.

    ENABLE_EXTENDED_FLAGS has to be set alongside, or Windows ignores the change
    to the QuickEdit bit.
    """
    return (mode & ~ENABLE_QUICK_EDIT_MODE) | ENABLE_EXTENDED_FLAGS


def with_ansi_colors(mode: int) -> int:
    """A console output mode that renders colour codes instead of printing them."""
    return mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING


def disable_quick_edit() -> bool:
    """Turn QuickEdit off in this console window only. Returns whether it did.

    Does nothing off Windows, or when stdin isn't a console (redirected, or
    running as a service). Text can still be selected through the window menu's
    Edit > Mark, which blocks output just the same until Esc is pressed.
    """
    return bool(_update_console_mode(STD_INPUT_HANDLE, without_quick_edit))


def enable_ansi_colors() -> bool:
    """Have the console render colour codes. Returns False only if it can't.

    uvicorn colours its log prefixes, and the Windows 10 console prints the codes
    as raw text (``←[32mINFO←[0m``) unless virtual terminal processing is on.
    Other platforms, and output that isn't a console, need nothing.
    """
    return all(
        _update_console_mode(handle, with_ansi_colors) is not False
        for handle in (STD_OUTPUT_HANDLE, STD_ERROR_HANDLE)
    )


def _update_console_mode(std_handle: int, change: Callable[[int], int]) -> bool | None:
    """Apply ``change`` to a standard handle's console mode.

    Returns None off Windows or when the handle isn't a console, otherwise
    whether Windows accepted the new mode.
    """
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetStdHandle.restype = wintypes.HANDLE
    kernel32.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]

    handle = kernel32.GetStdHandle(std_handle)
    mode = wintypes.DWORD()
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        return None
    return bool(kernel32.SetConsoleMode(handle, change(mode.value)))


def log_console_events() -> bool:
    """Log why Windows is about to stop the server. Returns whether it's listening.

    Closing the console window ends the process with no exception and no
    shutdown log, so without this a closed window and a crash look the same
    afterwards: the log just stops.
    """
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    global _console_handler

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
    def on_console_event(event: int) -> bool:
        log.warning("Stopping: %s.", CONSOLE_EVENTS.get(event, f"console event {event}"))
        # Not handled here, so Python's Ctrl+C handling and Windows' defaults still run.
        return False

    _console_handler = on_console_event
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    return bool(kernel32.SetConsoleCtrlHandler(on_console_event, True))


@contextlib.contextmanager
def keep_awake() -> Iterator[None]:
    """Hold off Windows' idle sleep until the block ends.

    A laptop left alone mid-job otherwise sleeps on its idle timer, which stalls
    the job, and its network calls (the provider, a YouTube download) fail on
    waking. Only the idle timer is held off: the screen can still turn off, and
    closing the lid still sleeps. Windows keeps the request per thread, so enter
    this on the thread doing the work.
    """
    if sys.platform != "win32":
        yield
        return
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
    kernel32.SetThreadExecutionState.restype = ctypes.c_uint32
    kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
