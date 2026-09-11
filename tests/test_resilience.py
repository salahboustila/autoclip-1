"""Server resilience: the log file, crash capture, the Windows console, and sleep."""

from __future__ import annotations

import contextlib
import ctypes
import faulthandler
import logging
import logging.config
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from autoclip import cli, paths, resilience
from autoclip.jobs import queue as job_queue
from typer.testing import CliRunner


def _silent(*_args: object) -> None:
    """Stands in for the default hooks, which would print into the test output."""


def _raise(exc: BaseException) -> None:
    raise exc


@contextlib.contextmanager
def restored_logging() -> Iterator[None]:
    """Put back every logger the server's config rewires, closing what it opened.

    Used inside each test rather than as a fixture so it unwinds in the same test
    phase as pytest's own log capture, which also hangs handlers on the root.
    """
    names = ["", *resilience.log_config(Path("unused.log"))["loggers"]]
    loggers = [logging.getLogger(name) for name in names]
    saved = [(lg, lg.handlers[:], lg.level, lg.propagate, lg.filters[:]) for lg in loggers]
    try:
        yield
    finally:
        for logger, handlers, level, propagate, filters in saved:
            for handler in logger.handlers:
                if handler not in handlers:
                    handler.close()
            logger.handlers[:] = handlers
            logger.setLevel(level)
            logger.propagate = propagate
            logger.filters[:] = filters


class TestLogFile:
    def test_app_and_uvicorn_logs_reach_it(self, tmp_path: Path) -> None:
        log_file = tmp_path / "server.log"
        with restored_logging():
            logging.config.dictConfig(resilience.log_config(log_file))
            # Nothing used to configure the root logger, so the app's own INFO
            # lines (job started, job failed) were dropped entirely.
            logging.getLogger("autoclip.jobs.queue").info("Starting job abc123.")
            logging.getLogger("uvicorn.error").info("Uvicorn running on port 8010")
            logging.getLogger("uvicorn.access").info(
                '%s - "%s %s HTTP/%s" %d', "127.0.0.1:5000", "GET", "/api/health", "1.1", 200
            )
            logging.getLogger("httpx").info("HTTP Request: POST https://provider.invalid")

        text = log_file.read_text(encoding="utf-8")
        assert "Starting job abc123." in text
        assert "Uvicorn running on port 8010" in text
        assert '"GET /api/health HTTP/1.1" 200' in text
        assert "provider.invalid" not in text

    def test_harmless_connection_resets_are_left_out(self, tmp_path: Path) -> None:
        # Windows logs this traceback whenever a browser drops a connection. In a
        # log read after an outage it looks like the cause, and it never is.
        log_file = tmp_path / "server.log"
        asyncio_log = logging.getLogger("asyncio")
        with restored_logging():
            logging.config.dictConfig(resilience.log_config(log_file))
            try:
                raise ConnectionResetError(10054, "An existing connection was forcibly closed")
            except ConnectionResetError:
                asyncio_log.error(
                    "Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)",
                    exc_info=True,
                )
            try:
                raise ValueError("a real problem")
            except ValueError:
                asyncio_log.error("Task exception was never retrieved", exc_info=True)

        text = log_file.read_text(encoding="utf-8")
        assert "_call_connection_lost" not in text
        assert "a real problem" in text


def test_unhandled_exceptions_are_logged_and_native_crashes_go_to_crash_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    faulthandler_files: list[str] = []
    monkeypatch.setattr(
        faulthandler, "enable", lambda file, all_threads: faulthandler_files.append(file.name)
    )
    monkeypatch.setattr(resilience, "_crash_file", None)
    log_file = tmp_path / "server.log"

    # Swapped by hand rather than with monkeypatch so they're restored within this
    # test phase: pytest installs its own threading.excepthook around each test.
    saved_hooks = sys.excepthook, threading.excepthook
    sys.excepthook = threading.excepthook = _silent
    try:
        with restored_logging():
            logging.config.dictConfig(resilience.log_config(log_file))
            resilience.install_crash_logging(tmp_path)

            sys.excepthook(RuntimeError, RuntimeError("main thread boom"), None)
            worker = threading.Thread(
                target=_raise, args=(ValueError("worker boom"),), name="export-worker"
            )
            worker.start()
            worker.join()
    finally:
        sys.excepthook, threading.excepthook = saved_hooks
        if resilience._crash_file is not None:
            resilience._crash_file.close()

    text = log_file.read_text(encoding="utf-8")
    assert "main thread boom" in text
    assert "export-worker" in text
    assert "worker boom" in text
    assert faulthandler_files == [str(tmp_path / resilience.CRASH_FILE_NAME)]


def test_console_tweaks_do_nothing_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")

    assert resilience.disable_quick_edit() is False
    assert resilience.log_console_events() is False
    # True meaning colours already work there, so the formatters keep them.
    assert resilience.enable_ansi_colors() is True
    with resilience.keep_awake():
        pass


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        # A stock Windows 10 console: QuickEdit and the extended flags both on.
        (0x01F7, 0x01B7),
        # Extended flags off: they must be set, or Windows ignores the change.
        (0x0047, 0x0087),
    ],
)
def test_quick_edit_is_cleared_and_every_other_mode_kept(mode: int, expected: int) -> None:
    assert resilience.without_quick_edit(mode) == expected


@pytest.mark.parametrize(("mode", "expected"), [(0x0003, 0x0007), (0x0007, 0x0007)])
def test_colour_rendering_is_switched_on_and_every_other_mode_kept(
    mode: int, expected: int
) -> None:
    assert resilience.with_ansi_colors(mode) == expected


def test_colours_are_turned_off_where_the_console_cant_show_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resilience, "enable_ansi_colors", lambda: False)
    monkeypatch.setattr(resilience, "install_crash_logging", lambda directory: None)
    monkeypatch.setattr(resilience, "disable_quick_edit", lambda: False)
    monkeypatch.setattr(resilience, "log_console_events", lambda: False)

    with restored_logging():
        config = resilience.prepare_server()

    assert config["formatters"]["default"]["use_colors"] is False
    assert config["formatters"]["access"]["use_colors"] is False


def test_keep_awake_holds_off_idle_sleep_until_the_block_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def set_thread_execution_state(flags: int) -> int:
        calls.append(flags)
        return 0

    kernel32 = SimpleNamespace(SetThreadExecutionState=set_thread_execution_state)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: kernel32, raising=False)
    held = resilience.ES_CONTINUOUS | resilience.ES_SYSTEM_REQUIRED

    with resilience.keep_awake():
        assert calls == [held]
    assert calls == [held, resilience.ES_CONTINUOUS]

    # A job that fails must still let the laptop sleep again.
    with pytest.raises(RuntimeError), resilience.keep_awake():
        raise RuntimeError("job failed")
    assert calls[-1] == resilience.ES_CONTINUOUS


def test_jobs_keep_the_laptop_awake_while_they_run(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    @contextlib.contextmanager
    def recording_keep_awake() -> Iterator[None]:
        events.append("held")
        yield
        events.append("released")

    class FakeRunner:
        async def run(self) -> list:
            events.append("job ran")
            return []

    monkeypatch.setattr(resilience, "keep_awake", recording_keep_awake)

    assert job_queue._run_blocking(FakeRunner()) == []
    assert events == ["held", "job ran", "released"]


def test_serve_logs_to_a_file_in_the_autoclip_home(monkeypatch: pytest.MonkeyPatch) -> None:
    uvicorn_calls: list[dict] = []
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: uvicorn_calls.append(kwargs))
    # Process-wide side effects, covered by the tests above.
    monkeypatch.setattr(resilience, "install_crash_logging", lambda directory: None)
    monkeypatch.setattr(resilience, "disable_quick_edit", lambda: False)
    monkeypatch.setattr(resilience, "enable_ansi_colors", lambda: True)
    monkeypatch.setattr(resilience, "log_console_events", lambda: False)
    log_file = paths.logs_dir() / resilience.LOG_FILE_NAME

    with restored_logging():
        result = CliRunner().invoke(cli.app, ["serve", "--no-open", "--port", "8123"])

    assert result.exit_code == 0, result.output
    assert uvicorn_calls[0]["log_config"]["handlers"]["file"]["filename"] == str(log_file)
    text = log_file.read_text(encoding="utf-8")
    assert "server starting" in text
    assert "server stopped" in text
