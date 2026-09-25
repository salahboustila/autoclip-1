"""Ingestion — YouTube URLs and local file uploads become source records.

Both paths converge on a validated :class:`~autoclip.db.models.Source` with a
file on disk and probed metadata.

A note on YouTube: as of 2026 most anonymous downloads hit a bot check, and
proof-of-origin tokens no longer clear it reliably — browser cookies do. So
:class:`IngestError` distinguishes that specific failure and tells the user how
to fix it, rather than surfacing a raw yt-dlp traceback.

A second, distinct failure — YouTube's own player says "The page needs to be
reloaded" — means the signature/n-value JS challenge wasn't solved. yt-dlp's
built-in solvers cover most cases; the rest need its official EJS solver
script, fetched from GitHub on demand, and a signed-in session. See
``_is_challenge_error`` for how the two fixes are applied.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path

from .. import paths
from ..config import IngestSettings
from ..db.models import Source, new_id
from . import ffmpeg

log = logging.getLogger(__name__)

#: Container and audio formats we accept for upload.
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
ACCEPTED_SUFFIXES = VIDEO_SUFFIXES | AUDIO_SUFFIXES

_YOUTUBE_HOSTS = ("youtube.com", "youtu.be", "www.youtube.com", "m.youtube.com")

#: yt-dlp error fragments that mean "YouTube wants proof you're a human".
_BOT_CHECK_MARKERS = (
    "sign in to confirm",
    "confirm you're not a bot",
    "confirm you are not a bot",
    "this content isn't available",
    "player response",
)

#: Browsers whose cookies yt-dlp cannot decrypt on Windows: Chromium locks them
#: behind app-bound encryption, which it has no support for. Naming them in a hint
#: sends people to set up cookies that can never work.
_UNREADABLE_ON_WINDOWS = ("chrome", "chromium", "edge", "brave", "opera", "vivaldi")

#: yt-dlp error fragments that mean the request never reached YouTube: the DNS
#: lookup or the connection itself failed. A Wi-Fi drop or a VPN reconnecting
#: causes these, so they're worth retrying — and updating yt-dlp won't fix them.
_NETWORK_MARKERS = (
    "getaddrinfo failed",
    "failed to resolve",
    "name resolution",
    "timed out",
    "connection reset",
    "connection aborted",
    "connection refused",
    "network is unreachable",
    "no route to host",
)

#: Tries for a download that couldn't connect. The pause between them starts at
#: NETWORK_RETRY_DELAY_S and doubles: 3 s, then 6 s.
NETWORK_ATTEMPTS = 3
NETWORK_RETRY_DELAY_S = 3.0

#: YouTube's own player-facing text when the signature/n-value JS challenge
#: couldn't be solved and the request is refused. Not yt-dlp's wording — it
#: comes straight from YouTube's player response, so it can't be found anywhere
#: in the yt-dlp package itself.
_CHALLENGE_MARKERS = ("the page needs to be reloaded",)

#: Browser to retry with when the JS challenge fails and no browser is already
#: configured. Firefox because it's the only one yt-dlp can read cookies from
#: on Windows (see _UNREADABLE_ON_WINDOWS) — the same reason _cookie_hint steers
#: people there. Overridable per machine without touching Settings.
ENV_CHALLENGE_BROWSER = "AUTOCLIP_YTDLP_CHALLENGE_BROWSER"
DEFAULT_CHALLENGE_BROWSER = "firefox"

#: Permission, not a fetch: yt-dlp only downloads this when a challenge actually
#: needs solving, and it's yt-dlp's own signed release — nothing browser-related.
#: Confirmed 2026-09-24 (https://youtu.be/mPQlJp7Rqr0) that this alone clears
#: some "page needs to be reloaded" failures, so it's sent on every request
#: rather than held back for a retry.
_REMOTE_COMPONENTS = ["ejs:github"]


class IngestError(RuntimeError):
    """Ingestion failed in a way worth explaining to the user."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base}\n\n{self.hint}" if self.hint else base


def is_youtube_url(url: str) -> bool:
    from urllib.parse import urlparse

    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return host in _YOUTUBE_HOSTS


def slugify(text: str, *, max_length: int = 60) -> str:
    """Turn a title into a filesystem- and URL-safe slug."""
    slug = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug[:max_length] or "clip"


# --------------------------------------------------------------------------
# YouTube
# --------------------------------------------------------------------------


def ingest_youtube(
    url: str,
    settings: IngestSettings | None = None,
    *,
    on_progress: Callable[[float], None] | None = None,
) -> Source:
    """Download a YouTube video and return a validated source record.

    Preconditions:
        url points at content the user owns or has the rights to process.
    """
    import yt_dlp

    settings = settings or IngestSettings()
    source_id = new_id()
    target_dir = paths.source_media_dir(source_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    def hook(status: dict) -> None:
        if not on_progress or status.get("status") != "downloading":
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate")
        done = status.get("downloaded_bytes")
        if total and done:
            on_progress(min(1.0, done / total))

    options: dict = {
        "format": settings.ytdlp_format,
        "outtmpl": str(target_dir / "source.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
        "retries": 3,
        "fragment_retries": 3,
        "remote_components": list(_REMOTE_COMPONENTS),
    }
    if settings.cookies_from_browser:
        # yt-dlp expects a tuple; only the browser name is required.
        options["cookiesfrombrowser"] = (settings.cookies_from_browser,)
    if settings.prefer_youtube_captions:
        options["writeautomaticsub"] = True
        options["subtitleslangs"] = ["en.*"]
        options["subtitlesformat"] = "json3"

    # A fresh cookie jar is only worth trying once, and only if the request
    # didn't already carry one — retrying with the same cookies that just
    # failed would just fail again the same way.
    challenge_retry_available = not settings.cookies_from_browser
    challenge_retried = False

    metadata: dict | None = None
    network_attempt = 1
    while True:
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                metadata = ydl.extract_info(url, download=True)
            break
        except yt_dlp.utils.DownloadError as exc:
            if challenge_retry_available and not challenge_retried and _is_challenge_error(exc):
                challenge_retried = True
                browser = _challenge_browser(settings)
                options = {**options, "cookiesfrombrowser": (browser,)}
                log.warning(
                    "YouTube's JS challenge failed for %s; retrying once with %s cookies: %s",
                    url,
                    browser,
                    exc,
                )
                continue  # not a network blip, so no backoff — retry right away
            if network_attempt < NETWORK_ATTEMPTS and _is_network_error(exc):
                delay = NETWORK_RETRY_DELAY_S * 2 ** (network_attempt - 1)
                log.warning(
                    "Couldn't reach YouTube (attempt %d/%d), retrying in %.0fs: %s",
                    network_attempt,
                    NETWORK_ATTEMPTS,
                    delay,
                    exc,
                )
                time.sleep(delay)
                network_attempt += 1
                continue
            shutil.rmtree(target_dir, ignore_errors=True)
            raise _translate_ytdlp_error(
                exc, settings, challenge_retried=challenge_retried
            ) from exc
        except Exception as exc:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise IngestError(f"Could not download {url}: {exc}") from exc

    downloaded = _find_downloaded_file(target_dir)
    if downloaded is None:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise IngestError("yt-dlp reported success but produced no media file.")

    info = _probe_and_validate(downloaded)

    return Source(
        id=source_id,
        type="youtube",
        url=url,
        path=str(downloaded),
        filename=downloaded.name,
        title=(metadata or {}).get("title") or downloaded.stem,
        channel=(metadata or {}).get("uploader") or (metadata or {}).get("channel"),
        duration_s=info.duration_s or float((metadata or {}).get("duration") or 0.0),
        width=info.width,
        height=info.height,
        fps=info.fps,
        has_audio=info.has_audio,
        has_video=info.has_video,
    )


def _is_network_error(exc: Exception) -> bool:
    """Whether a yt-dlp failure is the connection's fault rather than YouTube's."""
    message = str(exc).lower()
    return any(marker in message for marker in _NETWORK_MARKERS)


def _is_challenge_error(exc: Exception) -> bool:
    """Whether a yt-dlp failure is YouTube's own "page needs to be reloaded"."""
    message = str(exc).lower()
    return any(marker in message for marker in _CHALLENGE_MARKERS)


def _is_cookie_database_missing(exc: Exception) -> bool:
    """Whether a retry with browser cookies failed because there was no database
    to read — yt-dlp's own wording for "this browser isn't installed, or has
    never been run", for either its Firefox path or the generic one."""
    return "cookies database" in str(exc).lower()


def _challenge_browser(settings: IngestSettings) -> str:
    """Browser to retry with when YouTube's JS challenge fails.

    Prefers one the user already configured for the ordinary bot-check flow —
    no reason to ask twice. Otherwise falls back to ENV_CHALLENGE_BROWSER, or
    DEFAULT_CHALLENGE_BROWSER.
    """
    return (
        settings.cookies_from_browser
        or os.environ.get(ENV_CHALLENGE_BROWSER)
        or DEFAULT_CHALLENGE_BROWSER
    )


def _cookie_hint(settings: IngestSettings) -> str:
    """What to do about a bot check, given where cookies are set to come from."""
    browser = (settings.cookies_from_browser or "").lower()
    windows = sys.platform == "win32"

    if browser and windows and browser in _UNREADABLE_ON_WINDOWS:
        return (
            f"Cookies are set to come from {settings.cookies_from_browser}, but on Windows "
            "Chromium browsers encrypt their cookies with a key yt-dlp cannot read, so none "
            "of them ever reach YouTube. Use Firefox instead: sign in to YouTube there, set "
            '`ingest.cookies_from_browser` to "firefox", and close Firefox before downloading.'
        )
    if browser:
        return (
            f"Cookies are already being read from {settings.cookies_from_browser}, but "
            "YouTube still refused. Make sure you are signed in to YouTube in that "
            "browser and that the browser is fully closed — it locks its cookie "
            "database while running."
        )

    choices = (
        '"firefox". Chrome and Edge cannot be used on Windows: they encrypt their cookies '
        "with a key yt-dlp cannot read"
        if windows
        else '"chrome", "firefox", "edge"'
    )
    return (
        "YouTube is asking for proof you're not a bot. Set a browser to pull cookies from — "
        f"in Settings, or via config.json's `ingest.cookies_from_browser` (e.g. {choices}). "
        "You must be signed in to YouTube in that browser, and it must be closed while "
        "AutoClip downloads."
    )


def _translate_ytdlp_error(
    exc: Exception, settings: IngestSettings, *, challenge_retried: bool = False
) -> IngestError:
    """Turn a yt-dlp failure into something the user can act on."""
    message = str(exc).lower()

    if _is_network_error(exc):
        return IngestError(
            "Couldn't reach YouTube.",
            hint=(
                "This computer couldn't connect to YouTube, or look up its address, even "
                f"after {NETWORK_ATTEMPTS} tries. That's the internet connection, not the "
                "video: check the Wi-Fi, or a VPN that is connecting, then try again. "
                "Updating yt-dlp won't help with this one.\n\n"
                f"Original error: {exc}"
            ),
        )

    if _is_challenge_error(exc) or (challenge_retried and _is_cookie_database_missing(exc)):
        return _translate_challenge_error(exc, settings, challenge_retried=challenge_retried)

    if any(marker in message for marker in _BOT_CHECK_MARKERS):
        return IngestError(
            "YouTube blocked this download with a bot check.", hint=_cookie_hint(settings)
        )

    if "private video" in message or "members-only" in message:
        return IngestError(
            "This video is private or members-only.",
            hint="AutoClip only downloads content you can access and have the rights to use.",
        )
    if "unavailable" in message or "removed" in message:
        return IngestError("This video is unavailable or has been removed.")
    if "drm" in message:
        return IngestError(
            "This content is DRM-protected.",
            hint="AutoClip does not and will not circumvent DRM.",
        )

    return IngestError(
        "yt-dlp could not download this video.",
        hint=(
            "YouTube changes frequently and yt-dlp is updated often. Try "
            "`autoclip update-ytdlp` to pull the latest version.\n\n"
            f"Original error: {exc}"
        ),
    )


def _translate_challenge_error(
    exc: Exception, settings: IngestSettings, *, challenge_retried: bool
) -> IngestError:
    """YouTube's JS challenge beat yt-dlp. Name whichever half of the fix —
    the EJS solver script, or a signed-in browser's cookies — is missing,
    rather than a generic failure that sends people to `update-ytdlp`."""
    browser = _challenge_browser(settings)

    if challenge_retried and _is_cookie_database_missing(exc):
        return IngestError(
            "Couldn't read browser cookies to get past YouTube's challenge.",
            hint=(
                f"AutoClip tried reading cookies from {browser}, but yt-dlp found no "
                f"cookie database for it on this machine — {browser} may not be "
                f"installed, or has never been opened. Open {browser} once, sign in to "
                "youtube.com, then retry. To use a different browser instead, set "
                f"`ingest.cookies_from_browser` in Settings, or the {ENV_CHALLENGE_BROWSER} "
                "environment variable.\n\n"
                f"Original error: {exc}"
            ),
        )

    tried = (
        f"AutoClip already retried with yt-dlp's EJS challenge solver and {browser} cookies, "
        "and YouTube still refused."
        if challenge_retried
        else "AutoClip sent yt-dlp's EJS challenge solver (--remote-components ejs:github), "
        "and YouTube still refused."
    )
    return IngestError(
        "YouTube's challenge beat yt-dlp on this video.",
        hint=(
            f"{tried} That usually means this machine can't reach github.com to fetch the "
            "solver script, or the browser's cookies aren't actually signed in to YouTube. "
            "Check that github.com is reachable, and that you're signed in to YouTube in "
            f"{browser}. A newer yt-dlp occasionally handles this differently too — "
            "`autoclip update-ytdlp` is worth a try, though it isn't the likely fix here.\n\n"
            f"Original error: {exc}"
        ),
    )


def _find_downloaded_file(directory: Path) -> Path | None:
    """Return the largest media file in a directory, ignoring sidecars."""
    candidates = [
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in ACCEPTED_SUFFIXES
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


# --------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------


def ingest_file(
    path: Path,
    *,
    move: bool = False,
    title: str | None = None,
    filename: str | None = None,
) -> Source:
    """Register a local file as a source, copying it into AutoClip's media store.

    Copying rather than referencing in place means a job stays reproducible even
    if the user moves or deletes the original.

    ``filename`` is the name the file arrived under. An upload is staged to a
    temporary file first, so without it the source would remember the staging
    name (``tmpvkq0d2me.mp4``) instead of what the user chose.

    Preconditions:
        path exists and is a readable media file.
    """
    path = Path(path)
    if not path.exists():
        raise IngestError(f"{path} does not exist.")
    if not path.is_file():
        raise IngestError(f"{path} is not a file.")

    suffix = path.suffix.lower()
    if suffix not in ACCEPTED_SUFFIXES:
        accepted = ", ".join(sorted(ACCEPTED_SUFFIXES))
        raise IngestError(
            f"{suffix or 'This file'} is not a supported format.",
            hint=f"Accepted formats: {accepted}",
        )

    source_id = new_id()
    target_dir = paths.source_media_dir(source_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"source{suffix}"

    try:
        if move:
            shutil.move(str(path), target)
        else:
            shutil.copy2(path, target)
    except OSError as exc:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise IngestError(f"Could not store {path.name}: {exc}") from exc

    try:
        info = _probe_and_validate(target)
    except Exception:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise

    return Source(
        id=source_id,
        type="upload",
        path=str(target),
        filename=filename or path.name,
        title=title or info.title or path.stem,
        duration_s=info.duration_s,
        width=info.width,
        height=info.height,
        fps=info.fps,
        has_audio=info.has_audio,
        has_video=info.has_video,
    )


def _probe_and_validate(path: Path) -> ffmpeg.MediaInfo:
    """Probe a media file and reject anything the pipeline can't process."""
    try:
        info = ffmpeg.probe(path)
    except ffmpeg.FFmpegError as exc:
        raise IngestError(f"{path.name} could not be read as media.", hint=str(exc)) from exc

    for kind, codec in (("video", info.video_codec), ("audio", info.audio_codec)):
        if codec and not ffmpeg.can_decode(codec):
            raise IngestError(
                f"This FFmpeg build cannot decode {codec.upper()} {kind}.",
                hint=(
                    f"{path.name} uses {codec} for its {kind}, and the ffmpeg on this "
                    "machine has no decoder for it. Install a full build (on Windows: "
                    "winget install Gyan.FFmpeg), or convert the file first:\n"
                    f'  ffmpeg -i "{path.name}" -c:v libx264 -c:a aac converted.mp4'
                ),
            )

    if not info.has_audio:
        raise IngestError(
            f"{path.name} has no audio track.",
            hint=(
                "AutoClip finds clips by transcribing speech, so a file with no audio "
                "has nothing to work from."
            ),
        )
    if info.duration_s <= 0:
        raise IngestError(
            f"{path.name} reports zero duration.",
            hint="The file may be corrupt or still being written.",
        )

    return info
