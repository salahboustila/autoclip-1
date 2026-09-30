"""Metadata-only YouTube lookups for campaigns: channel listings, heatmaps.

Nothing here downloads media. Results are cached under
``<root>/campaigns/cache/`` so a campaign doesn't hit YouTube on every job,
and every call degrades to "unknown" rather than failing a job — a campaign
rule that can't be checked is reported, not silently passed or fatal.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .. import paths
from ..config import IngestSettings

log = logging.getLogger(__name__)

CHANNEL_TTL_S = 30 * 60
SHORTS_TTL_S = 12 * 60 * 60
INFO_TTL_S = 24 * 60 * 60

_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


class LookupError_(RuntimeError):
    """A YouTube metadata lookup failed."""


def video_id(url: str | None) -> str | None:
    """The 11-character video id in a watch, youtu.be, shorts or live URL."""
    if not url:
        return None
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()
    candidate = None
    if host.endswith("youtu.be"):
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif "youtube.com" in host:
        if parsed.path == "/watch":
            candidate = (parse_qs(parsed.query).get("v") or [None])[0]
        else:
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0] in ("shorts", "live", "embed", "v"):
                candidate = parts[1]
    return candidate if candidate and _ID.match(candidate) else None


def _cache_path(kind: str, key: str) -> Path:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return paths.root() / "campaigns" / "cache" / f"{kind}-{digest}.json"


def _cached(kind: str, key: str, ttl_s: float, fetch):
    path = _cache_path(kind, key)
    if path.is_file() and time.time() - path.stat().st_mtime < ttl_s:
        with contextlib.suppress(OSError, json.JSONDecodeError):
            return json.loads(path.read_text(encoding="utf-8"))
    value = fetch()
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    return value


def _extract(url: str, *, flat: bool, limit: int | None, settings: IngestSettings | None) -> dict:
    """One yt-dlp metadata call. Tests replace this so nothing touches the network."""
    import yt_dlp

    settings = settings or IngestSettings()
    options: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noprogress": True,
    }
    if flat:
        options["extract_flat"] = "in_playlist"
    if limit:
        options["playlistend"] = limit
    if settings.cookies_from_browser:
        options["cookiesfrombrowser"] = (settings.cookies_from_browser,)
    cookie_copy: Path | None = None
    if settings.cookies_file and Path(settings.cookies_file).expanduser().is_file():
        # Same reason as ingest: yt-dlp rewrites the jar, so give it a copy.
        fd, name = tempfile.mkstemp(prefix="autoclip-cookies-", suffix=".txt")
        os.close(fd)
        cookie_copy = Path(name)
        shutil.copyfile(Path(settings.cookies_file).expanduser(), cookie_copy)
        options["cookiefile"] = str(cookie_copy)
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            return ydl.extract_info(url, download=False) or {}
    except Exception as exc:
        raise LookupError_(f"YouTube lookup failed for {url}: {exc}") from exc
    finally:
        if cookie_copy is not None:
            cookie_copy.unlink(missing_ok=True)


def channel_videos(
    url: str, limit: int, *, settings: IngestSettings | None = None, ttl_s: float = CHANNEL_TTL_S
) -> list[dict[str, Any]]:
    """The newest ``limit`` entries of a channel tab, newest first.

    Each entry is ``{"id", "title", "duration_s", "url"}``.
    """

    def fetch() -> list[dict[str, Any]]:
        info = _extract(url, flat=True, limit=limit, settings=settings)
        entries = []
        for entry in (info.get("entries") or [])[:limit]:
            vid = entry.get("id")
            if not vid:
                continue
            entries.append(
                {
                    "id": vid,
                    "title": entry.get("title") or "",
                    "duration_s": entry.get("duration"),
                    "url": f"https://www.youtube.com/watch?v={vid}",
                }
            )
        return entries

    return _cached("channel", f"{url}|{limit}", ttl_s, fetch)


def replay_heatmap(url: str, *, settings: IngestSettings | None = None) -> list[dict[str, float]]:
    """YouTube's "most replayed" curve, or [] when YouTube doesn't expose it."""

    def fetch() -> list[dict[str, float]]:
        info = _extract(url, flat=False, limit=None, settings=settings)
        return [
            {
                "start": float(point["start_time"]),
                "end": float(point["end_time"]),
                "value": float(point["value"]),
            }
            for point in info.get("heatmap") or []
            if {"start_time", "end_time", "value"} <= set(point)
        ]

    return _cached("heatmap", url, INFO_TTL_S, fetch)
