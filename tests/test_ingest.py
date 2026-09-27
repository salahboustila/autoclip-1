"""YouTube ingest's handling of the user's cookies file."""

from __future__ import annotations

from pathlib import Path

import pytest
import yt_dlp
from autoclip.config import IngestSettings
from autoclip.pipeline import ingest

EXPORTED = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tLOGIN_INFO\tx\n"


class _RotatingYoutubeDL:
    """Stands in for yt-dlp: saves a pruned jar back, then fails like a bot check."""

    seen_cookiefile: str | None = None

    def __init__(self, options: dict) -> None:
        type(self).seen_cookiefile = options.get("cookiefile")

    def __enter__(self) -> _RotatingYoutubeDL:
        return self

    def __exit__(self, *exc: object) -> None:
        # Real yt-dlp saves the jar on close; a revoked session drops LOGIN_INFO.
        Path(self.seen_cookiefile).write_text("# Netscape HTTP Cookie File\n")

    def extract_info(self, url: str, download: bool) -> dict:
        raise yt_dlp.utils.DownloadError("Sign in to confirm you're not a bot")


def test_ytdlp_never_rewrites_the_exported_cookies(
    autoclip_home, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cookies = tmp_path / "youtube_cookies.txt"
    cookies.write_text(EXPORTED)
    monkeypatch.setattr(yt_dlp, "YoutubeDL", _RotatingYoutubeDL)

    with pytest.raises(ingest.IngestError):
        ingest.ingest_youtube(
            "https://www.youtube.com/watch?v=x", IngestSettings(cookies_file=str(cookies))
        )

    assert cookies.read_text() == EXPORTED
    assert _RotatingYoutubeDL.seen_cookiefile != str(cookies)
    assert not Path(_RotatingYoutubeDL.seen_cookiefile).exists()
