"""YouTube ingest: telling a dropped connection apart from a yt-dlp problem."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yt_dlp
from autoclip import paths
from autoclip.config import IngestSettings
from autoclip.pipeline import ingest

#: What yt-dlp raised on a Windows laptop when the Wi-Fi blinked mid-request.
DNS_FAILURE = (
    "ERROR: [youtube] NqAdZpYmefU: Unable to download API page: HTTPSConnection("
    "host='www.youtube.com', port=443): Failed to resolve 'www.youtube.com' "
    "([Errno 11001] getaddrinfo failed)"
)
BOT_CHECK = "ERROR: [youtube] abc: Sign in to confirm you're not a bot"
URL = "https://www.youtube.com/watch?v=NqAdZpYmefU"


@pytest.fixture(autouse=True)
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record retry pauses instead of waiting, and skip probing the fake video."""
    pauses: list[float] = []
    monkeypatch.setattr(ingest.time, "sleep", pauses.append)
    monkeypatch.setattr(
        ingest,
        "_probe_and_validate",
        lambda path: SimpleNamespace(
            duration_s=60.0, width=1920, height=1080, fps=30.0, has_audio=True, has_video=True
        ),
    )
    return pauses


def fake_youtube_dl(monkeypatch: pytest.MonkeyPatch, errors: list[str]) -> list[str]:
    """Replace yt-dlp with a stand-in that raises ``errors`` in turn, then downloads."""
    attempts: list[str] = []

    class FakeYoutubeDL:
        def __init__(self, options: dict) -> None:
            self.target_dir = Path(options["outtmpl"]).parent

        def __enter__(self) -> FakeYoutubeDL:
            return self

        def __exit__(self, *exc_info: object) -> bool:
            return False

        def extract_info(self, url: str, download: bool) -> dict:
            attempts.append(url)
            if errors:
                raise yt_dlp.utils.DownloadError(errors.pop(0))
            (self.target_dir / "source.mp4").write_bytes(b"video")
            return {"title": "A talk", "duration": 60}

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYoutubeDL)
    return attempts


def test_a_dropped_connection_is_retried_until_it_comes_back(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    attempts = fake_youtube_dl(monkeypatch, [DNS_FAILURE, DNS_FAILURE])

    source = ingest.ingest_youtube(URL, IngestSettings())

    assert len(attempts) == 3
    assert sleeps == [3.0, 6.0]
    assert source.title == "A talk"
    assert Path(source.path).name == "source.mp4"


def test_a_connection_that_stays_down_is_reported_as_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = fake_youtube_dl(monkeypatch, [DNS_FAILURE] * ingest.NETWORK_ATTEMPTS)

    with pytest.raises(ingest.IngestError) as caught:
        ingest.ingest_youtube(URL, IngestSettings())

    assert len(attempts) == ingest.NETWORK_ATTEMPTS
    message = str(caught.value)
    assert message.startswith("Couldn't reach YouTube.")
    # Advice to update yt-dlp sent people after the wrong fix for a network drop.
    assert "update-ytdlp" not in message
    assert list(paths.media_dir().iterdir()) == []


def test_a_bot_check_is_reported_at_once_without_retrying(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    attempts = fake_youtube_dl(monkeypatch, [BOT_CHECK])

    with pytest.raises(ingest.IngestError, match="bot check"):
        ingest.ingest_youtube(URL, IngestSettings())

    assert len(attempts) == 1
    assert sleeps == []
