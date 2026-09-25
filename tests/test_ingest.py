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
#: YouTube's own player text when the signature/n-value JS challenge fails —
#: confirmed on https://youtu.be/mPQlJp7Rqr0 (2026-09-24).
CHALLENGE_FAILURE = "ERROR: [youtube] mPQlJp7Rqr0: The page needs to be reloaded"
COOKIE_DB_MISSING = "ERROR: could not find firefox cookies database in '/some/path'"
URL = "https://www.youtube.com/watch?v=NqAdZpYmefU"


#: Captured before the fixture below stubs it out, so the tests that are about
#: validation itself can still reach the real one.
_REAL_PROBE_AND_VALIDATE = ingest._probe_and_validate


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


def fake_youtube_dl(
    monkeypatch: pytest.MonkeyPatch, errors: list[str]
) -> tuple[list[str], list[dict]]:
    """Replace yt-dlp with a stand-in that raises ``errors`` in turn, then downloads.

    Returns the URL and the options dict yt-dlp was constructed with on each
    attempt, so a retry can be checked for exactly what changed about it.
    """
    attempts: list[str] = []
    options_per_attempt: list[dict] = []

    class FakeYoutubeDL:
        def __init__(self, options: dict) -> None:
            self.target_dir = Path(options["outtmpl"]).parent
            options_per_attempt.append(options)

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
    return attempts, options_per_attempt


def test_a_dropped_connection_is_retried_until_it_comes_back(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    attempts, _ = fake_youtube_dl(monkeypatch, [DNS_FAILURE, DNS_FAILURE])

    source = ingest.ingest_youtube(URL, IngestSettings())

    assert len(attempts) == 3
    assert sleeps == [3.0, 6.0]
    assert source.title == "A talk"
    assert Path(source.path).name == "source.mp4"


def test_a_connection_that_stays_down_is_reported_as_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts, _ = fake_youtube_dl(monkeypatch, [DNS_FAILURE] * ingest.NETWORK_ATTEMPTS)

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
    attempts, _ = fake_youtube_dl(monkeypatch, [BOT_CHECK])

    with pytest.raises(ingest.IngestError, match="bot check"):
        ingest.ingest_youtube(URL, IngestSettings())

    assert len(attempts) == 1
    assert sleeps == []


def test_every_download_allows_the_ejs_challenge_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    # Permission, not a fetch: yt-dlp only downloads the solver script when a
    # challenge actually needs it, so this costs nothing on ordinary downloads.
    _, options = fake_youtube_dl(monkeypatch, [])

    ingest.ingest_youtube(URL, IngestSettings())

    assert options[0]["remote_components"] == ["ejs:github"]


class TestChallengeFailure:
    """YouTube's "page needs to be reloaded" is a different failure than a bot
    check: it needs the EJS solver *and* real browser cookies together —
    confirmed manually on https://youtu.be/mPQlJp7Rqr0 (2026-09-24)."""

    def test_retries_once_with_firefox_cookies_when_none_were_configured(
        self, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
    ) -> None:
        # No browser configured yet — the common case, since this only comes up
        # the first time a video hits this failure.
        attempts, options = fake_youtube_dl(monkeypatch, [CHALLENGE_FAILURE])

        source = ingest.ingest_youtube(URL, IngestSettings())

        assert len(attempts) == 2
        assert sleeps == []  # not a network blip — no backoff before retrying
        assert "cookiesfrombrowser" not in options[0]
        assert options[1]["cookiesfrombrowser"] == ("firefox",)
        assert options[1]["remote_components"] == ["ejs:github"]
        assert source.title == "A talk"

    def test_the_fallback_browser_is_configurable_by_environment_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ingest.ENV_CHALLENGE_BROWSER, "edge")
        _, options = fake_youtube_dl(monkeypatch, [CHALLENGE_FAILURE])

        ingest.ingest_youtube(URL, IngestSettings())

        assert options[1]["cookiesfrombrowser"] == ("edge",)

    def test_a_second_challenge_failure_after_the_retry_is_reported_plainly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts, _ = fake_youtube_dl(monkeypatch, [CHALLENGE_FAILURE, CHALLENGE_FAILURE])

        with pytest.raises(ingest.IngestError) as caught:
            ingest.ingest_youtube(URL, IngestSettings())

        assert len(attempts) == 2  # no third attempt — one retry, not a loop
        message = str(caught.value)
        assert message.startswith("YouTube's challenge beat yt-dlp")
        assert "already retried" in message
        assert "github.com" in message

    def test_no_cookies_database_is_named_specifically(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Firefox not installed, or never opened — the retry's own failure, not
        # a second copy of the original error.
        fake_youtube_dl(monkeypatch, [CHALLENGE_FAILURE, COOKIE_DB_MISSING])

        with pytest.raises(ingest.IngestError) as caught:
            ingest.ingest_youtube(URL, IngestSettings())

        message = str(caught.value)
        assert message.startswith("Couldn't read browser cookies")
        assert "firefox" in message
        assert "never been opened" in message
        assert ingest.ENV_CHALLENGE_BROWSER in message

    def test_cookies_already_configured_are_not_retried_a_second_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Retrying with the same cookies that just failed would only fail the
        # same way again.
        attempts, options = fake_youtube_dl(monkeypatch, [CHALLENGE_FAILURE])
        settings = IngestSettings().model_copy(update={"cookies_from_browser": "firefox"})

        with pytest.raises(ingest.IngestError) as caught:
            ingest.ingest_youtube(URL, settings)

        assert len(attempts) == 1  # cookies were already there — no point retrying
        message = str(caught.value)
        assert "AutoClip sent yt-dlp's EJS challenge solver" in message
        assert "already retried" not in message


class TestBotCheckHint:
    """The hint used to offer "chrome" and "edge" on Windows, where yt-dlp cannot read
    either browser's cookies — so following it could only ever fail again."""

    def test_windows_is_pointed_at_firefox_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ingest.sys, "platform", "win32")

        hint = ingest._translate_ytdlp_error(Exception(BOT_CHECK), IngestSettings()).hint

        assert "firefox" in hint
        assert '"chrome"' not in hint
        assert '"edge"' not in hint

    def test_a_chromium_browser_already_set_is_explained(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ingest.sys, "platform", "win32")
        settings = IngestSettings().model_copy(update={"cookies_from_browser": "edge"})

        hint = ingest._translate_ytdlp_error(Exception(BOT_CHECK), settings).hint

        assert "cannot read" in hint
        assert "Firefox" in hint

    def test_elsewhere_the_full_list_still_stands(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ingest.sys, "platform", "linux")

        hint = ingest._translate_ytdlp_error(Exception(BOT_CHECK), IngestSettings()).hint

        assert '"chrome"' in hint
        assert '"firefox"' in hint


class TestUndecodableMedia:
    """An unplayable codec used to surface much later, in ffmpeg's own words, from
    the middle of a job. At import we can name the codec and the way out."""

    def _probed_as(
        self, monkeypatch: pytest.MonkeyPatch, video: str, audio: str, decodable: set[str]
    ) -> None:
        monkeypatch.setattr(
            ingest.ffmpeg,
            "probe",
            lambda path: SimpleNamespace(
                video_codec=video,
                audio_codec=audio,
                has_audio=True,
                duration_s=60.0,
                width=1920,
                height=1080,
                fps=60.0,
                title=None,
            ),
        )
        monkeypatch.setattr(ingest.ffmpeg, "can_decode", lambda codec: codec in decodable)

    def test_the_codec_and_a_way_out_are_named(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._probed_as(monkeypatch, "av1", "opus", decodable={"opus"})

        with pytest.raises(ingest.IngestError) as caught:
            _REAL_PROBE_AND_VALIDATE(tmp_path / "clip.mp4")

        assert "cannot decode AV1 video" in str(caught.value)
        assert "libx264" in caught.value.hint

    def test_a_decodable_file_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._probed_as(monkeypatch, "av1", "opus", decodable={"av1", "opus"})

        info = _REAL_PROBE_AND_VALIDATE(tmp_path / "clip.mp4")

        assert info.video_codec == "av1"


def test_an_upload_keeps_the_name_it_arrived_with(tmp_path: Path) -> None:
    # An upload is staged to a temp file, so without the original name the source
    # would remember "tmpvkq0d2me.mp4" and show that in the UI.
    staged = tmp_path / "tmpvkq0d2me.mp4"
    staged.write_bytes(b"video")

    source = ingest.ingest_file(
        staged, move=True, title="24 Hours in China", filename="24 Hours in China.mp4"
    )

    assert source.filename == "24 Hours in China.mp4"
    assert source.title == "24 Hours in China"
