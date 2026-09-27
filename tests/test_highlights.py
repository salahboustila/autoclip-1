"""Highlight detection: windowing, per-window traces, the score cut-off, fallback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from autoclip.pipeline import highlights
from autoclip.pipeline.highlights import HighlightError, build_windows, detect
from autoclip.pipeline.transcript import Transcript, Word
from autoclip.providers import DetectionConfig
from autoclip.providers.base import LLMProvider, ProviderStatus


def _transcript(times: list[float]) -> Transcript:
    """One word per start time, 0.4 s long, a sentence ending every tenth word."""
    return Transcript(
        words=[
            Word(text="end." if i % 10 == 9 else "word", start=t, end=t + 0.4)
            for i, t in enumerate(times)
        ]
    )


def _steady(n: int, *, start: float = 0.0, pace: float = 0.5) -> list[float]:
    return [start + i * pace for i in range(n)]


class TestWindowing:
    def test_a_gap_longer_than_a_window_starts_a_fresh_window(self) -> None:
        # The "24 Hours in China" shape: 36 words, an 8.5-minute gap, then talk.
        times = _steady(36, start=4.0, pace=5.3) + _steady(2000, start=705.0)
        windows = build_windows(_transcript(times))

        assert (windows[0].first_word, windows[0].last_word) == (0, 35)
        assert windows[1].first_word == 36

    def test_every_window_reaches_past_the_previous_one(self) -> None:
        times = _steady(40) + _steady(40, start=600.0) + _steady(3000, start=1300.0)
        windows = build_windows(_transcript(times))

        for before, after in zip(windows, windows[1:], strict=False):
            assert after.last_word > before.last_word
        assert windows[-1].last_word == len(times) - 1

    def test_continuous_speech_still_overlaps(self) -> None:
        windows = build_windows(_transcript(_steady(3000)))

        assert len(windows) > 1
        for before, after in zip(windows, windows[1:], strict=False):
            assert after.first_word < before.last_word


class QueuedProvider(LLMProvider):
    """Replies by pass: `main` for the normal pass, `fallback` for the second."""

    name = "queued"
    requires_key = False

    def __init__(self, main: str | Exception, fallback: str | Exception = '{"clips": []}'):
        super().__init__("test-model")
        self.replies = {"main": main, "fallback": fallback}
        self.calls = {"main": 0, "fallback": 0}

    async def _complete(self, system: str, user: str, config: DetectionConfig) -> str:
        kind = "fallback" if config.fallback_clips else "main"
        self.calls[kind] += 1
        reply = self.replies[kind]
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def health_check(self) -> ProviderStatus:
        return ProviderStatus(name=self.name, available=True)


def _clip_json(*clips: tuple[int, int, int]) -> str:
    return json.dumps(
        {
            "clips": [
                {"start_word_index": s, "end_word_index": e, "title": f"Clip {s}", "score": score}
                for s, e, score in clips
            ]
        }
    )


@pytest.fixture
def transcript() -> Transcript:
    # 400 words at 0.5 s: one window, room for several 40-second clips.
    return _transcript(_steady(400))


async def _detect(transcript: Transcript, provider: LLMProvider, tmp_path: Path, **config):
    return await detect(
        transcript,
        provider,
        DetectionConfig(**config),
        job_id="job",
        trace_dir=tmp_path / "highlights",
    )


class TestTraces:
    async def test_each_window_writes_its_raw_reply(
        self, transcript: Transcript, tmp_path: Path
    ) -> None:
        reply = _clip_json((0, 79, 80))
        await _detect(transcript, QueuedProvider(reply), tmp_path)

        trace = json.loads((tmp_path / "highlights" / "main_00.json").read_text())
        assert trace["attempts"] == [{"raw": reply, "error": None}]
        assert trace["candidates"][0]["score"] == 80
        assert trace["error"] is None

    async def test_a_failed_window_records_both_replies_and_the_error(
        self, transcript: Transcript, tmp_path: Path
    ) -> None:
        with pytest.raises(HighlightError, match="Every transcript window failed"):
            await _detect(transcript, QueuedProvider("not json"), tmp_path)

        trace = json.loads((tmp_path / "highlights" / "main_00.json").read_text())
        assert [a["raw"] for a in trace["attempts"]] == ["not json", "not json"]
        assert "malformed clip data twice" in trace["error"]

    async def test_a_rerun_clears_the_previous_traces(
        self, transcript: Transcript, tmp_path: Path
    ) -> None:
        stale = tmp_path / "highlights" / "fallback_07.json"
        stale.parent.mkdir()
        stale.write_text("{}")

        await _detect(transcript, QueuedProvider(_clip_json((0, 79, 80))), tmp_path)

        assert not stale.exists()


class TestScoreCutoff:
    async def test_candidates_below_the_cutoff_are_dropped_in_code(
        self, transcript: Transcript, tmp_path: Path
    ) -> None:
        reply = _clip_json((0, 79, 80), (200, 279, 40))
        clips = await _detect(transcript, QueuedProvider(reply), tmp_path, min_score=50)

        assert [c.score for c in clips] == [80]
        assert not clips[0].low_confidence

    async def test_the_cutoff_is_configurable(self, transcript: Transcript, tmp_path: Path) -> None:
        reply = _clip_json((0, 79, 80), (200, 279, 40))
        clips = await _detect(transcript, QueuedProvider(reply), tmp_path, min_score=30)

        assert sorted(c.score for c in clips) == [40, 80]


class TestFallback:
    async def test_all_empty_windows_trigger_a_low_confidence_pass(
        self, transcript: Transcript, tmp_path: Path
    ) -> None:
        provider = QueuedProvider('{"clips": []}', _clip_json((0, 79, 35), (200, 279, 45)))
        clips = await _detect(transcript, provider, tmp_path)

        assert provider.calls == {"main": 1, "fallback": 1}
        assert [c.score for c in clips] == [45, 35]
        assert all(c.low_confidence for c in clips)
        assert (tmp_path / "highlights" / "fallback_00.json").exists()

    async def test_fallback_keeps_at_most_five(
        self, transcript: Transcript, tmp_path: Path
    ) -> None:
        many = _clip_json(*[(i * 50, i * 50 + 45, 30 + i) for i in range(7)])
        clips = await _detect(transcript, QueuedProvider('{"clips": []}', many), tmp_path)

        assert len(clips) == highlights.FALLBACK_MAX_CLIPS

    async def test_the_fallback_prompt_lifts_the_cutoff(
        self, transcript: Transcript, tmp_path: Path
    ) -> None:
        prompts: list[tuple[str, str]] = []

        class Recording(QueuedProvider):
            async def _complete(self, system: str, user: str, config: DetectionConfig) -> str:
                prompts.append((system, user))
                return await super()._complete(system, user, config)

        await _detect(transcript, Recording('{"clips": []}', _clip_json((0, 79, 35))), tmp_path)

        (main_system, _), (fallback_system, fallback_user) = prompts
        assert "Only return clips scoring 50 or higher" in main_system
        assert "second pass" in fallback_system
        assert "even if they score below 50" in fallback_user

    async def test_an_empty_fallback_still_fails_clearly(
        self, transcript: Transcript, tmp_path: Path
    ) -> None:
        with pytest.raises(HighlightError, match="even on a second pass"):
            await _detect(transcript, QueuedProvider('{"clips": []}'), tmp_path)

    async def test_no_fallback_when_every_window_errored(
        self, transcript: Transcript, tmp_path: Path
    ) -> None:
        provider = QueuedProvider(RuntimeError("401 invalid x-api-key"))

        with pytest.raises(HighlightError, match="Every transcript window failed"):
            await _detect(transcript, provider, tmp_path)
        assert provider.calls["fallback"] == 0
