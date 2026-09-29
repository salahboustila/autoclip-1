"""Viral Hook Mode: the prompt, four-criterion scoring, opening rules, wiring."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from autoclip import app as app_module
from autoclip import config
from autoclip.app import create_app
from autoclip.db import store
from autoclip.db.models import Clip, Job, Source, new_id
from autoclip.pipeline import highlights, viral
from autoclip.pipeline.runner import settings_for_job
from autoclip.pipeline.transcript import Transcript, Word
from autoclip.providers import DetectionConfig, detection_config
from autoclip.providers.base import (
    ClipCandidate,
    LLMProvider,
    ProviderStatus,
    load_prompt,
    render_system_prompt,
)
from fastapi.testclient import TestClient


def make_transcript(sentences: list[tuple[str, str | None]], *, pace: float = 0.4) -> Transcript:
    """Words from ``(sentence, speaker)`` pairs, spoken at a steady pace."""
    words: list[Word] = []
    t = 0.0
    for sentence, speaker in sentences:
        for token in sentence.split():
            words.append(Word(text=token, start=t, end=t + pace * 0.9, speaker=speaker))
            t += pace
    return Transcript(words=words)


def index_of(transcript: Transcript, text: str) -> int:
    return next(i for i, w in enumerate(transcript.words) if w.text == text)


BODY = " ".join(f"Point number {i} matters a lot to everyone here." for i in range(12))


class TestCandidateParsing:
    def test_viral_fields_parse_and_coerce(self) -> None:
        candidate = ClipCandidate.model_validate(
            {
                "start_word_index": 1,
                "end_word_index": 9,
                "topic": "Money",
                "hook_strength": 0.9,
                "comment_potential": "80",
                "standalone_clarity": None,
                "has_number": "yes",
            }
        )
        assert candidate.topic == "Money"
        assert candidate.hook_strength == 90
        assert candidate.comment_potential == 80
        assert candidate.standalone_clarity is None
        assert candidate.has_number is True

    def test_default_replies_leave_them_empty(self) -> None:
        candidate = ClipCandidate(start_word_index=1, end_word_index=5, score=70)
        assert candidate.topic == ""
        assert candidate.hook_strength is None


class TestScoring:
    def _candidate(self, **kwargs) -> ClipCandidate:
        return ClipCandidate(start_word_index=0, end_word_index=10, **kwargs)

    def test_weights_sum_to_one(self) -> None:
        assert sum(viral.WEIGHTS.values()) == pytest.approx(1.0)

    def test_combines_the_four_criteria(self) -> None:
        candidate = self._candidate(
            hook_strength=90, comment_potential=80, standalone_clarity=70, has_number=True
        )
        expected = round(0.35 * 90 + 0.30 * 80 + 0.20 * 70 + 0.15 * 100)
        assert viral.combined_score(candidate, "He makes 50 thousand a year.") == expected

    def test_a_claimed_number_the_speaker_never_says_earns_nothing(self) -> None:
        candidate = self._candidate(
            hook_strength=90, comment_potential=80, standalone_clarity=70, has_number=True
        )
        with_number = viral.combined_score(candidate, "Most men earn 40 percent less.")
        without = viral.combined_score(candidate, "Most men earn far less than you think.")
        assert with_number - without == pytest.approx(15, abs=1)

    def test_no_sub_scores_keeps_the_model_score(self) -> None:
        assert viral.combined_score(self._candidate(score=64), "anything") == 64

    @pytest.mark.parametrize(
        "text", ["It was 69 percent.", "About $100k.", "Half of them.", "A 10% rise."]
    )
    def test_detects_numbers(self, text: str) -> None:
        assert viral.mentions_number(text)

    def test_ordinary_text_has_no_number(self) -> None:
        assert not viral.mentions_number("Women want a man who leads, not follows.")


class TestOpening:
    def test_leading_filler_words_are_dropped(self) -> None:
        t = make_transcript([("So yeah, um, the average man makes very little money.", None)])
        t.words.extend(make_transcript([(BODY, None)]).words)
        assert t.words[viral.trim_opening(t, 0, len(t.words) - 1)].text == "the"

    def test_filler_only_sentences_are_skipped(self) -> None:
        t = make_transcript(
            [
                ("Yeah. Exactly. Right.", None),
                ("Women rate most men below average.", None),
                (BODY, None),
            ]
        )
        assert t.words[viral.trim_opening(t, 0, len(t.words) - 1)].text == "Women"

    def test_the_hosts_question_is_skipped_with_diarization(self) -> None:
        t = make_transcript(
            [
                ("What do women actually want?", "HOST"),
                ("Women want a man who earns more than them.", "GUEST"),
                (BODY, "GUEST"),
            ]
        )
        assert t.words[viral.trim_opening(t, 0, len(t.words) - 1)].text == "Women"

    def test_the_main_speakers_own_question_is_kept_as_a_hook(self) -> None:
        t = make_transcript([("How much does the average man make?", "GUEST"), (BODY, "GUEST")])
        assert viral.trim_opening(t, 0, len(t.words) - 1) == 0

    def test_a_clean_start_is_untouched(self) -> None:
        t = make_transcript([("Sixty nine percent voted not guilty.", None), (BODY, None)])
        assert viral.trim_opening(t, 0, len(t.words) - 1) == 0

    def test_never_trims_more_than_a_third_of_the_clip(self) -> None:
        t = make_transcript([("so " * 30 + "okay.", None), ("Real content starts here.", None)])
        end = len(t.words) - 1
        assert viral.trim_opening(t, 0, end) <= int((end + 1) * viral.MAX_TRIM_SHARE)


class TestDetectionConfig:
    def test_default_path_is_unchanged(self) -> None:
        settings = config.Settings()
        cfg = detection_config(settings)
        assert cfg.prompt_version == "highlight_v1"
        assert cfg.viral_hook is False
        assert (cfg.min_duration_s, cfg.max_duration_s, cfg.max_clips) == (20.0, 90.0, 10)

    def test_viral_mode(self) -> None:
        settings = config.Settings()
        settings.viral_hook.enabled = True
        settings.viral_hook.top_n = 7
        settings.clips.min_score = 60
        cfg = detection_config(settings)
        assert cfg.prompt_version == "highlight_viral_v1"
        assert cfg.viral_hook is True
        assert (cfg.min_duration_s, cfg.max_duration_s, cfg.max_clips) == (25.0, 50.0, 7)
        assert cfg.min_score == 60


class TestPrompt:
    @pytest.mark.parametrize("fallback", [0, 3])
    def test_renders_with_every_placeholder_filled(self, fallback: int) -> None:
        cfg = DetectionConfig(prompt_version=viral.VIRAL_PROMPT, fallback_clips=fallback)
        system = render_system_prompt(load_prompt(viral.VIRAL_PROMPT), cfg)
        assert "<<" not in system
        for phrase in ("hook_strength", "comment_potential", "standalone_clarity", "has_number"):
            assert phrase in system
        assert "25 to 50 seconds" in system


class FakeProvider(LLMProvider):
    name = "fake"
    requires_key = False

    def __init__(self, reply: dict) -> None:
        super().__init__("fake")
        self.reply = json.dumps(reply)
        self.systems: list[str] = []

    async def _complete(self, system: str, user: str, config: DetectionConfig) -> str:
        self.systems.append(system)
        return self.reply

    async def health_check(self) -> ProviderStatus:
        return ProviderStatus(name=self.name, available=True)


@pytest.fixture
def podcast() -> Transcript:
    # ~100 s of speech: filler opener, a strong claim, then a long answer.
    return make_transcript(
        [
            ("So yeah, um, okay.", "GUEST"),
            ("The average man makes 50 thousand dollars a year.", "GUEST"),
            (BODY + " " + BODY, "GUEST"),
        ],
        pace=0.4,
    )


class TestDetect:
    def _reply(self, start: int, end: int, **scores) -> dict:
        return {
            "clips": [
                {
                    "start_word_index": start,
                    "end_word_index": end,
                    "topic": "Money",
                    "title": "What the average man makes",
                    "hook": "So yeah",
                    "score": 80,
                    "reason": "Specific income figure.",
                    **scores,
                }
            ]
        }

    async def test_viral_detection_end_to_end(self, podcast: Transcript) -> None:
        cfg = detection_config(_viral_settings())
        provider = FakeProvider(
            self._reply(
                0,
                100,
                hook_strength=90,
                comment_potential=85,
                standalone_clarity=80,
                has_number=True,
            )
        )

        clips = await highlights.detect(podcast, provider, cfg, job_id="j")

        assert "Start on the strongest sentence" in provider.systems[0]
        (clip,) = clips
        assert podcast.words[clip.start_word].text == "The"
        assert 25.0 <= clip.duration_s <= 50.0
        assert clip.topic == "Money"
        assert clip.score == round(0.35 * 90 + 0.30 * 85 + 0.20 * 80 + 15)

    async def test_min_score_applies_to_the_combined_score(self, podcast: Transcript) -> None:
        settings = _viral_settings()
        settings.clips.min_score = 70
        cfg = detection_config(settings)
        # Model says 80 overall, but its own criteria combine to ~45.
        provider = FakeProvider(
            self._reply(
                0,
                100,
                hook_strength=50,
                comment_potential=50,
                standalone_clarity=60,
                has_number=False,
            )
        )

        clips = await highlights.detect(podcast, provider, cfg, job_id="j")

        # Nothing clears 70, so the fallback pass returns it as low-confidence.
        assert all(clip.low_confidence for clip in clips)

    async def test_default_mode_ignores_viral_fields(self, podcast: Transcript) -> None:
        cfg = detection_config(config.Settings())
        provider = FakeProvider(self._reply(0, 100, hook_strength=10, has_number=True))

        (clip,) = await highlights.detect(podcast, provider, cfg, job_id="j")

        assert clip.score == 80
        assert clip.topic == ""
        assert clip.start_word == 0
        assert "Start on the strongest sentence" not in provider.systems[0]


def _viral_settings() -> config.Settings:
    settings = config.Settings()
    settings.viral_hook.enabled = True
    return settings


class TestJobWiring:
    @pytest.fixture
    def client(self, autoclip_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
        monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
        with TestClient(create_app()) as test_client:
            yield test_client

    def test_per_job_toggle_reaches_the_runner(self, client: TestClient) -> None:
        source = store.create_source(Source(id=new_id(), type="upload", path="x.mp4"))
        body = client.post(
            "/api/jobs", json={"source_id": source.id, "settings": {"viral_hook": True}}
        ).json()

        job = store.get_job(body["id"])
        assert job.settings["viral_hook"]["enabled"] is True
        assert settings_for_job(job).viral_hook.enabled is True
        assert config.load().viral_hook.enabled is False  # saved settings untouched

    def test_jobs_without_a_snapshot_use_saved_settings(self, initialised_db) -> None:
        saved = config.Settings()
        saved.clips.max_clips = 4
        config.save(saved)
        job = Job(id=new_id(), source_id="s", settings={})
        assert settings_for_job(job).clips.max_clips == 4

    def test_snapshot_keeps_plaintext_fallback_secrets(self, initialised_db, no_keyring) -> None:
        config.set_secret("anthropic", "sk-test")
        job = Job(id=new_id(), source_id="s", settings=config.Settings().model_dump(mode="json"))
        assert config.get_secret("anthropic", settings_for_job(job)) == "sk-test"

    def test_topic_round_trips(self, client: TestClient) -> None:
        source = store.create_source(Source(id=new_id(), type="upload", path="x.mp4"))
        job = store.create_job(Job(id=new_id(), source_id=source.id))
        clip = Clip(id=new_id(), job_id=job.id, start_s=0, end_s=30, topic="Dating")
        store.replace_clips(job.id, [clip])
        assert client.get(f"/api/clips/{clip.id}").json()["topic"] == "Dating"

    def test_viral_settings_round_trip(self, client: TestClient) -> None:
        body = client.put("/api/settings", json={"viral_hook": {"top_n": 5}}).json()
        assert body["viral_hook"]["top_n"] == 5
        assert body["viral_hook"]["enabled"] is False
