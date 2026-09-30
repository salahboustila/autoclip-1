"""Podcast Campaign Mode: every clip opens on the host's question."""

from __future__ import annotations

import json

import pytest
from autoclip import campaigns, config
from autoclip.campaigns import question_first as qf
from autoclip.campaigns import report
from autoclip.db import store
from autoclip.db.models import Job, Source, new_id
from autoclip.pipeline import highlights
from autoclip.pipeline import runner as runner_module
from autoclip.pipeline.boundaries import Boundary
from autoclip.pipeline.runner import PipelineRunner
from autoclip.pipeline.transcript import Transcript, Word
from autoclip.providers import DetectionConfig, detection_config
from autoclip.providers.base import (
    ClipCandidate,
    LLMProvider,
    ProviderStatus,
    load_prompt,
    render_system_prompt,
)

PRESET = campaigns.load_preset("jack_neel")
ANSWER = (
    "Honestly I made about four million dollars last year and most of it came from one deal. "
    "People think it was luck but it was ten years of work before anyone noticed. "
    "The first three years I made nothing at all and I almost quit twice. "
    "What changed was that I stopped selling my time and started selling a product. "
)


def build(parts: list[tuple[str, str | None]], *, pace: float = 0.35) -> Transcript:
    words: list[Word] = []
    t = 0.0
    for text, speaker in parts:
        for token in text.split():
            words.append(Word(text=token, start=t, end=t + pace * 0.9, speaker=speaker))
            t += pace
    return Transcript(words=words)


def index_of(transcript: Transcript, token: str, after: int = 0) -> int:
    return next(i for i, w in enumerate(transcript.words) if i >= after and w.text == token)


def policy() -> qf.QuestionFirstPolicy:
    return qf.QuestionFirstPolicy(PRESET)


class TestIsQuestion:
    @pytest.mark.parametrize(
        "text",
        [
            "How much money did you make last year?",
            "how much do you make",  # Whisper dropped the question mark
            "Do you think men have it harder",
            "So what would you tell your younger self",
            "Is that the biggest mistake?",
        ],
    )
    def test_questions(self, text: str) -> None:
        t = build([(text, None)])
        assert qf.is_question(t, 0, len(t.words) - 1)

    @pytest.mark.parametrize(
        "text",
        [
            "What I learned was patience.",
            "How I made it is simple.",
            "Do that and you will win!",
            "I made four million dollars.",
        ],
    )
    def test_statements(self, text: str) -> None:
        t = build([(text, None)])
        assert not qf.is_question(t, 0, len(t.words) - 1)


class TestHost:
    def test_the_speaker_asking_most_questions(self) -> None:
        t = build(
            [
                ("How did you start?", "S0"),
                ("I started young.", "S1"),
                ("Why money?", "S0"),
                ("Because I was broke. Do you know what that feels like?", "S1"),
                ("What was the first deal?", "S0"),
            ]
        )
        assert qf.identify_host(t) == "S0"

    def test_none_without_labels_or_enough_questions(self) -> None:
        assert qf.identify_host(build([("How did you start?", None)])) is None
        assert qf.identify_host(build([("How did you start?", "S0"), ("Young.", "S1")])) is None


class TestTrimStart:
    def test_a_clip_on_the_question_is_kept_as_is(self) -> None:
        t = build([("How much money did you make last year?", None), (ANSWER, None)])
        assert policy().trim_start(t, 0, len(t.words) - 1) == 0

    def test_lead_in_and_filler_are_trimmed(self) -> None:
        t = build(
            [
                ("Yeah that makes sense.", None),
                ("So um how much did you make?", None),
                (ANSWER, None),
            ]
        )
        start = policy().trim_start(t, 0, len(t.words) - 1)
        assert t.words[start].text == "how"

    def test_a_start_on_the_answer_moves_back_to_the_question(self) -> None:
        t = build([("How much did you make last year?", None), (ANSWER, None)])
        answer = index_of(t, "Honestly")
        assert policy().trim_start(t, answer, len(t.words) - 1) == 0

    def test_no_question_is_rejected_with_a_reason(self) -> None:
        t = build([(ANSWER, None), (ANSWER, None)])
        checker = policy()
        assert checker.trim_start(t, 0, len(t.words) - 1) is None
        assert checker.rejections[0]["reason"] == "does not open on a question"

    def test_a_question_far_into_the_clip_is_not_an_opener(self) -> None:
        t = build([(ANSWER, None), ("How much did you make?", None), (ANSWER, None)])
        assert policy().trim_start(t, 0, len(t.words) - 1) is None

    def test_a_long_rambling_question_is_rejected(self) -> None:
        question = "So " + "and then you went and did the other thing " * 5 + "why?"
        t = build([(question, None), (ANSWER, None)])
        checker = policy()
        assert checker.trim_start(t, 0, len(t.words) - 1) is None
        assert "answer starts late" in checker.rejections[0]["reason"]

    def test_a_question_with_no_answer_is_rejected(self) -> None:
        t = build([("How much did you make?", None), ("A lot.", None)])
        checker = policy()
        assert checker.trim_start(t, 0, len(t.words) - 1) is None
        assert "no answer" in checker.rejections[0]["reason"]


class TestHostCheck:
    def _episode(self, asker: str, answerer: str) -> Transcript:
        return build(
            [
                ("Why did you quit your job?", "HOST"),
                ("Because I hated it.", "GUEST"),
                ("What happened next?", "HOST"),
                ("I started a company.", "GUEST"),
                ("How much did you make last year?", asker),
                (ANSWER, answerer),
            ]
        )

    def _start(self, t: Transcript) -> int:
        return index_of(t, "How", after=10)

    def test_host_asks_guest_answers(self) -> None:
        t = self._episode("HOST", "GUEST")
        checker = policy()
        assert checker.trim_start(t, self._start(t), len(t.words) - 1) == self._start(t)
        assert checker.host_check(t) == qf.HOST_CHECK_VERIFIED

    def test_guest_asking_is_rejected(self) -> None:
        t = self._episode("GUEST", "HOST")
        # HOST asked the two earlier questions, so HOST is still identified as host.
        checker = policy()
        assert checker.trim_start(t, self._start(t), len(t.words) - 1) is None
        assert "not the host" in checker.rejections[0]["reason"]

    def test_host_answering_his_own_question_is_rejected(self) -> None:
        t = self._episode("HOST", "HOST")
        checker = policy()
        assert checker.trim_start(t, self._start(t), len(t.words) - 1) is None
        assert "answers his own question" in checker.rejections[0]["reason"]

    def test_without_labels_the_check_reports_it(self) -> None:
        t = build([("How much did you make?", None), (ANSWER, None)])
        assert policy().host_check(t) == qf.HOST_CHECK_NO_LABELS


class TestScore:
    def _boundary(self, t: Transcript) -> Boundary:
        return Boundary(start_s=0, end_s=t.words[-1].end, start_word=0, end_word=len(t.words) - 1)

    def test_criteria_combine_to_0_100(self) -> None:
        t = build([("How much did you make?", None), (ANSWER, None)])
        checker = policy()
        checker.trim_start(t, 0, len(t.words) - 1)
        candidate = ClipCandidate(
            start_word_index=0, end_word_index=5,
            question_hook=9, controversy=8, number_stat=10, clarity=7,
        )  # fmt: skip
        expected = round(10 * (0.35 * 9 + 0.30 * 8 + 0.20 * 7 + 0.15 * 10))
        assert checker.score(candidate, t, self._boundary(t)) == expected

    def test_a_number_the_guest_never_says_earns_nothing(self) -> None:
        t = build([("How did you feel?", None), ("Great, it felt amazing and free. " * 8, None)])
        checker = policy()
        checker.trim_start(t, 0, len(t.words) - 1)
        candidate = ClipCandidate(
            start_word_index=0, end_word_index=5,
            question_hook=9, controversy=8, number_stat=10, clarity=7,
        )  # fmt: skip
        expected = round(10 * (0.35 * 9 + 0.30 * 8 + 0.20 * 7))
        assert checker.score(candidate, t, self._boundary(t)) == expected

    def test_criteria_parse_on_0_10(self) -> None:
        candidate = ClipCandidate.model_validate(
            {"start_word_index": 0, "end_word_index": 5, "question_hook": 1, "clarity": "85"}
        )
        assert candidate.question_hook == 1.0  # not mistaken for a 0-1 fraction
        assert candidate.clarity == 8.5


class TestPrompt:
    def test_placeholders_are_filled(self) -> None:
        cfg = question_first_config()
        system = render_system_prompt(load_prompt(cfg.prompt_version), cfg)
        assert "<<" not in system
        assert "Jack Neel asking a question" in system
        assert "money, business, relationships" in system


def question_first_config() -> DetectionConfig:
    settings = config.Settings()
    settings.campaign.enabled = True
    settings = campaigns.apply(settings)
    return qf.configure(detection_config(settings), campaigns.rules_of(settings))


class FakeProvider(LLMProvider):
    name = "fake"
    requires_key = False

    def __init__(self, reply: dict) -> None:
        super().__init__("fake")
        self.reply = json.dumps(reply)
        self.calls = 0

    async def _complete(self, system: str, user: str, config: DetectionConfig) -> str:
        self.calls += 1
        return self.reply

    async def health_check(self) -> ProviderStatus:
        return ProviderStatus(name=self.name, available=True)


def episode() -> Transcript:
    return build(
        [
            (ANSWER, None),  # a guest monologue: no question
            (ANSWER, None),
            ("How much money did you make last year?", None),
            (ANSWER, None),
            (ANSWER, None),
        ]
    )


def candidate(start: int, end: int, **extra) -> dict:
    return {
        "start_word_index": start,
        "end_word_index": end,
        "topic": "Money",
        "question_hook": 9,
        "controversy": 8,
        "number_stat": 10,
        "clarity": 8,
        "score": 88,
        "reason": "Strong.",
        **extra,
    }


class TestDetect:
    async def test_only_question_first_clips_survive(self, tmp_path) -> None:
        t = episode()
        question = index_of(t, "How")
        provider = FakeProvider({"clips": [candidate(0, 110), candidate(question, question + 110)]})

        clips = await highlights.detect(
            t, provider, question_first_config(), job_id="j", trace_dir=tmp_path
        )

        assert len(clips) == 1
        assert clips[0].start_word == question
        assert clips[0].question_text == "How much money did you make last year?"
        assert 25 <= clips[0].duration_s <= 50
        trace = json.loads((tmp_path / "campaign_rejections.json").read_text())
        assert trace["rejections"][0]["reason"] == "does not open on a question"
        assert trace["host_check"] == "no_speaker_labels"

    async def test_nothing_passing_fails_clearly_without_a_fallback_pass(self) -> None:
        t = episode()
        provider = FakeProvider({"clips": [candidate(0, 110)]})

        with pytest.raises(highlights.HighlightError, match="No moment in this episode passed"):
            await highlights.detect(t, provider, question_first_config(), job_id="j")
        assert provider.calls == 1  # no second, low-confidence pass

    async def test_default_mode_ignores_the_campaign_rules(self) -> None:
        t = episode()
        provider = FakeProvider({"clips": [candidate(0, 110)]})

        clips = await highlights.detect(
            t, provider, detection_config(config.Settings()), job_id="j"
        )

        assert clips[0].start_word == 0
        assert clips[0].question_text == ""


class TestRunnerReport:
    async def test_highlights_stage_writes_selected_clips_json(
        self, initialised_db, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        t = episode()
        question = index_of(t, "How")
        highlight = {"clips": [candidate(question, question + 110)]}
        copy = {
            "titles": [{"text": "How Much Money Did You Make? 🤔", "score": 90}],
            "caption": "He made four million dollars last year. Could you?",
        }

        class Both(FakeProvider):
            async def _complete(self, system, user, config) -> str:
                return json.dumps(copy if "CLIP TRANSCRIPT" in user else highlight)

        monkeypatch.setattr(runner_module, "build_provider", lambda *a, **k: Both({}))
        settings = config.Settings()
        settings.campaign.enabled = True
        source = store.create_source(
            Source(id=new_id(), type="youtube", path="x.mp4", title="Ep", url="https://y/1")
        )
        job = store.create_job(Job(id=new_id(), source_id=source.id))

        clips = await PipelineRunner(job, source, settings=settings)._stage_highlights(t, [])

        stored = store.get_clip(clips[0].id)
        assert stored.question_text == "How much money did you make last year?"
        data = report.read(job.id)
        assert data["campaign"] == "jack_neel"
        assert data["host_check"] == "no_speaker_labels"
        entry = data["clips"][0]
        assert entry["question_text"] == "How much money did you make last year?"
        assert set(entry) >= {"start", "end", "score", "topic", "question_text", "reason"}
        assert entry["criteria"]["question_hook"] == 9
