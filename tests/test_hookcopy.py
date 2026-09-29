"""Viral Hook Mode copy: title options, grounding, captions, files, wiring."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from autoclip import app as app_module
from autoclip import config, paths
from autoclip.app import create_app
from autoclip.db import store
from autoclip.db.models import Clip, Export, Job, Source, new_id
from autoclip.pipeline import export, ffmpeg, hookcopy
from autoclip.pipeline import runner as runner_module
from autoclip.pipeline.hookcopy import TitleOption
from autoclip.pipeline.runner import PipelineRunner
from autoclip.pipeline.transcript import Transcript, Word
from autoclip.providers import DetectionConfig
from autoclip.providers.base import LLMProvider, ProviderStatus
from fastapi.testclient import TestClient

CLIP = (
    "I'm a multimillionaire, and I read a study that blew my mind. "
    "Sixty nine percent of people voted not guilty. "
    "Men with 10 partners go up in value, while women plummet. "
    "The average American man makes fifty thousand dollars a year. "
    "If you want a high value man, you have to learn this."
)

EXAMPLES = [
    "How Much Does the Average American Man Make? 🤔",
    "Multi-Millionaire Reveals Shocking Study on Women 💰",
    "If You Want a High-Value Man, Learn This 👀",
    "69% Voted Not Guilty For This?! 😱",
    "10+ Partners: Men Go Up, Women Plummet 😳",
]


class TestNumbers:
    @pytest.mark.parametrize(
        ("text", "value"),
        [
            ("69%", 69),
            ("sixty-nine percent", 69),
            ("sixty nine", 69),
            ("$50,000", 50_000),
            ("50k", 50_000),
            ("fifty thousand dollars", 50_000),
            ("a hundred men", 100),
            ("10+ partners", 10),
            ("2.5 million", 2_500_000),
            ("half of them", 50),
        ],
    )
    def test_extracts(self, text: str, value: float) -> None:
        assert value in hookcopy.extract_numbers(text)

    def test_plain_text_has_none(self) -> None:
        assert hookcopy.extract_numbers("Women want a man who leads.") == set()


class TestGrounding:
    @pytest.mark.parametrize("title", EXAMPLES)
    def test_every_pattern_passes_when_the_clip_supports_it(self, title: str) -> None:
        assert hookcopy.check_grounding(title, CLIP).ok, hookcopy.check_grounding(title, CLIP)

    def test_a_number_the_speaker_never_says_is_rejected(self) -> None:
        result = hookcopy.check_grounding("80% of Women Agree 😱", CLIP)
        assert not result.ok
        assert result.unsupported_numbers == [80]

    def test_a_study_nobody_mentions_is_rejected(self) -> None:
        clip = "Women want a man who earns more than they do."
        result = hookcopy.check_grounding("Shocking Study on Women 😳", clip)
        assert result.unsupported_words == ["study"]

    def test_an_authority_nobody_states_is_rejected(self) -> None:
        clip = "Women want a man who earns more than they do."
        assert not hookcopy.check_grounding("Doctor Reveals What Women Want 😳", clip).ok

    def test_speaker_facts_can_supply_the_authority(self) -> None:
        clip = "Women want a man who earns more than they do."
        title = "Multi-Millionaire Reveals What Women Want 💰"
        assert not hookcopy.check_grounding(title, clip).ok
        assert hookcopy.check_grounding(title, clip, "He is a multi-millionaire.").ok

    def test_a_verdict_the_speaker_never_gives_is_rejected(self) -> None:
        clip = "The jury voted after two days of deliberation."
        assert not hookcopy.check_grounding("Jury Voted Not Guilty 😱", clip).ok

    def test_synonyms_count_but_only_within_their_group(self) -> None:
        clip = "New research shows most girls prefer taller guys."
        assert hookcopy.check_grounding("Study: Women Prefer Taller Men 📊", clip).ok
        assert not hookcopy.check_grounding("Poll: Women Prefer Taller Men 📊", clip).ok

    def test_one_as_phrasing_is_not_a_statistic(self) -> None:
        assert hookcopy.check_grounding("The One Thing Women Want 👀", "Women want respect.").ok

    def test_word_forms_match(self) -> None:
        assert (
            hookcopy.check_grounding("Why Dating Is Harder 💔", "Dates are hard now.").ok is False
        )
        assert hookcopy.check_grounding("Why Dating Is Hard 💔", "Dates are hard now.").ok


class TestFormat:
    def test_title_case_with_small_words(self) -> None:
        assert hookcopy.title_case("how much does the average man make?") == (
            "How Much Does the Average Man Make?"
        )

    def test_all_caps_is_kept(self) -> None:
        assert hookcopy.normalise_title("MEN GO UP 😳") == "MEN GO UP 😳"

    def test_exactly_one_emoji_at_the_end(self) -> None:
        assert hookcopy.normalise_title("💰 money 😱 talk 😳") == "Money Talk 😳"

    def test_missing_emoji_gets_the_topics(self) -> None:
        assert hookcopy.normalise_title("Money Talk", topic="Money") == "Money Talk 💰"
        assert hookcopy.normalise_title("Money Talk") == "Money Talk 🤔"

    def test_more_than_eight_words_is_rejected(self) -> None:
        title, reason = hookcopy.validate_option(
            "If You Want a High-Value Man You Have to Learn This", clip_text=CLIP
        )
        assert title is None
        assert "8 words" in reason

    def test_numbers_and_acronyms_keep_their_form(self) -> None:
        assert hookcopy.title_case("69% of USA men make $50K") == "69% of USA Men Make $50K"


class TestChoosing:
    def test_best_first_invalid_dropped_and_topped_up(self) -> None:
        options = [
            TitleOption("Men Go Up, Women Plummet 😳", 70),
            TitleOption("Doctor Says 80% Lie 😱", 99),  # ungrounded
            TitleOption("How Much Does the Average American Man Make? 🤔", 90),
        ]
        chosen, rejected = hookcopy.choose_titles(
            options, clip_text=CLIP, fallbacks=["Fallback Title Here 🤔"]
        )
        assert [o.text for o in chosen] == [
            "How Much Does the Average American Man Make? 🤔",
            "Men Go Up, Women Plummet 😳",
            "Fallback Title Here 🤔",
        ]
        assert chosen[2].source == "fallback"
        assert rejected[0]["text"] == "Doctor Says 80% Lie 😱"

    def test_duplicates_collapse(self) -> None:
        options = [TitleOption("Men Go Up 😳", 80), TitleOption("men go up 😳", 70)]
        chosen, _ = hookcopy.choose_titles(options, clip_text=CLIP, fallbacks=[])
        assert len(chosen) == 1

    def test_fallbacks_are_verbatim_and_grounded(self) -> None:
        transcript = _transcript(CLIP)
        titles = hookcopy.fallback_titles(transcript, 0, len(transcript.words) - 1)
        assert titles
        for title in titles:
            assert hookcopy.word_count(title) <= hookcopy.MAX_TITLE_WORDS
            assert hookcopy.check_grounding(title, CLIP).ok, title


class TestCaption:
    def _caption(self, line: str, tags: list[str], **kwargs) -> str:
        return hookcopy.build_caption(
            line,
            tags,
            title="Men Go Up, Women Plummet 😳",
            clip_text=CLIP,
            fixed_hashtags=["#usa", "#uk", "#canada"],
            **kwargs,
        )

    def test_structure(self) -> None:
        caption = self._caption(
            "He says men go up and women plummet. Is he right?", ["#dating", "#Men", "women"]
        )
        line, footer = caption.split("\n\n")
        assert line == "He says men go up and women plummet. Is he right?"
        assert footer == "@michaelsartain #dating #men #women #usa #uk #canada"

    def test_hashtags_are_capped_at_five_and_topped_up_to_three(self) -> None:
        many = self._caption("Right?", [f"#t{i}" for i in range(9)])
        few = self._caption("Right?", ["#dating"], topic="Dating")
        count = lambda c: len([t for t in c.split() if t.startswith("#")]) - 3  # noqa: E731
        assert count(many) == 5
        assert count(few) == 3

    def test_fixed_hashtags_are_not_repeated(self) -> None:
        caption = self._caption("Right?", ["#usa", "#dating", "#men", "#women"])
        assert caption.count("#usa") == 1

    def test_a_line_without_a_question_is_rebuilt_from_the_title(self) -> None:
        caption = self._caption("Men go up and women plummet.", ["#dating"])
        assert caption.startswith("Men Go Up, Women Plummet. Do you agree?")

    def test_a_line_with_an_invented_number_is_rebuilt(self) -> None:
        caption = self._caption("90% of men agree. Do you?", ["#dating"])
        assert "90%" not in caption

    def test_stray_tags_on_the_line_are_stripped(self) -> None:
        caption = self._caption("Is he right? #dating @someone", ["#dating"])
        assert caption.split("\n\n")[0] == "Is he right?"

    def test_file_is_named_by_rank_and_removed_when_empty(self, tmp_path: Path) -> None:
        path = hookcopy.write_caption_file(tmp_path, 3, "Hello?\n\n#x")
        assert path == tmp_path / "clip_03_caption.txt"
        assert path.read_text(encoding="utf-8") == "Hello?\n\n#x\n"
        assert hookcopy.write_caption_file(tmp_path, 3, "") is None
        assert not path.exists()


class FakeProvider(LLMProvider):
    name = "fake"
    requires_key = False

    def __init__(self, reply: str | Exception) -> None:
        super().__init__("fake")
        self.reply = reply
        self.users: list[str] = []

    async def _complete(self, system: str, user: str, config: DetectionConfig) -> str:
        self.users.append(user)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    async def health_check(self) -> ProviderStatus:
        return ProviderStatus(name=self.name, available=True)


GOOD_REPLY = json.dumps(
    {
        "titles": [
            {"text": "Men Go Up, Women Plummet 😳", "pattern": 5, "score": 80},
            {"text": "How Much Does the Average American Man Make? 🤔", "pattern": 1, "score": 90},
            {"text": "Therapist Reveals 80% of Women Lie 😱", "pattern": 2, "score": 99},
        ],
        "caption": "He says the average man makes fifty thousand. Is that enough?",
        "hashtags": ["#money", "#dating", "#men"],
    }
)


class TestGenerate:
    async def test_happy_path(self) -> None:
        provider = FakeProvider(GOOD_REPLY)
        copy = await hookcopy.generate(provider, CLIP, fallbacks=["Fallback Title Now 🤔"])

        assert copy.title == "How Much Does the Average American Man Make? 🤔"
        assert copy.alternatives == ["Men Go Up, Women Plummet 😳", "Fallback Title Now 🤔"]
        assert copy.rejected[0]["text"].startswith("Therapist")
        assert copy.caption.startswith("He says the average man makes fifty thousand.")
        assert "CLIP TRANSCRIPT" in provider.users[0] and CLIP in provider.users[0]

    @pytest.mark.parametrize("reply", ["not json at all", RuntimeError("rate limited")])
    async def test_model_failure_falls_back_to_the_clips_own_words(self, reply) -> None:
        transcript = _transcript(CLIP)
        fallbacks = hookcopy.fallback_titles(transcript, 0, len(transcript.words) - 1)

        copy = await hookcopy.generate(FakeProvider(reply), CLIP, fallbacks=fallbacks)

        assert copy.title == fallbacks[0]
        assert len(copy.alternatives) == 2
        assert copy.caption.split("\n\n")[0].endswith("Do you agree?")
        assert "#usa" not in copy.caption  # no fixed tags unless configured

    async def test_facts_are_sent_only_when_given(self) -> None:
        provider = FakeProvider(GOOD_REPLY)
        await hookcopy.generate(provider, CLIP, fallbacks=[], facts="He founded two companies.")
        await hookcopy.generate(provider, CLIP, fallbacks=[])
        assert "He founded two companies." in provider.users[0]
        assert "Do not describe the speaker" in provider.users[1]

    async def test_a_title_the_user_already_set_is_kept(self, tmp_path: Path) -> None:
        transcript = _transcript(CLIP)
        clip = Clip(
            id="c", job_id="j", start_s=0, end_s=30, start_word=0,
            end_word=len(transcript.words) - 1, rank=1, hook_title="My Own Title",
        )  # fmt: skip
        await hookcopy.generate_for_clips(
            [clip], transcript, FakeProvider(GOOD_REPLY), trace_dir=tmp_path
        )
        assert clip.hook_title == "My Own Title"
        assert clip.post_caption
        assert (tmp_path / "copy_01.json").exists()


class TestRunnerAndApi:
    @pytest.fixture
    def client(self, autoclip_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
        monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
        with TestClient(create_app()) as test_client:
            yield test_client

    async def test_viral_highlights_stage_fills_titles_and_captions(
        self, initialised_db, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transcript = _transcript(CLIP + " " + " ".join(["Point made here clearly."] * 30))
        highlight_reply = json.dumps(
            {
                "clips": [
                    {"start_word_index": 0, "end_word_index": len(transcript.words) - 1,
                     "topic": "Money", "score": 90, "hook_strength": 90,
                     "comment_potential": 90, "standalone_clarity": 90, "has_number": True}
                ]
            }
        )  # fmt: skip

        class Both(FakeProvider):
            async def _complete(self, system, user, config) -> str:
                return GOOD_REPLY if "CLIP TRANSCRIPT" in user else highlight_reply

        monkeypatch.setattr(runner_module, "build_provider", lambda *a, **k: Both(""))
        settings = config.Settings()
        settings.viral_hook.enabled = True
        source = store.create_source(Source(id=new_id(), type="upload", path="x.mp4"))
        job = store.create_job(Job(id=new_id(), source_id=source.id))

        clips = await PipelineRunner(job, source, settings=settings)._stage_highlights(
            transcript, []
        )

        stored = store.get_clip(clips[0].id)
        assert stored.hook_title == "How Much Does the Average American Man Make? 🤔"
        assert len(stored.hook_title_alts) == 2
        assert stored.post_caption.endswith("#usa #uk #canada")
        assert "@michaelsartain" in stored.post_caption

    def test_default_mode_writes_no_copy(self) -> None:
        assert export.output_filename("Hi", "9:16") == "hi_9x16.mp4"
        assert export.output_filename("Hi", "9:16", rank=3) == "clip_03_hi_9x16.mp4"

    def test_patching_the_caption_rewrites_the_file(self, client: TestClient) -> None:
        source = store.create_source(Source(id=new_id(), type="upload", path="x.mp4"))
        job = store.create_job(Job(id=new_id(), source_id=source.id, status="done"))
        clip = Clip(id=new_id(), job_id=job.id, start_s=0, end_s=30, rank=2, post_caption="Old?")
        store.replace_clips(job.id, [clip])
        out = paths.exports_dir() / job.id
        out.mkdir(parents=True)
        (out / "clip_02_x_9x16.mp4").write_bytes(b"x")
        store.create_export(
            Export(id=new_id(), clip_id=clip.id, path=str(out / "clip_02_x_9x16.mp4"))
        )

        body = client.patch(
            f"/api/clips/{clip.id}",
            json={
                "post_caption": " New line? \n\n@michaelsartain #x ",
                "hook_title_alts": ["A", " B "],
            },
        ).json()

        assert body["post_caption"] == "New line? \n\n@michaelsartain #x"
        assert body["hook_title_alts"] == ["A", "B"]
        assert (out / "clip_02_caption.txt").read_text(encoding="utf-8").startswith("New line?")

    def test_caption_patch_before_export_writes_no_file(self, client: TestClient) -> None:
        source = store.create_source(Source(id=new_id(), type="upload", path="x.mp4"))
        job = store.create_job(Job(id=new_id(), source_id=source.id))
        clip = Clip(id=new_id(), job_id=job.id, start_s=0, end_s=30, rank=1)
        store.replace_clips(job.id, [clip])

        client.patch(f"/api/clips/{clip.id}", json={"post_caption": "Hi?"})

        assert not (paths.exports_dir() / job.id / "clip_01_caption.txt").exists()


@pytest.mark.slow
def test_export_stage_writes_caption_next_to_a_ranked_clip(initialised_db, tmp_path) -> None:
    video = tmp_path / "src.mp4"
    subprocess.run(
        [ffmpeg.ffmpeg_path(), "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=640x360:rate=30:duration=4", "-f", "lavfi", "-i",
         "sine=frequency=440:duration=4", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(video)],
        check=True, capture_output=True,
    )  # fmt: skip
    transcript = _transcript("Men go up. Women plummet. Is he right?", pace=0.3)
    source = store.create_source(
        Source(id=new_id(), type="upload", path=str(video), width=640, height=360, duration_s=4)
    )
    job = store.create_job(Job(id=new_id(), source_id=source.id))
    clip = Clip(
        id=new_id(), job_id=job.id, start_s=0.0, end_s=2.5, start_word=0,
        end_word=len(transcript.words) - 1, rank=1, title="Men go up",
        hook_title="Men Go Up, Women Plummet 😳", post_caption="Is he right?\n\n#x",
    )  # fmt: skip
    store.replace_clips(job.id, [clip])
    settings = config.Settings()
    settings.cleanup.auto_delete_sources = False
    runner = PipelineRunner(job, source, settings=settings)

    runner._stage_export([clip], transcript, {})

    out = paths.exports_dir() / job.id
    assert (out / "clip_01_men-go-up_9x16.mp4").exists()
    assert (out / "clip_01_caption.txt").read_text(encoding="utf-8") == "Is he right?\n\n#x\n"


def _transcript(text: str, *, pace: float = 0.35) -> Transcript:
    return Transcript(
        words=[
            Word(text=token, start=i * pace, end=i * pace + pace * 0.9)
            for i, token in enumerate(text.split())
        ]
    )
