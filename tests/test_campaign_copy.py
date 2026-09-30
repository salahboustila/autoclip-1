"""Campaign copy: guest descriptor, the lines caption, handles never in titles."""

from __future__ import annotations

import json

import pytest
from autoclip import campaigns
from autoclip.campaigns.copy import campaign_facts, guest_descriptor
from autoclip.db.models import Clip, Source
from autoclip.pipeline import hookcopy
from autoclip.pipeline.transcript import Transcript, Word
from autoclip.providers import DetectionConfig
from autoclip.providers.base import LLMProvider, ProviderStatus

PRESET = campaigns.load_preset("jack_neel")
LINES = ["@jackhneel", "@jackneel", "#jackneelpod"]
CLIP = (
    "How much money did you make last year? I made four million dollars from one deal. "
    "Most people never see that kind of money in their whole life."
)


class TestGuestDescriptor:
    @pytest.mark.parametrize(
        ("title", "expected"),
        [
            ("Ex-CIA Spy: “His Body Was Switched!” The 7 People │Jack Neel", "Ex-CIA Spy"),
            ('$120M CEO: "You Can Make $1M in a Week!" The 3 Hidden AI Gold Rushes', "$120M CEO"),
            (
                "“You Can’t Make It!” How I Would Make 100K | Alex Hormozi x Jack Neel",
                "Alex Hormozi",
            ),
            (
                "(NEW!) Andrew Tate's Emergency Interview Before Miami Arrest | Jack Neel Podcast",
                "",
            ),
            ('"Quote First": then words', ""),
        ],
    )
    def test_from_real_episode_titles(self, title: str, expected: str) -> None:
        assert guest_descriptor(title, PRESET.host.aliases) == expected

    def test_facts_include_the_preset_and_the_descriptor(self) -> None:
        source = Source(id="s", type="youtube", path="x", title="Ex-CIA Spy: “Quote!” More")
        facts = campaign_facts(PRESET, source)
        assert "Jack Neel is the host" in facts
        assert "describes the guest as: Ex-CIA Spy." in facts

    def test_the_descriptor_lets_an_authority_title_pass_grounding(self) -> None:
        source = Source(id="s", type="youtube", path="x", title="Ex-CIA Spy: “Quote!” More")
        title = "Ex-CIA Spy Reveals How Much Money He Made 💰"
        assert not hookcopy.check_grounding(title, CLIP).ok
        assert hookcopy.check_grounding(title, CLIP, campaign_facts(PRESET, source)).ok


class TestLinesCaption:
    def _caption(self, text: str, **kw) -> str:
        return hookcopy.build_lines_caption(
            text, title=kw.pop("title", "Four Million From One Deal 💰"), clip_text=CLIP,
            lines=LINES, **kw,
        )  # fmt: skip

    def test_hook_line_question_line_then_each_tag_on_its_own_line(self) -> None:
        caption = self._caption("He made four million dollars from one deal. Could you?")
        assert caption.split("\n") == [
            "He made four million dollars from one deal.",
            "Could you?",
            "",
            "@jackhneel",
            "@jackneel",
            "#jackneelpod",
        ]

    def test_model_tags_are_dropped_for_the_campaigns(self) -> None:
        caption = self._caption("He made four million. Is that a lot? #money @someone #rich")
        assert "#money" not in caption and "@someone" not in caption
        assert caption.count("#jackneelpod") == 1

    def test_a_missing_question_becomes_do_you_agree(self) -> None:
        assert self._caption("He made four million.").split("\n")[1] == "Do you agree?"

    def test_an_invented_number_is_replaced_from_the_clip(self) -> None:
        caption = self._caption(
            "He made nine million. Could you?",
            title="How Much Did You Make? 🤔",
            hook_fallback="I made four million dollars from one deal.",
        )
        assert caption.split("\n")[0] == "“I made four million dollars from one deal.”"
        assert "nine" not in caption

    def test_with_no_model_caption_the_title_is_used(self) -> None:
        assert self._caption("").split("\n")[:2] == ["Four Million From One Deal.", "Do you agree?"]


class TestTitles:
    @pytest.mark.parametrize(
        "title", ["Follow @jackneel For More 🤔", "#jackneelpod Money Talk 💰"]
    )
    def test_handles_and_hashtags_never_become_titles(self, title: str) -> None:
        assert hookcopy.validate_option(title, clip_text=CLIP + " follow jackneel more")[0] is None


class FakeProvider(LLMProvider):
    name = "fake"
    requires_key = False

    def __init__(self, reply: dict) -> None:
        super().__init__("fake")
        self.reply = json.dumps(reply)
        self.users: list[str] = []

    async def _complete(self, system: str, user: str, config: DetectionConfig) -> str:
        self.users.append(user)
        return self.reply

    async def health_check(self) -> ProviderStatus:
        return ProviderStatus(name=self.name, available=True)


class TestGenerateForClips:
    async def test_campaign_captions_and_three_titles(self) -> None:
        words = [Word(t, i * 0.4, i * 0.4 + 0.35) for i, t in enumerate(CLIP.split())]
        transcript = Transcript(words=words)
        clip = Clip(
            id="c", job_id="j", start_s=0, end_s=12, start_word=0, end_word=len(words) - 1,
            rank=1, question_text="How much money did you make last year?", topic="Money",
        )  # fmt: skip
        provider = FakeProvider(
            {
                "titles": [
                    {"text": "How Much Money Did You Make? 🤔", "score": 90},
                    {"text": "Four Million Dollars From One Deal 💰", "score": 85},
                    {"text": "Follow @jackneel 👀", "score": 99},
                ],
                "caption": "He made four million dollars from one deal. Could you?",
                "hashtags": ["#money"],
            }
        )

        await hookcopy.generate_for_clips(
            [clip], transcript, provider, facts="Jack Neel is the host.", caption_lines=LINES
        )

        assert clip.hook_title == "How Much Money Did You Make? 🤔"
        assert clip.hook_title_alts[0] == "Four Million Dollars From One Deal 💰"
        assert len(clip.hook_title_alts) == 2  # the handle title was rejected, a fallback fills
        assert clip.post_caption.endswith("@jackhneel\n@jackneel\n#jackneelpod")
        assert "Jack Neel is the host." in provider.users[0]

    async def test_default_viral_captions_are_unchanged(self) -> None:
        provider = FakeProvider(
            {"titles": [{"text": "Four Million From One Deal 💰"}], "caption": "Could you?"}
        )
        copy = await hookcopy.generate(provider, CLIP, fallbacks=[], fixed_hashtags=["#usa"])
        assert copy.caption.startswith("Could you?\n\n@michaelsartain")
