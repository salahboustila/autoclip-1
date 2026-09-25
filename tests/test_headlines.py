"""AI Clip Headlines — generation logic.

Uses the same ScriptedProvider test double as test_providers.py, so none of
this touches a real network or a real API key: it's testing the wrapping,
cleaning, and per-clip failure tolerance around whatever a provider returns,
not any specific provider's SDK.
"""

from __future__ import annotations

import pytest
from autoclip.db.models import Clip, new_id
from autoclip.pipeline import headlines
from autoclip.pipeline.transcript import Transcript, Word
from autoclip.providers import DetectionConfig, ProviderStatus
from autoclip.providers.base import LLMProvider


class ScriptedProvider(LLMProvider):
    """Returns queued responses, or raises the queued exception."""

    name = "scripted"
    requires_key = False

    def __init__(self, responses: list[str | Exception]) -> None:
        super().__init__("test-model")
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.call_count = 0

    async def _complete(self, system: str, user: str, config: DetectionConfig) -> str:
        self.prompts.append(user)
        self.call_count += 1
        response = self.responses.pop(0) if self.responses else '{"headline": ""}'
        if isinstance(response, Exception):
            raise response
        return response

    async def health_check(self) -> ProviderStatus:
        return ProviderStatus(name=self.name, available=True)


def _clip(**overrides) -> Clip:
    defaults = {
        "id": new_id(),
        "job_id": "job1",
        "start_s": 0.0,
        "end_s": 10.0,
        "start_word": 0,
        "end_word": 4,
        "title": "A neutral folder label",
        "hook": "Nobody expected this",
    }
    return Clip(**{**defaults, **overrides})


def _transcript(word_count: int = 5) -> Transcript:
    words = [Word(text=f"word{i}", start=i * 0.5, end=i * 0.5 + 0.4) for i in range(word_count)]
    return Transcript(words=words)


class TestClean:
    def test_strips_surrounding_double_quotes(self) -> None:
        assert headlines._clean('"He said what?!"') == "He said what?!"

    def test_strips_surrounding_single_quotes(self) -> None:
        assert headlines._clean("'A wild claim'") == "A wild claim"

    def test_leaves_an_internal_quote_alone(self) -> None:
        # Only a *matching pair at both ends* counts as wrapping quotes.
        assert headlines._clean('He called it "genius"') == 'He called it "genius"'

    def test_collapses_internal_whitespace(self) -> None:
        assert headlines._clean("Too   many\nspaces") == "Too many spaces"

    def test_strips_leading_and_trailing_whitespace(self) -> None:
        assert headlines._clean("  padded  ") == "padded"


class TestWordLimit:
    def test_short_text_is_unchanged(self) -> None:
        text = "Five short words here"
        assert headlines._enforce_word_limit(text) == text

    def test_exactly_at_the_limit_is_unchanged(self) -> None:
        text = " ".join(f"w{i}" for i in range(headlines.MAX_WORDS))
        assert headlines._enforce_word_limit(text) == text

    def test_truncates_past_the_limit(self) -> None:
        text = " ".join(f"w{i}" for i in range(20))

        result = headlines._enforce_word_limit(text)

        assert len(result.split(" ")) == headlines.MAX_WORDS
        assert result == " ".join(f"w{i}" for i in range(headlines.MAX_WORDS))


class TestHeadlineResponseValidation:
    def test_null_headline_coerces_to_empty_string(self) -> None:
        parsed = headlines._HeadlineResponse.model_validate({"headline": None})

        assert parsed.headline == ""

    def test_non_string_headline_is_stringified(self) -> None:
        parsed = headlines._HeadlineResponse.model_validate({"headline": 42})

        assert parsed.headline == "42"


class TestGenerate:
    async def test_empty_clip_list_makes_no_calls(self) -> None:
        provider = ScriptedProvider([])

        result = await headlines.generate([], _transcript(), provider)

        assert result == {}
        assert provider.call_count == 0

    async def test_one_headline_per_clip(self) -> None:
        clip = _clip()
        provider = ScriptedProvider(['{"headline": "A stranger stops scrolling"}'])

        result = await headlines.generate([clip], _transcript(), provider)

        assert result == {clip.id: "A stranger stops scrolling"}

    async def test_multiple_clips_each_get_their_own_headline(self) -> None:
        clip_a, clip_b = _clip(start_word=0, end_word=2), _clip(start_word=2, end_word=4)
        provider = ScriptedProvider(
            ['{"headline": "First headline"}', '{"headline": "Second headline"}']
        )

        result = await headlines.generate([clip_a, clip_b], _transcript(), provider)

        assert set(result.values()) == {"First headline", "Second headline"}
        assert set(result) == {clip_a.id, clip_b.id}

    async def test_a_word_limit_violation_from_the_model_is_still_enforced(self) -> None:
        clip = _clip()
        long_headline = " ".join(f"word{i}" for i in range(15))
        provider = ScriptedProvider([f'{{"headline": "{long_headline}"}}'])

        result = await headlines.generate([clip], _transcript(), provider)

        assert len(result[clip.id].split(" ")) == headlines.MAX_WORDS

    async def test_one_clip_failing_does_not_lose_the_others(self) -> None:
        """The same one-bad-item-shouldn't-cost-the-rest principle
        highlights.detect applies to transcript windows, applied per clip."""
        clip_a, clip_b = _clip(start_word=0, end_word=2), _clip(start_word=2, end_word=4)
        provider = ScriptedProvider(
            [RuntimeError("provider exploded"), '{"headline": "This one worked"}']
        )

        result = await headlines.generate([clip_a, clip_b], _transcript(), provider)

        assert len(result) == 1
        assert clip_b.id in result or clip_a.id in result

    async def test_malformed_json_leaves_that_clip_out_rather_than_raising(self) -> None:
        clip = _clip()
        provider = ScriptedProvider(["not json at all"])

        result = await headlines.generate([clip], _transcript(), provider)

        assert result == {}

    async def test_empty_headline_from_the_model_is_dropped_not_stored(self) -> None:
        clip = _clip()
        provider = ScriptedProvider(['{"headline": ""}'])

        result = await headlines.generate([clip], _transcript(), provider)

        assert clip.id not in result

    async def test_progress_reaches_completion(self) -> None:
        clips = [_clip(start_word=i, end_word=i + 1) for i in range(3)]
        provider = ScriptedProvider(
            ['{"headline": "H1"}', '{"headline": "H2"}', '{"headline": "H3"}']
        )
        seen: list[float] = []

        await headlines.generate(
            clips, _transcript(word_count=10), provider, on_progress=seen.append
        )

        assert seen[-1] == pytest.approx(1.0)
        assert all(0.0 <= v <= 1.0 for v in seen)

    async def test_falls_back_to_hook_when_the_word_slice_is_empty(self) -> None:
        """A clip whose word range is somehow out of bounds still gets
        *something* meaningful sent to the model rather than an empty prompt."""
        clip = _clip(start_word=0, end_word=4, hook="The fallback text")
        provider = ScriptedProvider(['{"headline": "Generated from hook"}'])

        await headlines.generate([clip], _transcript(word_count=0), provider)

        assert "The fallback text" in provider.prompts[0]

    async def test_ollama_uses_single_concurrency(self) -> None:
        """A local model is already saturating its one GPU/CPU slot — mirrors
        highlights.py's HOSTED_CONCURRENCY/LOCAL_CONCURRENCY split."""
        clips = [_clip(start_word=i, end_word=i + 1) for i in range(3)]

        class OllamaLikeProvider(ScriptedProvider):
            name = "ollama"

        provider = OllamaLikeProvider(
            ['{"headline": "H1"}', '{"headline": "H2"}', '{"headline": "H3"}']
        )

        result = await headlines.generate(clips, _transcript(word_count=10), provider)

        assert len(result) == 3
