"""LLM provider abstraction.

Highlight detection is the one place AutoClip asks a language model to make a
judgement call, so the interface is deliberately narrow: a window of transcript
goes in, a validated list of clip candidates comes out.

Two design choices carry most of the weight:

**Word indices, not seconds.** The model returns ``start_word_index`` /
``end_word_index``. Timing is then looked up from measured word timestamps, so a
model that is bad at arithmetic — which they all are — cannot produce a clip
that starts at the wrong moment.

**Validate, then retry with the error.** Small local models produce malformed
JSON often enough that a single retry carrying the actual validation message
turns most failures into successes. That loop lives here so every provider
inherits it.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr, ValidationError, field_validator

log = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


class ProviderError(RuntimeError):
    """A provider could not produce a usable response."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        hint: str = "",
        attempts: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.hint = hint
        #: Raw responses that led here, for the per-window debug trace.
        self.attempts = attempts or []

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base}\n\n{self.hint}" if self.hint else base


# --------------------------------------------------------------------------
# Contracts
# --------------------------------------------------------------------------


class ClipCandidate(BaseModel):
    """One proposed clip, as returned by the model."""

    start_word_index: int = Field(ge=0)
    end_word_index: int = Field(ge=0)
    title: str = ""
    hook: str = ""
    score: int = Field(default=50, ge=0, le=100)
    reason: str = ""

    @field_validator("title", "hook", "reason", mode="before")
    @classmethod
    def _coerce_to_string(cls, value: Any) -> str:
        # Models occasionally return null or a number where text was asked for.
        return "" if value is None else str(value)

    @field_validator("score", mode="before")
    @classmethod
    def _coerce_score(cls, value: Any) -> int:
        """Accept floats and 0-1 fractions, which models emit despite the schema."""
        if value is None:
            return 50
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 50
        if 0.0 < number <= 1.0:
            number *= 100
        return max(0, min(100, round(number)))


class ClipCandidates(BaseModel):
    clips: list[ClipCandidate] = Field(default_factory=list)
    #: Each model reply behind this result, as ``{"raw", "error"}``. Private so
    #: it never takes part in validating the model's JSON.
    _attempts: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    @property
    def attempts(self) -> list[dict[str, Any]]:
        return self._attempts


@dataclass
class TranscriptWindow:
    """A slice of transcript presented to the model.

    ``first_word`` lets a provider work on a window while still reporting
    indices in the full transcript's coordinate space.
    """

    text: str
    first_word: int
    last_word: int
    duration_s: float = 0.0
    speakers: list[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return self.last_word - self.first_word + 1


@dataclass
class DetectionConfig:
    min_duration_s: float = 20.0
    max_duration_s: float = 90.0
    max_clips: int = 10
    language: str = ""
    #: Prompt file stem in ``autoclip/prompts/``. Versioned so contributors can
    #: iterate on prompts without touching code.
    prompt_version: str = "highlight_v1"
    temperature: float = 0.3
    #: Candidates scoring below this are not wanted. Stated in the prompt and
    #: enforced again in code, since models don't always obey.
    min_score: int = 50
    #: Non-zero only for the fallback pass: ask for this many best moments per
    #: window regardless of ``min_score``.
    fallback_clips: int = 0


@dataclass
class ProviderStatus:
    name: str
    available: bool
    detail: str = ""
    models: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Base provider
# --------------------------------------------------------------------------


class LLMProvider(ABC):
    """Base class for highlight-detection providers."""

    #: Stable identifier, matching the key used in settings.
    name: str = ""
    #: Whether the provider needs an API key.
    requires_key: bool = True

    def __init__(self, model: str, *, api_key: str | None = None, base_url: str | None = None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    # -- to implement ------------------------------------------------------

    @abstractmethod
    async def _complete(self, system: str, user: str, config: DetectionConfig) -> str:
        """Send one prompt and return the raw response text."""

    @abstractmethod
    async def health_check(self) -> ProviderStatus:
        """Report whether this provider is usable right now."""

    # -- shared behaviour --------------------------------------------------

    async def detect_highlights(
        self, window: TranscriptWindow, config: DetectionConfig
    ) -> ClipCandidates:
        """Ask the model for clip candidates in ``window``, validating the reply.

        One retry is attempted on a schema violation, feeding the validation
        error back so the model can correct itself.
        """
        system = render_system_prompt(load_prompt(config.prompt_version), config)
        user = render_window_prompt(window, config)
        attempts: list[dict[str, Any]] = []

        raw = await self._complete(system, user, config)
        try:
            result = self._parse(raw, window)
            attempts.append({"raw": raw, "error": None})
            result._attempts = attempts
            return result
        except (ValidationError, ValueError) as first_error:
            attempts.append({"raw": raw, "error": str(first_error)})
            log.warning("%s returned invalid JSON; retrying with feedback.", self.name)
            repair = (
                f"{user}\n\n"
                "Your previous response did not match the required schema.\n"
                f"Validation error:\n{first_error}\n\n"
                "Respond again with ONLY the corrected JSON object. No prose, no "
                "markdown fences."
            )
            raw = await self._complete(system, repair, config)
            try:
                result = self._parse(raw, window)
                attempts.append({"raw": raw, "error": None})
                result._attempts = attempts
                return result
            except (ValidationError, ValueError) as second_error:
                attempts.append({"raw": raw, "error": str(second_error)})
                raise ProviderError(
                    f"{self.name} returned malformed clip data twice.",
                    provider=self.name,
                    attempts=attempts,
                    hint=(
                        f"Last validation error: {second_error}\n\n"
                        "Smaller local models struggle with strict JSON. Try a larger "
                        "model, or switch to a hosted provider."
                    ),
                ) from second_error

    def _parse(self, raw: str, window: TranscriptWindow) -> ClipCandidates:
        """Validate a raw response and clamp indices into the window."""
        payload = extract_json_object(raw)
        candidates = ClipCandidates.model_validate(payload)

        cleaned: list[ClipCandidate] = []
        for candidate in candidates.clips:
            start = max(window.first_word, min(candidate.start_word_index, window.last_word))
            end = max(window.first_word, min(candidate.end_word_index, window.last_word))
            if end <= start:
                # A zero-or-negative-length clip is a model slip, not a candidate.
                continue
            candidate.start_word_index = start
            candidate.end_word_index = end
            cleaned.append(candidate)

        return ClipCandidates(clips=cleaned)


# --------------------------------------------------------------------------
# Prompt handling
# --------------------------------------------------------------------------


def load_prompt(version: str) -> str:
    """Read a versioned prompt file from ``autoclip/prompts/``."""
    path = PROMPT_DIR / f"{version}.txt"
    if not path.exists():
        raise ProviderError(
            f"Prompt '{version}' not found at {path}.",
            hint="Prompt files live in backend/autoclip/prompts/ as versioned .txt files.",
        )
    return path.read_text(encoding="utf-8")


#: Placeholders in prompt files for the parts that depend on the pass. Both
#: must go in the fallback: left in, "return nothing" beats "return your best
#: three" and the fallback comes back empty (seen with claude-sonnet-5).
SCORE_RULE_TOKEN = "<<SCORE_RULE>>"
EMPTY_ANSWER_TOKEN = "<<EMPTY_ANSWER>>"


def render_system_prompt(template: str, config: DetectionConfig) -> str:
    """Fill in the score rule: the normal cut-off, or the fallback's override."""
    if config.fallback_clips:
        rule = (
            f"This is a second pass. An earlier pass found nothing scoring {config.min_score} "
            f"or higher anywhere in this video. For this pass, ignore that cut-off: return the "
            f"{config.fallback_clips} strongest moments in this section, even if they score "
            f"below {config.min_score}. Score them honestly; they will be shown to the editor as "
            "low-confidence. Return an empty list only if the section has no complete thought "
            "at all."
        )
        empty_answer = ""
    else:
        rule = (
            f"Only return clips scoring {config.min_score} or higher. Leave out anything below.\n\n"
            "Return nothing rather than padding the list. A transcript section with three good "
            "moments should return three clips, not ten. Most sections of most videos contain "
            "nothing worth clipping; saying so is a correct answer."
        )
        empty_answer = 'If nothing in this section is worth clipping, return {"clips": []}.'
    return template.replace(SCORE_RULE_TOKEN, rule).replace(EMPTY_ANSWER_TOKEN, empty_answer)


def render_window_prompt(window: TranscriptWindow, config: DetectionConfig) -> str:
    """Build the user message for one transcript window."""
    speaker_note = ""
    if window.speakers:
        speaker_note = (
            f"\nThis section has {len(window.speakers)} distinct speakers "
            f"({', '.join(window.speakers)}); speaker labels are shown inline.\n"
        )

    return (
        f"Transcript section, words {window.first_word} to {window.last_word}.\n"
        f"Each word is tagged with its index as [index]word.\n"
        f"{speaker_note}\n"
        f"Clip length must be between {config.min_duration_s:.0f} and "
        f"{config.max_duration_s:.0f} seconds.\n"
        f"{_clip_count_line(config)}\n\n"
        f"---\n{window.text}\n---\n\n"
        "Respond with ONLY a JSON object matching the schema. No prose, no markdown fences."
    )


def _clip_count_line(config: DetectionConfig) -> str:
    if config.fallback_clips:
        return (
            f"Return your {config.fallback_clips} best moments in this section, ranked by score, "
            f"even if they score below {config.min_score}."
        )
    return f"Return at most {config.max_clips} clips."


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_object(raw: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response.

    Handles the three things models do despite being told not to: wrap the JSON
    in markdown fences, prepend an explanatory sentence, and append a trailing
    note. Falls back to brace matching so a stray character outside the object
    doesn't cost a retry round-trip.
    """
    text = raw.strip()
    if not text:
        raise ValueError("The provider returned an empty response.")

    if match := _JSON_FENCE.search(text):
        text = match.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = _json_by_brace_matching(text)

    if isinstance(parsed, list):
        # Some models skip the wrapper object and return the array directly.
        return {"clips": parsed}
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}.")
    return parsed


def _json_by_brace_matching(text: str) -> Any:
    """Extract the first balanced ``{...}`` or ``[...]`` region and parse it."""
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            char = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])

    raise ValueError("No JSON object found in the response.")
