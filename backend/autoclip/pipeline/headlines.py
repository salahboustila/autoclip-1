"""AI Clip Headlines — one short, attention-grabbing on-screen title per clip.

Distinct from ``Clip.title``, which highlight detection already produces:
that field is deliberately a neutral folder label ("how a human editor would
name this clip"), not clickbait — see prompts/highlight_v1.txt. This module
asks a second, much smaller question with its own prompt: given just this
clip's words, what's the one line that makes a stranger stop scrolling.

Everything reusable is reused rather than rebuilt: the same LLMProvider
implementations, the same transient-failure backoff, and the same
JSON-in-JSON-out contract highlight detection already relies on (see
LLMProvider.complete_json in providers/base.py). What's new here is just the
prompt and a one-field schema instead of a list of clip candidates.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable

from pydantic import BaseModel, field_validator

from ..db.models import Clip
from ..providers import DetectionConfig, LLMProvider
from ..providers.base import load_prompt
from .transcript import Transcript

log = logging.getLogger(__name__)

#: Hosted providers tolerate this many concurrent clip calls; a local model is
#: already saturating its one GPU/CPU slot. Mirrors highlights.py's split.
HOSTED_CONCURRENCY = 4
LOCAL_CONCURRENCY = 1

#: Hard ceiling enforced here too, in case a model ignores the prompt's own
#: "10 words maximum" instruction — this is the actual safety net.
MAX_WORDS = 10

#: Creative but not erratic. Lower than highlight detection's 0.3 would make
#: every headline read the same; higher risks drifting from the transcript.
TEMPERATURE = 0.7

_WHITESPACE = re.compile(r"\s+")


class _HeadlineResponse(BaseModel):
    headline: str = ""

    @field_validator("headline", mode="before")
    @classmethod
    def _coerce_to_string(cls, value: object) -> str:
        # Models occasionally return null or a number where text was asked for.
        return "" if value is None else str(value)


def _clean(text: str) -> str:
    """Strip the quoting and whitespace noise a model adds despite instructions."""
    text = text.strip()
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1].strip()
    return _WHITESPACE.sub(" ", text)


def _enforce_word_limit(text: str, *, max_words: int = MAX_WORDS) -> str:
    words = text.split(" ")
    return text if len(words) <= max_words else " ".join(words[:max_words])


async def _generate_one(provider: LLMProvider, clip_text: str) -> str:
    system = load_prompt("headline_v1")
    user = (
        f"Clip transcript:\n---\n{clip_text}\n---\n\n"
        "Respond with ONLY a JSON object matching the schema. No prose, no markdown fences."
    )
    config = DetectionConfig(temperature=TEMPERATURE)
    payload = await provider.complete_json(system, user, config)
    headline = _HeadlineResponse.model_validate(payload).headline
    return _enforce_word_limit(_clean(headline))


async def generate(
    clips: list[Clip],
    transcript: Transcript,
    provider: LLMProvider,
    *,
    on_progress: Callable[[float], None] | None = None,
) -> dict[str, str]:
    """Generate one headline per clip. Returns ``{clip_id: headline_text}``.

    A clip whose generation fails, or whose response doesn't validate, is
    simply left out of the returned mapping rather than failing the batch —
    the same one-bad-item-shouldn't-cost-the-rest principle
    :func:`highlights.detect` applies to transcript windows. The caller treats
    a missing entry as "no headline for this clip", which degrades to
    rendering nothing rather than blocking the job.
    """
    if not clips:
        return {}

    concurrency = LOCAL_CONCURRENCY if provider.name == "ollama" else HOSTED_CONCURRENCY
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    lock = asyncio.Lock()
    results: dict[str, str] = {}

    async def run_one(clip: Clip) -> None:
        nonlocal completed
        async with semaphore:
            words = transcript.slice(clip.start_word, clip.end_word)
            clip_text = " ".join(w.text for w in words).strip() or clip.hook or clip.title
            try:
                headline = await _generate_one(provider, clip_text)
                if headline:
                    results[clip.id] = headline
            except Exception as exc:
                log.warning(
                    "Headline generation failed for clip %s (%s); leaving it blank.",
                    clip.id,
                    exc,
                )
            async with lock:
                completed += 1
                if on_progress:
                    on_progress(completed / len(clips))

    await asyncio.gather(*(run_one(clip) for clip in clips))
    return results
