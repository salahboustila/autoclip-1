"""Viral Hook Mode — scoring and opening rules for the podcast-clip preset.

The prompt (``prompts/highlight_viral_v1.txt``) asks the model to rate four
things per candidate. The combined score is computed here rather than trusted
from the model, so the weights are fixed and testable, and the one claim that
can be checked against the transcript — "the speaker says a number" — is.

The opening rules back up the prompt's "start on the strongest sentence":
models regularly start a sentence early, so leading filler words, filler-only
sentences, and (with diarization) a different speaker's question are trimmed
off the front in code.
"""

from __future__ import annotations

import re

from ..providers.base import ClipCandidate
from .transcript import Transcript

VIRAL_PROMPT = "highlight_viral_v1"

#: Weight of each criterion in the combined 0-100 score.
WEIGHTS: dict[str, float] = {
    "hook_strength": 0.35,
    "comment_potential": 0.30,
    "standalone_clarity": 0.20,
    "has_number": 0.15,
}

_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety", "hundred", "thousand", "million", "billion", "trillion",
    "percent", "half", "double", "triple", "twice", "quarter", "dozen",
}  # fmt: skip
_DIGIT = re.compile(r"\d")


def normalise_word(text: str) -> str:
    return re.sub(r"[^\w'%$]", "", text).lower()


def mentions_number(text: str) -> bool:
    """True if ``text`` contains a digit, a % or $ sign, or a number word."""
    if _DIGIT.search(text) or "%" in text or "$" in text:
        return True
    return any(normalise_word(word) in _NUMBER_WORDS for word in text.split())


def combined_score(candidate: ClipCandidate, clip_text: str) -> int:
    """Combine the four criteria into one 0-100 score.

    ``has_number`` counts only when the transcript agrees: a model claiming a
    statistic the speaker never says gets no credit for it. When the model
    returned no sub-scores at all, its own overall score stands.
    """
    parts = (
        candidate.hook_strength,
        candidate.comment_potential,
        candidate.standalone_clarity,
    )
    if all(part is None for part in parts):
        return candidate.score

    fallback = candidate.score
    has_number = bool(candidate.has_number) and mentions_number(clip_text)
    total = (
        WEIGHTS["hook_strength"] * _or(candidate.hook_strength, fallback)
        + WEIGHTS["comment_potential"] * _or(candidate.comment_potential, fallback)
        + WEIGHTS["standalone_clarity"] * _or(candidate.standalone_clarity, fallback)
        + WEIGHTS["has_number"] * (100 if has_number else 0)
    )
    return max(0, min(100, round(total)))


def _or(value: int | None, default: int) -> int:
    return default if value is None else value


# --------------------------------------------------------------------------
# Opening rules
# --------------------------------------------------------------------------

#: Words that never make a good first word. Trimmed off the front of a clip.
LEADING_FILLER = {
    "so", "um", "uh", "umm", "uhh", "er", "ah", "and", "but", "yeah", "yes", "okay", "ok",
    "well", "right", "anyway", "look", "listen", "mhm", "hmm", "oh",
}  # fmt: skip
#: A sentence made only of these ("Yeah. Exactly. Right.") is skipped whole.
FILLER_SENTENCE = LEADING_FILLER | {
    "exactly", "totally", "absolutely", "true", "sure", "i", "mean", "you", "know", "like",
    "that's", "it", "is", "no", "definitely", "for", "real", "wow",
}  # fmt: skip

#: Never trim more than this share of a clip's words off the front.
MAX_TRIM_SHARE = 0.3
MAX_LEADING_FILLER = 4


def _sentence_end(transcript: Transcript, start: int, limit: int) -> int:
    for index in range(start, limit + 1):
        if transcript.words[index].ends_sentence:
            return index
    return limit


def trim_opening(transcript: Transcript, start_word: int, end_word: int) -> int:
    """Return a new start index that begins on real content.

    Applied in order, repeatedly, until nothing changes:
    - a sentence consisting only of filler is skipped whole;
    - with diarization, an opening question from someone other than the clip's
      main speaker (the host) is skipped, so the clip opens on the answer;
    - up to four leading filler words ("so yeah, um…") are dropped.
    """
    words = transcript.words
    budget = start_word + int((end_word - start_word + 1) * MAX_TRIM_SHARE)
    main_speaker = transcript.dominant_speaker(start_word, end_word)
    start = start_word

    changed = True
    while changed and start < budget:
        changed = False

        sentence_end = _sentence_end(transcript, start, end_word)
        sentence = words[start : sentence_end + 1]
        next_start = sentence_end + 1

        if next_start <= budget and all(
            normalise_word(w.text) in FILLER_SENTENCE for w in sentence
        ):
            start, changed = next_start, True
            continue

        if (
            next_start <= budget
            and main_speaker is not None
            and sentence[0].speaker not in (None, main_speaker)
            and sentence[-1].text.strip().endswith("?")
        ):
            start, changed = next_start, True
            continue

        dropped = 0
        while (
            dropped < MAX_LEADING_FILLER
            and start < min(budget, sentence_end)
            and normalise_word(words[start].text) in LEADING_FILLER
        ):
            start += 1
            dropped += 1
            changed = True

    return start
