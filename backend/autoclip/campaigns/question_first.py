"""Question-first clips: every clip opens on the host asking a question.

The campaign prompt asks for this, but a prompt is a request. This module is
the rule: each candidate the model proposes is checked against the transcript
and either moved to start exactly on the question or rejected.

A candidate passes when:

- a question sentence starts at (or within a few seconds of) its start — any
  lead-in before it, and filler like "so" or "um" inside it, is trimmed off;
- the question is short (``max_question_s``) and at least three words;
- an answer of real length follows it inside the clip;
- with speaker labels: the host asks it and someone else answers.

Without speaker labels only the first three can be checked. That is reported
(``host_check``) so the UI can warn rather than pretend.

The same object scores each survivor from the four 0-10 criteria, applies the
freshness penalty, and records the details the campaign report needs.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..pipeline.transcript import Transcript
from ..pipeline.viral import LEADING_FILLER, WEIGHTS, mentions_number, normalise_word

if TYPE_CHECKING:
    from ..db.models import Clip
    from ..pipeline.boundaries import Boundary
    from ..providers.base import ClipCandidate
    from .freshness import Freshness
    from .preset import CampaignPreset

log = logging.getLogger(__name__)

#: How far into a proposed clip the question may start; the lead-in is cut.
LEAD_IN_MAX_S = 6.0
#: A question sentence that ends this close before a proposed start counts:
#: the model started on the answer, one sentence late.
LATE_START_GAP_S = 1.5
MIN_QUESTION_WORDS = 3
#: The answer must run at least this long inside the clip.
MIN_ANSWER_S = 12.0
MAX_LEADING_FILLER = 4

_WH = {"what", "how", "why", "when", "where", "who", "which", "whose"}
_AUX = {"do", "does", "did", "is", "are", "was", "were", "can", "could", "would", "should",
        "will", "have", "has", "had", "am", "shall", "don't", "didn't", "isn't", "aren't",
        "wouldn't", "shouldn't", "can't", "won't", "haven't"}  # fmt: skip
_WH_NEXT = _AUX | {"much", "many", "long", "old", "come", "if", "else", "about", "kind", "type",
                   "happens", "happened", "makes", "made", "exactly", "the", "you"}  # fmt: skip
_SUBJECTS = {"you", "it", "he", "she", "they", "we", "i", "that", "this", "there", "your",
             "the", "a", "anyone", "people", "men", "women", "money", "someone"}  # fmt: skip
_CLOSING = "\"'”’)]"

HOST_CHECK_VERIFIED = "verified"
HOST_CHECK_NO_LABELS = "no_speaker_labels"
HOST_CHECK_NO_HOST = "host_not_identified"


def sentences(transcript: Transcript) -> list[tuple[int, int]]:
    """Every sentence as an inclusive ``(first, last)`` word range."""
    out: list[tuple[int, int]] = []
    first = 0
    for index, word in enumerate(transcript.words):
        if word.ends_sentence:
            out.append((first, index))
            first = index + 1
    if first < len(transcript.words):
        out.append((first, len(transcript.words) - 1))
    return out


def is_question(transcript: Transcript, first: int, last: int) -> bool:
    """A sentence ending in "?", or one Whisper left unpunctuated that opens
    like a question ("how much do…", "do you…", "what would…")."""
    words = transcript.words[first : last + 1]
    if not words:
        return False
    if words[-1].text.strip().rstrip(_CLOSING).endswith("?"):
        return True
    if words[-1].text.strip().rstrip(_CLOSING).endswith("!"):
        return False
    tokens = [normalise_word(w.text) for w in words[:6]]
    tokens = [t for t in tokens if t not in LEADING_FILLER][:2]
    if len(tokens) < 2:
        return False
    head, nxt = tokens
    return (head in _WH and nxt in _WH_NEXT) or (head in _AUX and nxt in _SUBJECTS)


def has_speaker_labels(transcript: Transcript) -> bool:
    return any(word.speaker for word in transcript.words)


def identify_host(transcript: Transcript) -> str | None:
    """The speaker who asks the most questions across the episode.

    On an interview podcast that is the host. None without speaker labels, or
    when no one asks at least two questions.
    """
    counts: Counter[str] = Counter()
    for first, last in sentences(transcript):
        if is_question(transcript, first, last):
            speaker = transcript.dominant_speaker(first, last)
            if speaker:
                counts[speaker] += 1
    if not counts:
        return None
    speaker, asked = counts.most_common(1)[0]
    return speaker if asked >= 2 else None


@dataclass
class QuestionFirstPolicy:
    preset: CampaignPreset
    freshness: Freshness | None = None
    #: Every rejected candidate and why, for ``campaign_rejections.json``.
    rejections: list[dict[str, Any]] = field(default_factory=list)
    #: Per accepted clip id: question range, criteria, freshness notes.
    details: dict[str, dict[str, Any]] = field(default_factory=dict)
    _host: dict[int, str | None] = field(default_factory=dict)
    _sentences: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    _questions: dict[int, tuple[int, int]] = field(default_factory=dict)
    _scores: dict[int, dict[str, Any]] = field(default_factory=dict)

    @property
    def allow_fallback(self) -> bool:
        return self.preset.detection.fallback_pass

    # -- host --------------------------------------------------------------

    def host(self, transcript: Transcript) -> str | None:
        key = id(transcript)
        if key not in self._host:
            self._host[key] = identify_host(transcript)
        return self._host[key]

    def host_check(self, transcript: Transcript) -> str:
        if not has_speaker_labels(transcript):
            return HOST_CHECK_NO_LABELS
        return HOST_CHECK_VERIFIED if self.host(transcript) else HOST_CHECK_NO_HOST

    # -- start -------------------------------------------------------------

    def _sentences_of(self, transcript: Transcript) -> list[tuple[int, int]]:
        key = id(transcript)
        if key not in self._sentences:
            self._sentences[key] = sentences(transcript)
        return self._sentences[key]

    def find_question(self, transcript: Transcript, start: int, end: int) -> tuple[int, int] | None:
        words = transcript.words
        spans = self._sentences_of(transcript)
        index = next((i for i, (_, last) in enumerate(spans) if last >= start), None)
        if index is None:
            return None

        first, last = spans[index]
        if first < start:
            # ``start`` fell mid-sentence; begin at the next sentence.
            index += 1

        if index > 0:
            prev_first, prev_last = spans[index - 1]
            gap = words[min(start, len(words) - 1)].start - words[prev_last].end
            if (
                prev_last < start
                and gap <= LATE_START_GAP_S
                and is_question(transcript, prev_first, prev_last)
            ):
                return prev_first, prev_last

        origin = words[start].start
        for first, last in spans[index:]:
            if first > end or words[first].start - origin > LEAD_IN_MAX_S:
                break
            if is_question(transcript, first, last):
                return first, last
        return None

    def trim_start(self, transcript: Transcript, start: int, end: int) -> int | None:
        """The index to start the clip on — the host's question — or None to reject."""
        words = transcript.words
        found = self.find_question(transcript, start, end)
        if found is None:
            return self._reject(transcript, start, end, "does not open on a question")
        first, last = found

        dropped = 0
        while (
            dropped < MAX_LEADING_FILLER
            and first < last
            and normalise_word(words[first].text) in LEADING_FILLER
        ):
            first += 1
            dropped += 1

        if last - first + 1 < MIN_QUESTION_WORDS:
            return self._reject(transcript, first, end, "question is too short")
        question_s = words[last].end - words[first].start
        if question_s > self.preset.detection.max_question_s:
            return self._reject(
                transcript, first, end, f"question takes {question_s:.1f}s; the answer starts late"
            )
        if last >= end or words[end].end - words[last].end < MIN_ANSWER_S:
            return self._reject(transcript, first, end, "no answer of real length follows")

        host = self.host(transcript) if has_speaker_labels(transcript) else None
        if host is not None:
            asker = transcript.dominant_speaker(first, last)
            if asker != host:
                return self._reject(
                    transcript, first, end, f"question asked by {asker}, not the host ({host})"
                )
            answer = next((span for span in self._sentences_of(transcript) if span[0] > last), None)
            if answer is not None and transcript.dominant_speaker(*answer) == host:
                return self._reject(transcript, first, end, "the host answers his own question")

        self._questions[first] = (first, last)
        return first

    def _reject(self, transcript: Transcript, start: int, end: int, reason: str) -> None:
        words = transcript.words
        self.rejections.append(
            {
                "start_s": round(words[start].start, 2),
                "end_s": round(words[min(end, len(words) - 1)].end, 2),
                "opening": transcript.text_between(start, min(start + 12, end)),
                "reason": reason,
            }
        )
        log.info("Campaign rejected a candidate at %.1fs: %s", words[start].start, reason)
        return None

    # -- score -------------------------------------------------------------

    def score(
        self, candidate: ClipCandidate, transcript: Transcript, boundary: Boundary
    ) -> int | None:
        """The 0-100 score: the four 0-10 criteria weighted, less freshness."""
        question = self._questions.get(boundary.start_word)
        answer_from = question[1] + 1 if question else boundary.start_word
        answer_text = transcript.text_between(answer_from, boundary.end_word)

        criteria = {
            "question_hook": candidate.question_hook,
            "controversy": candidate.controversy,
            "number_stat": candidate.number_stat,
            "clarity": candidate.clarity,
        }
        if all(value is None for value in criteria.values()):
            base = candidate.score
        else:
            fallback = candidate.score / 10
            filled = {k: (fallback if v is None else v) for k, v in criteria.items()}
            if not mentions_number(answer_text):
                # A number the guest never says earns nothing.
                filled["number_stat"] = 0.0
            base = round(
                10
                * (
                    WEIGHTS["hook_strength"] * filled["question_hook"]
                    + WEIGHTS["comment_potential"] * filled["controversy"]
                    + WEIGHTS["standalone_clarity"] * filled["clarity"]
                    + WEIGHTS["has_number"] * filled["number_stat"]
                )
            )
            criteria = {k: round(v, 1) for k, v in filled.items()}

        penalty, notes, block = 0, [], None
        if self.freshness is not None:
            penalty, notes, block = self.freshness.assess(transcript, boundary)
        if block:
            self._reject(transcript, boundary.start_word, boundary.end_word, block)
            return None

        score = max(0, min(100, base - penalty))
        self._scores[boundary.start_word] = {
            "criteria": criteria,
            "base_score": base,
            "freshness_penalty": penalty,
            "freshness_notes": notes,
        }
        return score

    # -- annotate ----------------------------------------------------------

    def annotate(self, clip: Clip, candidate: ClipCandidate, transcript: Transcript) -> None:
        first, last = self._questions.get(clip.start_word, (clip.start_word, clip.start_word))
        clip.question_text = transcript.text_between(first, last)
        self.details[clip.id] = {
            "question_s": [
                round(transcript.words[first].start, 2),
                round(transcript.words[last].end, 2),
            ],
            "model_question_text": candidate.question_text.strip(),
            **self._scores.get(clip.start_word, {}),
        }

    def empty_message(self) -> str:
        reasons = Counter(r["reason"].split(";")[0] for r in self.rejections)
        top = ", ".join(f"{count}× {reason}" for reason, count in reasons.most_common(3))
        return (
            f"No moment in this episode passed the {self.preset.name} rules "
            f"(opens on the host's question, 25-50 s, score ≥ {self.preset.detection.min_score}, "
            "not already used)." + (f" Rejected: {top}." if top else "")
        )

    def write_trace(self, trace_dir: Path, transcript: Transcript) -> None:
        try:
            trace_dir.mkdir(parents=True, exist_ok=True)
            (trace_dir / "campaign_rejections.json").write_text(
                json.dumps(
                    {"host_check": self.host_check(transcript), "rejections": self.rejections},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning("Could not write the campaign trace: %s", exc)


def configure(config: Any, preset: CampaignPreset, *, freshness: Freshness | None = None) -> Any:
    """A DetectionConfig for this campaign: prompt placeholders and the policy."""
    import dataclasses

    return dataclasses.replace(
        config,
        prompt_vars={
            "HOST": preset.host.name or "the host",
            "TOPICS": ", ".join(preset.detection.topics) or "anything with a strong answer",
        },
        clip_policy=QuestionFirstPolicy(preset, freshness=freshness),
    )
