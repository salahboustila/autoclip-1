"""Highlight detection — transcript in, ranked clips out.

The transcript is cut into overlapping windows so that long videos fit inside
small context windows, each window is scored by the configured provider, and the
resulting candidates are merged, deduplicated, boundary-refined, and ranked.

Overlap matters: a clip straddling a window edge would otherwise be seen only in
halves by both windows and proposed by neither. The overlap costs tokens and
buys back the clips that live on the seams.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from collections.abc import Callable
from pathlib import Path

from ..db.models import Clip, new_id
from ..providers import ClipCandidate, DetectionConfig, LLMProvider, TranscriptWindow
from ..providers.base import ProviderError
from . import boundaries, viral
from .prepare import Silence
from .transcript import Transcript

log = logging.getLogger(__name__)

#: Window length and overlap in seconds of speech.
WINDOW_S = 8 * 60
OVERLAP_S = 60

#: Two candidates covering this much of the same words are the same clip.
DEDUPE_IOU = 0.4

#: Hosted providers tolerate parallel windows; a local model is already
#: saturating the GPU, so extra concurrency only adds contention.
HOSTED_CONCURRENCY = 3
LOCAL_CONCURRENCY = 1

#: Fallback pass: best moments requested per window, and the most kept overall.
FALLBACK_PER_WINDOW = 3
FALLBACK_MAX_CLIPS = 5


class HighlightError(RuntimeError):
    """Highlight detection produced nothing usable."""


def build_windows(
    transcript: Transcript,
    *,
    window_s: float = WINDOW_S,
    overlap_s: float = OVERLAP_S,
) -> list[TranscriptWindow]:
    """Split a transcript into overlapping windows.

    Preconditions:
        overlap_s is less than window_s, otherwise windows would never advance.
    """
    if not transcript.words:
        return []
    if overlap_s >= window_s:
        raise ValueError("overlap_s must be smaller than window_s")

    windows: list[TranscriptWindow] = []
    total_words = len(transcript.words)
    cursor = 0

    while cursor < total_words:
        window_start_time = transcript.words[cursor].start
        window_end_time = window_start_time + window_s

        last = cursor
        while last + 1 < total_words and transcript.words[last + 1].end <= window_end_time:
            last += 1

        windows.append(_make_window(transcript, cursor, last))

        if last >= total_words - 1:
            break

        # Step forward by (window - overlap), measured in time then converted
        # back to a word index so overlap stays constant regardless of pace.
        next_time = transcript.words[last].end - overlap_s
        next_cursor = max(cursor + 1, transcript.index_at_time(next_time))

        # Overlap only earns its tokens if the next window reaches past this
        # one. Across a gap longer than a window it can't, and stepping back
        # would re-send the tail words alone, a word further each time. There
        # is nothing to straddle across a gap, so start fresh after it.
        if transcript.words[last + 1].end > transcript.words[next_cursor].start + window_s:
            next_cursor = last + 1
        cursor = next_cursor

    return windows


def _make_window(transcript: Transcript, first: int, last: int) -> TranscriptWindow:
    start_s, end_s = transcript.time_range(first, last)
    words = transcript.slice(first, last)
    speakers = sorted({w.speaker for w in words if w.speaker})
    return TranscriptWindow(
        text=render_window_text(transcript, first, last),
        first_word=first,
        last_word=last,
        duration_s=end_s - start_s,
        speakers=speakers,
    )


def render_window_text(transcript: Transcript, first: int, last: int) -> str:
    """Render words as ``[index]word`` so the model can cite exact positions.

    Tagging every word is verbose, but it's the reason the model can't
    miscount — it never has to derive an index, only copy one.
    """
    parts: list[str] = []
    current_speaker: str | None = None

    for index in range(first, min(last + 1, len(transcript.words))):
        word = transcript.words[index]
        if word.speaker and word.speaker != current_speaker:
            current_speaker = word.speaker
            parts.append(f"\n<{word.speaker}>")
        parts.append(f"[{index}]{word.text}")

    return " ".join(parts).strip()


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


async def detect(
    transcript: Transcript,
    provider: LLMProvider,
    config: DetectionConfig,
    *,
    job_id: str,
    silences: list[Silence] | None = None,
    on_progress: Callable[[float], None] | None = None,
    trace_dir: Path | None = None,
) -> list[Clip]:
    """Run detection across the whole transcript and return ranked clips.

    If nothing clears ``config.min_score``, a fallback pass asks each window
    for its best moments regardless of score, and the top few come back marked
    low-confidence rather than failing the job. With ``trace_dir``, every
    window's raw model replies are written there as JSON for debugging.
    """
    windows = build_windows(transcript)
    if not windows:
        raise HighlightError("The transcript is empty, so there is nothing to clip.")

    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        for stale in trace_dir.glob("*.json"):
            stale.unlink()

    log.info("Detecting highlights across %d window(s) with %s.", len(windows), provider.name)
    candidates, failures = await _run_pass(
        "main", windows, transcript, provider, config, trace_dir, on_progress
    )
    if failures == len(windows):
        raise HighlightError(
            f"Every transcript window failed with {provider.name}; see the server log"
            + (f" and the traces in {trace_dir}." if trace_dir else ".")
        )

    log.info("Providers proposed %d raw candidates.", len(candidates))
    clips = build_clips(
        transcript,
        candidates,
        config,
        job_id=job_id,
        silences=silences or [],
        min_score=config.min_score,
    )
    policy = config.clip_policy
    if policy is not None and trace_dir is not None:
        policy.write_trace(trace_dir, transcript)
    if clips:
        return clips
    if policy is not None and not policy.allow_fallback:
        raise HighlightError(policy.empty_message())

    log.warning(
        "Nothing scored %d or higher; asking for the best moments regardless of score.",
        config.min_score,
    )
    fallback_config = dataclasses.replace(config, fallback_clips=FALLBACK_PER_WINDOW)
    candidates, _ = await _run_pass(
        "fallback", windows, transcript, provider, fallback_config, trace_dir, None
    )
    clips = build_clips(
        transcript,
        candidates,
        dataclasses.replace(config, max_clips=min(FALLBACK_MAX_CLIPS, config.max_clips)),
        job_id=job_id,
        silences=silences or [],
        low_confidence=True,
    )
    if not clips:
        raise HighlightError(
            "No clips were found, even on a second pass that ignored the score cut-off. "
            "The model found no self-contained moment in this video — try a different "
            "provider or model."
        )
    return clips


async def _run_pass(
    name: str,
    windows: list[TranscriptWindow],
    transcript: Transcript,
    provider: LLMProvider,
    config: DetectionConfig,
    trace_dir: Path | None,
    on_progress: Callable[[float], None] | None,
) -> tuple[list[ClipCandidate], int]:
    """Send every window once. Returns (candidates, number of failed windows)."""
    concurrency = LOCAL_CONCURRENCY if provider.name == "ollama" else HOSTED_CONCURRENCY
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    failures = 0
    lock = asyncio.Lock()

    async def run_window(index: int, window: TranscriptWindow) -> list[ClipCandidate]:
        nonlocal completed, failures
        async with semaphore:
            attempts: list[dict] = []
            error: str | None = None
            try:
                result = await provider.detect_highlights(window, config)
                candidates = result.clips
                attempts = getattr(result, "attempts", [])
            except Exception as exc:
                # One bad window shouldn't lose the whole video's other windows.
                log.warning(
                    "Window %d-%d failed (%s); continuing with the remaining windows.",
                    window.first_word,
                    window.last_word,
                    exc,
                )
                candidates = []
                error = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, ProviderError):
                    attempts = exc.attempts
            if trace_dir is not None:
                _write_trace(
                    trace_dir / f"{name}_{index:02d}.json",
                    transcript,
                    window,
                    config,
                    attempts,
                    candidates,
                    error,
                )
            async with lock:
                completed += 1
                failures += error is not None
                if on_progress:
                    on_progress(completed / len(windows))
            return candidates

    results = await asyncio.gather(*(run_window(i, w) for i, w in enumerate(windows)))
    return [c for group in results for c in group], failures


def _write_trace(
    path: Path,
    transcript: Transcript,
    window: TranscriptWindow,
    config: DetectionConfig,
    attempts: list[dict],
    candidates: list[ClipCandidate],
    error: str | None,
) -> None:
    """Record what one window sent and got back. Best effort — never fails a job."""
    start_s, end_s = transcript.time_range(window.first_word, window.last_word)
    trace = {
        "first_word": window.first_word,
        "last_word": window.last_word,
        "start_s": round(start_s, 2),
        "end_s": round(end_s, 2),
        "prompt_version": config.prompt_version,
        "min_score": config.min_score,
        "fallback_clips": config.fallback_clips,
        "attempts": attempts,
        "candidates": [c.model_dump() for c in candidates],
        "error": error,
    }
    try:
        path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        log.warning("Could not write highlight trace %s: %s", path, exc)


def build_clips(
    transcript: Transcript,
    candidates: list[ClipCandidate],
    config: DetectionConfig,
    *,
    job_id: str,
    silences: list[Silence] | None = None,
    min_score: int | None = None,
    low_confidence: bool = False,
) -> list[Clip]:
    """Deduplicate, refine, rank, and truncate raw candidates into clips.

    ``min_score`` drops candidates below the cut-off first; the prompt asks the
    model not to return them, but that's a request, not a guarantee.
    """
    # In Viral Hook Mode the cut-off applies to the combined score, which is
    # only known once each clip's final text is; it is enforced further down.
    viral_cutoff = min_score if config.viral_hook else None
    policy = config.clip_policy
    if min_score is not None and viral_cutoff is None:
        kept = [c for c in candidates if c.score >= min_score]
        if len(kept) < len(candidates):
            log.info(
                "Dropped %d candidate(s) scoring below %d.", len(candidates) - len(kept), min_score
            )
        candidates = kept

    deduped = dedupe(candidates)
    log.info("%d candidates remain after dedupe.", len(deduped))

    clips: list[Clip] = []
    for candidate in deduped:
        boundary = boundaries.refine(
            transcript,
            candidate.start_word_index,
            candidate.end_word_index,
            silences=silences or [],
            min_duration_s=config.min_duration_s,
            max_duration_s=config.max_duration_s,
            trim_start=(
                policy.trim_start
                if policy is not None
                else viral.trim_opening
                if config.viral_hook
                else None
            ),
        )
        if boundary is None:
            log.debug(
                "Dropped candidate %d-%d: no valid boundary within the duration range.",
                candidate.start_word_index,
                candidate.end_word_index,
            )
            continue

        score = candidate.score
        if policy is not None:
            policy_score = policy.score(candidate, transcript, boundary)
            if policy_score is None:
                continue
            score = policy_score
            if viral_cutoff is not None and score < viral_cutoff:
                policy.below_cutoff(transcript, boundary, score, viral_cutoff)
                continue
        elif config.viral_hook:
            clip_text = transcript.text_between(boundary.start_word, boundary.end_word)
            score = viral.combined_score(candidate, clip_text)
            if viral_cutoff is not None and score < viral_cutoff:
                log.info(
                    "Dropped a candidate whose combined score %d is below %d.", score, viral_cutoff
                )
                continue

        clip = Clip(
            id=new_id(),
            job_id=job_id,
            start_s=boundary.start_s,
            end_s=boundary.end_s,
            start_word=boundary.start_word,
            end_word=boundary.end_word,
            title=candidate.title.strip()
            or transcript.text_between(
                boundary.start_word, min(boundary.start_word + 8, boundary.end_word)
            ),
            hook=candidate.hook.strip(),
            score=score,
            reason=candidate.reason.strip(),
            low_confidence=low_confidence,
            topic=candidate.topic.strip() if config.viral_hook else "",
        )
        if policy is not None:
            policy.annotate(clip, candidate, transcript)
        clips.append(clip)

    # Refinement can move edges enough that two survivors now overlap.
    clips = _dedupe_clips(clips)
    clips.sort(key=lambda c: c.score, reverse=True)
    clips = clips[: config.max_clips]

    for rank, clip in enumerate(clips, start=1):
        clip.rank = rank

    log.info("Produced %d final clips.", len(clips))
    return clips


def dedupe(
    candidates: list[ClipCandidate], *, iou_threshold: float = DEDUPE_IOU
) -> list[ClipCandidate]:
    """Greedily keep the highest-scoring candidate from each overlapping cluster.

    Overlapping windows mean the same moment is often proposed two or three
    times, sometimes with slightly different edges. Highest score wins.
    """
    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    kept: list[ClipCandidate] = []

    for candidate in ordered:
        if any(
            _iou(
                candidate.start_word_index,
                candidate.end_word_index,
                existing.start_word_index,
                existing.end_word_index,
            )
            > iou_threshold
            for existing in kept
        ):
            continue
        kept.append(candidate)

    return kept


def _dedupe_clips(clips: list[Clip], *, iou_threshold: float = DEDUPE_IOU) -> list[Clip]:
    ordered = sorted(clips, key=lambda c: c.score, reverse=True)
    kept: list[Clip] = []
    for clip in ordered:
        if any(
            _iou(clip.start_word, clip.end_word, other.start_word, other.end_word) > iou_threshold
            for other in kept
        ):
            continue
        kept.append(clip)
    return kept


def _iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Intersection over union of two inclusive index ranges."""
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start) + 1)
    if intersection == 0:
        return 0.0
    union = (a_end - a_start + 1) + (b_end - b_start + 1) - intersection
    return intersection / union if union else 0.0
