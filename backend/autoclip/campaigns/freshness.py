"""Freshness: prefer moments that haven't already been clipped to death.

Nothing can know for certain what is already viral elsewhere. These are the
signals that are available, each reported in ``selected_clips.json``:

- **History** (block): the same moment of the same episode was already made
  into a clip by an earlier job of this campaign.
- **The host's own Shorts** (block): the channel's team clips the viral
  moments themselves; a moment matching one of their recent Short titles has
  been posted already.
- **The episode's headline quote** (penalty): the line quoted in the title is
  the most-reposted moment of any episode.
- **YouTube's "most replayed" peaks** (penalty): when yt-dlp reports them.

A lookup that fails is noted as a warning and never fails the job.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..pipeline.hookcopy import content_words, vocabulary
from ..pipeline.transcript import Transcript
from . import history, youtube

if TYPE_CHECKING:
    from ..config import IngestSettings
    from ..db.models import Source
    from ..pipeline.boundaries import Boundary
    from .preset import CampaignPreset

log = logging.getLogger(__name__)

#: Share of a previous clip's time a candidate must cover to count as the same moment.
SAME_MOMENT_OVERLAP = 0.5
#: Share of a Short title's content words a clip must contain to be that Short.
SHORT_MATCH = 0.6
#: Share of the headline quote's content words that marks the headline moment.
HEADLINE_MATCH = 0.6
#: Heatmap points at or above this (YouTube normalises the curve to 0-1) are peaks.
PEAK_VALUE = 0.6
#: Share of a clip inside replay peaks that marks it as a most-replayed moment.
PEAK_OVERLAP = 0.3

_QUOTE = re.compile(r"[\"“”]([^\"“”]{6,})[\"“”]")


def headline_quote(title: str) -> str:
    """The quoted line in an episode title ('Guest: "Quote!" Rest | Show')."""
    match = _QUOTE.search(title or "")
    return match.group(1).strip() if match else ""


def _match_ratio(phrase: str, clip_vocabulary: set[str]) -> tuple[float, int]:
    from ..pipeline.hookcopy import stem

    words = {stem(w) for w in content_words(phrase)}
    if not words:
        return 0.0, 0
    return len(words & clip_vocabulary) / len(words), len(words)


def _peaks(heatmap: list[dict[str, float]]) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for point in sorted(heatmap, key=lambda p: p["start"]):
        if point["value"] < PEAK_VALUE:
            continue
        if ranges and point["start"] <= ranges[-1][1] + 0.01:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], point["end"]))
        else:
            ranges.append((point["start"], point["end"]))
    return ranges


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


@dataclass
class Freshness:
    headline_penalty: int = 20
    most_replayed_penalty: int = 15
    previous: list[dict] = field(default_factory=list)
    shorts: list[str] = field(default_factory=list)
    headline: str = ""
    peaks: list[tuple[float, float]] = field(default_factory=list)
    #: Lookups that couldn't be made, for the report.
    warnings: list[str] = field(default_factory=list)

    def assess(
        self, transcript: Transcript, boundary: Boundary
    ) -> tuple[int, list[str], str | None]:
        """``(penalty, notes, block reason or None)`` for one candidate."""
        span = (boundary.start_s, boundary.end_s)
        length = max(0.01, span[1] - span[0])

        for entry in self.previous:
            earlier = (entry["start_s"], entry["end_s"])
            shorter = max(0.01, min(length, earlier[1] - earlier[0]))
            if _overlap(span, earlier) / shorter >= SAME_MOMENT_OVERLAP:
                return 0, [], f"already clipped in job {entry['job_id'][:8]}"

        clip_vocabulary = vocabulary(
            transcript.text_between(boundary.start_word, boundary.end_word)
        )
        for title in self.shorts:
            ratio, size = _match_ratio(title, clip_vocabulary)
            if size >= 3 and ratio >= SHORT_MATCH:
                return 0, [], f"matches the host's Short “{title}”"

        penalty, notes = 0, []
        ratio, size = _match_ratio(self.headline, clip_vocabulary)
        if size >= 2 and ratio >= HEADLINE_MATCH:
            penalty += self.headline_penalty
            notes.append(f"contains the episode's headline quote (−{self.headline_penalty})")

        inside = sum(_overlap(span, peak) for peak in self.peaks)
        if self.peaks and inside / length >= PEAK_OVERLAP:
            penalty += self.most_replayed_penalty
            notes.append(f"inside YouTube's most-replayed peak (−{self.most_replayed_penalty})")
        return penalty, notes, None


def load(
    preset: CampaignPreset,
    source: Source,
    job_id: str,
    ingest_settings: IngestSettings | None = None,
) -> Freshness:
    """Gather every freshness signal for one episode. Never raises."""
    rules = preset.freshness
    fresh = Freshness(
        headline_penalty=rules.headline_penalty,
        most_replayed_penalty=rules.most_replayed_penalty,
    )
    if rules.history:
        episode = history.episode_key(source)
        fresh.previous = [
            e for e in history.entries(preset.key, episode) if e.get("job_id") != job_id
        ]
    if rules.headline_penalty:
        fresh.headline = headline_quote(source.title)
    if rules.shorts_url and rules.shorts_to_check:
        try:
            fresh.shorts = [
                v["title"]
                for v in youtube.channel_videos(
                    rules.shorts_url,
                    rules.shorts_to_check,
                    settings=ingest_settings,
                    ttl_s=youtube.SHORTS_TTL_S,
                )
                if v.get("title")
            ]
        except Exception as exc:
            fresh.warnings.append(f"Could not read the host's Shorts ({exc}).")
    if rules.most_replayed_penalty and youtube.video_id(source.url):
        try:
            fresh.peaks = _peaks(youtube.replay_heatmap(source.url, settings=ingest_settings))
            if not fresh.peaks:
                fresh.warnings.append("YouTube reported no most-replayed data for this episode.")
        except Exception as exc:
            fresh.warnings.append(f"Could not read YouTube's most-replayed data ({exc}).")
    for warning in fresh.warnings:
        log.warning("%s", warning)
    return fresh
