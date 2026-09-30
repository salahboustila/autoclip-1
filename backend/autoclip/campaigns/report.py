"""``selected_clips.json``: the campaign's record of what was picked and why.

Written to the job's export folder after highlight detection and rewritten
after export (adding file names), so it survives source cleanup and travels
with the clips.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .. import paths
from ..db.models import Clip, Source, utcnow
from .preset import CampaignPreset

log = logging.getLogger(__name__)

FILENAME = "selected_clips.json"


def report_path(job_id: str) -> Path:
    return paths.exports_dir() / job_id / FILENAME


def read(job_id: str) -> dict[str, Any] | None:
    path = report_path(job_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write(
    job_id: str,
    preset: CampaignPreset,
    clips: list[Clip],
    *,
    source: Source | None = None,
    host_check: str | None = None,
    source_check: dict[str, Any] | None = None,
    details: dict[str, dict[str, Any]] | None = None,
    files: dict[str, str] | None = None,
    warnings: list[str] | None = None,
    rejected: list[dict[str, Any]] | None = None,
) -> Path:
    """Write (or refresh) the report. Anything not passed is kept from the last write."""
    previous = read(job_id) or {}
    previous_clips = {c.get("clip_id"): c for c in previous.get("clips", [])}
    details = details or {}
    files = files or {}

    entries = []
    for clip in sorted(clips, key=lambda c: c.rank):
        before = previous_clips.get(clip.id, {})
        detail = details.get(clip.id) or {
            k: before[k]
            for k in (
                "criteria",
                "base_score",
                "freshness_penalty",
                "freshness_notes",
                "question_s",
            )
            if k in before
        }
        entries.append(
            {
                "rank": clip.rank,
                "clip_id": clip.id,
                "start": round(clip.start_s, 2),
                "end": round(clip.end_s, 2),
                "duration": round(clip.duration_s, 2),
                "score": clip.score,
                "topic": clip.topic,
                "question_text": clip.question_text,
                "reason": clip.reason,
                "hook_title": clip.hook_title,
                "title_alternatives": clip.hook_title_alts,
                "caption_file": f"clip_{clip.rank:02d}_caption.txt" if clip.post_caption else None,
                "video_file": files.get(clip.id, before.get("video_file")),
                **detail,
            }
        )

    payload = {
        "campaign": preset.key,
        "campaign_name": preset.name,
        "job_id": job_id,
        "updated_at": utcnow(),
        "source": (
            {"title": source.title, "url": source.url, "duration_s": source.duration_s}
            if source is not None
            else previous.get("source")
        ),
        "source_check": source_check if source_check is not None else previous.get("source_check"),
        "host_check": host_check if host_check is not None else previous.get("host_check"),
        "warnings": warnings if warnings is not None else previous.get("warnings", []),
        "rules": {
            "opens_on_host_question": True,
            "length_s": [preset.detection.min_duration_s, preset.detection.max_duration_s],
            "min_score": preset.detection.min_score,
            "watermark": False,
        },
        "clips": entries,
        # Candidates the model proposed that broke a campaign rule, and why.
        "rejected": rejected if rejected is not None else previous.get("rejected", []),
    }
    path = report_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
