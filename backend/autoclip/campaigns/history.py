"""A per-campaign ledger of moments already made into clips.

One JSON line per exported clip, under ``<root>/campaigns/<preset>/``. Used to
never cut the same moment of an episode twice, across jobs. Deleting a job
forgets its lines: a deleted clip was never posted.
"""

from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path
from typing import Any

from .. import paths
from ..db.models import Clip, Source, utcnow
from .youtube import video_id

FILENAME = "history.jsonl"


def ledger_path(preset_key: str) -> Path:
    return paths.root() / "campaigns" / preset_key / FILENAME


def episode_key(source: Source) -> str:
    """The YouTube id when there is one; otherwise the normalised title."""
    vid = video_id(source.url)
    if vid:
        return f"yt:{vid}"
    title = re.sub(r"[^\w]+", "-", (source.title or source.filename or source.id).lower())
    return f"upload:{title.strip('-')}"


def entries(preset_key: str, episode: str | None = None) -> list[dict[str, Any]]:
    path = ledger_path(preset_key)
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        with contextlib.suppress(json.JSONDecodeError):
            entry = json.loads(line)
            if episode is None or entry.get("episode") == episode:
                out.append(entry)
    return out


def record(preset_key: str, source: Source, clip: Clip) -> None:
    """Add an exported clip, replacing an earlier line for the same clip."""
    episode = episode_key(source)
    kept = [e for e in entries(preset_key) if e.get("clip_id") != clip.id]
    kept.append(
        {
            "episode": episode,
            "job_id": clip.job_id,
            "clip_id": clip.id,
            "start_s": round(clip.start_s, 2),
            "end_s": round(clip.end_s, 2),
            "question_text": clip.question_text,
            "recorded_at": utcnow(),
        }
    )
    _write(preset_key, kept)


def forget_job(job_id: str) -> None:
    """Drop every campaign's lines for a deleted job."""
    base = paths.root() / "campaigns"
    if not base.is_dir():
        return
    for path in base.glob(f"*/{FILENAME}"):
        key = path.parent.name
        remaining = [e for e in entries(key) if e.get("job_id") != job_id]
        _write(key, remaining)


def _write(preset_key: str, items: list[dict[str, Any]]) -> None:
    path = ledger_path(preset_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in items), "utf-8")
    tmp.replace(path)
