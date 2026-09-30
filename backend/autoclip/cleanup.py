"""Removing a job's files, and packaging its results.

Two callers: the "delete job" endpoint, which removes everything, and the
runner's ``cleanup.auto_delete_sources`` step, which removes only the source
video and intermediates once the final clips exist.

Nothing outside AutoClip's own directories is ever deleted. A source whose file
lives elsewhere (a path the user pointed at) is left alone; only media AutoClip
downloaded or received as an upload under ``media/`` is removed.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from . import paths
from .db import store
from .db.models import Job, Source

log = logging.getLogger(__name__)

#: Job states in which a source may still be read by the pipeline.
_ACTIVE = ("queued", "running")


def _remove_tree(path: Path) -> bool:
    if not path.exists():
        return False
    shutil.rmtree(path, ignore_errors=True)
    return True


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except (ValueError, OSError):
        return False
    return True


def remove_source_media(source: Source) -> bool:
    """Delete a source's video, if AutoClip owns it. Returns True if anything went."""
    source_path = Path(source.path)
    if not _is_within(source_path, paths.media_dir()):
        return False
    directory = paths.source_media_dir(source.id)
    if _is_within(source_path, directory):
        return _remove_tree(directory)
    if source_path.exists():
        source_path.unlink()
        return True
    return False


def remove_work_files(job_id: str) -> bool:
    return _remove_tree(paths.job_work_dir(job_id))


def source_in_use(source_id: str, *, excluding_job: str) -> bool:
    """True if another queued or running job still needs this source."""
    return any(
        job.id != excluding_job and job.status in _ACTIVE
        for job in store.list_jobs_for_source(source_id)
    )


def purge_after_export(job: Job, source: Source) -> None:
    """Drop the source video and work files, keeping only the exported clips."""
    if not source_in_use(source.id, excluding_job=job.id) and remove_source_media(source):
        log.info("Deleted source media for job %s.", job.id)
    if remove_work_files(job.id):
        log.info("Deleted work files for job %s.", job.id)


def delete_job(job: Job) -> None:
    """Remove a job completely: exports, work files, DB rows, and its source.

    The source goes too unless another job still uses it.
    """
    from .campaigns import history

    source = store.get_source(job.source_id)
    history.forget_job(job.id)
    _remove_tree(paths.exports_dir() / job.id)
    remove_work_files(job.id)
    store.delete_job(job.id)

    if source is not None and not store.list_jobs_for_source(source.id):
        remove_source_media(source)
        store.delete_source(source.id)


# --------------------------------------------------------------------------
# Download-all archive
# --------------------------------------------------------------------------


def collect_job_files(job_id: str) -> list[Path]:
    """The newest existing export per clip, plus caption text files."""
    files: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        if path.is_file() and path not in seen:
            seen.add(path)
            files.append(path)

    for clip in store.list_clips(job_id):
        for record in store.list_exports(clip.id):  # newest first
            path = Path(record.path)
            if path.is_file():
                add(path)
                break

    export_dir = paths.exports_dir() / job_id
    for caption in sorted(export_dir.glob("*_caption.txt")):
        add(caption)
    return files


def build_zip(job_id: str) -> Path | None:
    """Zip a job's clips into a temp file. Returns None when there is nothing."""
    files = collect_job_files(job_id)
    if not files:
        return None
    fd, name = tempfile.mkstemp(prefix="autoclip-", suffix=".zip")
    os.close(fd)
    archive = Path(name)
    # MP4 is already compressed, so deflating would cost time and save nothing.
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
        for path in files:
            zf.write(path, arcname=path.name)
    return archive
