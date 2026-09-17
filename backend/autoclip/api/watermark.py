"""Watermark image upload, removal, and preview.

The image itself lives on disk (``paths.watermarks_dir()``); everything about
*how* it's drawn — position, size, opacity, on/off — lives in
``ExportSettings.watermark`` alongside the rest of the export configuration
(see :mod:`autoclip.config`), so those fields already round-trip through the
existing ``PUT /api/settings``. This router only owns the file itself:
uploading it, serving it back for the placement preview, and clearing it.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .. import paths
from ..config import WatermarkSettings
from ..config import load as load_settings
from ..config import save as save_settings
from .schemas import SettingsOut
from .settings import settings_out

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/watermark", tags=["watermark"])

#: PNG for transparency, JPG for a logo that's already opaque. Uploads are
#: small (a logo, not a video), so no chunk-size tuning is needed here the way
#: sources.py needs it for multi-gigabyte media.
ACCEPTED_SUFFIXES = {".png", ".jpg", ".jpeg"}
UPLOAD_CHUNK = 1024 * 1024

_MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


@router.post("", response_model=SettingsOut, status_code=201)
async def upload_watermark(file: UploadFile = File(...)) -> SettingsOut:
    """Store a watermark image, replacing any previous one, and turn it on."""
    filename = Path(file.filename or "watermark")
    suffix = filename.suffix.lower()
    if suffix not in ACCEPTED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail={
                "message": f"{suffix or 'This file'} is not a supported image format.",
                "hint": f"Accepted: {', '.join(sorted(ACCEPTED_SUFFIXES))}",
            },
        )

    directory = await asyncio.to_thread(paths.ensure_watermarks_dir)
    destination = directory / f"watermark{suffix}"

    # Staged in the same directory so the final replace is same-filesystem and
    # atomic, and a failed upload never leaves a half-written file behind.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=directory) as staging:
        staging_path = Path(staging.name)
        try:
            while chunk := await file.read(UPLOAD_CHUNK):
                staging.write(chunk)
        except Exception as exc:
            staging_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="Upload failed.") from exc

    def replace() -> None:
        # Only one watermark is kept. A re-upload in a different format (jpg
        # after a previous png, say) must remove the old extension too, or
        # both would exist and whichever config.json points at wins silently.
        for old in directory.glob("watermark.*"):
            if old != staging_path:
                old.unlink(missing_ok=True)
        staging_path.replace(destination)

    await asyncio.to_thread(replace)

    settings = await asyncio.to_thread(load_settings)
    settings.export.watermark = settings.export.watermark.model_copy(
        update={"filename": destination.name, "enabled": True}
    )
    await asyncio.to_thread(save_settings, settings)

    return settings_out(settings)


@router.delete("", response_model=SettingsOut)
async def remove_watermark() -> SettingsOut:
    """Delete the stored image and reset watermark settings to their defaults."""
    settings = await asyncio.to_thread(load_settings)
    path = settings.export.watermark.resolved_path()
    if path is not None:
        await asyncio.to_thread(lambda: path.unlink(missing_ok=True))

    settings.export.watermark = WatermarkSettings()
    await asyncio.to_thread(save_settings, settings)

    return settings_out(settings)


@router.get("/file")
async def watermark_file() -> FileResponse:
    """Serve the stored image, for the placement preview to render as an <img>."""
    settings = await asyncio.to_thread(load_settings)
    path = settings.export.watermark.resolved_path()
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="No watermark uploaded.")

    media_type = _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media_type)
