"""Podcast Campaign Mode: presets and their source rules."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import campaigns
from ..campaigns import source as campaign_source
from ..campaigns.preset import preset_summary
from ..config import load as load_settings

router = APIRouter(prefix="/api/campaigns", tags=["campaigns"])


def _preset_or_404(key: str) -> campaigns.CampaignPreset:
    try:
        return campaigns.load_preset(key)
    except campaigns.PresetError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("")
async def list_campaigns() -> list[dict]:
    return await asyncio.to_thread(campaigns.list_presets)


@router.get("/{key}")
async def get_campaign(key: str) -> dict:
    return preset_summary(await asyncio.to_thread(_preset_or_404, key))


@router.get("/{key}/episodes")
async def latest_episodes(key: str) -> list[dict]:
    """The channel's latest episodes, for the episode picker."""
    preset = await asyncio.to_thread(_preset_or_404, key)
    try:
        return await asyncio.to_thread(
            campaign_source.latest_episodes, preset, load_settings().ingest
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Couldn't read the channel's latest episodes: {exc}"
        ) from exc


class UrlCheckIn(BaseModel):
    url: str


@router.post("/{key}/check-url")
async def check_url(key: str, payload: UrlCheckIn) -> dict:
    """Check a link against the campaign's source rules before downloading it."""
    preset = await asyncio.to_thread(_preset_or_404, key)
    check = await asyncio.to_thread(
        campaign_source.check_url, preset, payload.url, load_settings().ingest
    )
    return check.as_dict()
