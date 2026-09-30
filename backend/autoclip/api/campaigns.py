"""Podcast Campaign Mode: presets and their source rules."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from .. import campaigns
from ..campaigns.preset import preset_summary

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
