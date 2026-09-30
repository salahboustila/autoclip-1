"""Campaign presets: loading, validation, and applying one to settings."""

from __future__ import annotations

import functools
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

from .. import paths

if TYPE_CHECKING:
    from ..config import Settings

log = logging.getLogger(__name__)

#: Presets that ship with AutoClip.
BUNDLED_DIR = Path(__file__).resolve().parent.parent / "presets"
_KEY = re.compile(r"^[a-z0-9_]+$")


class PresetError(ValueError):
    """A preset is missing or invalid."""


class SourceRules(BaseModel):
    channel_url: str = ""
    latest_episodes: int = Field(default=20, ge=1, le=200)
    allow_uploads: bool = True


class HostRules(BaseModel):
    name: str = ""
    aliases: list[str] = Field(default_factory=list)


class DetectionRules(BaseModel):
    prompt: str = "campaign_question_first_v1"
    topics: list[str] = Field(default_factory=list)
    min_duration_s: float = Field(default=25.0, gt=0)
    max_duration_s: float = Field(default=50.0, gt=0)
    top_n: int = Field(default=10, ge=1, le=50)
    min_score: int = Field(default=70, ge=0, le=100)
    max_question_s: float = Field(default=10.0, gt=0)
    fallback_pass: bool = False
    diarization: bool = True


class FreshnessRules(BaseModel):
    history: bool = True
    shorts_url: str = ""
    shorts_to_check: int = Field(default=60, ge=0, le=500)
    headline_penalty: int = Field(default=20, ge=0, le=100)
    most_replayed_penalty: int = Field(default=15, ge=0, le=100)


class CopyRules(BaseModel):
    speaker_facts: str = ""
    caption_lines: list[str] = Field(default_factory=list)


class RenderRules(BaseModel):
    layout: Literal["podcast_hook", "standard"] = "podcast_hook"
    video_fit: Literal["full_width", "tracked"] = "full_width"
    title_font: Literal["anton", "montserrat"] = "anton"
    highlight_title_keyword: bool = True
    caption_keyword_colour: Literal["green", "yellow"] = "green"
    #: Only False is supported: campaigns never burn in a logo or handle.
    watermark: Literal[False] = False


class CampaignPreset(BaseModel):
    key: str
    name: str
    platform: str = ""
    notes: str = ""
    source: SourceRules = Field(default_factory=SourceRules)
    host: HostRules = Field(default_factory=HostRules)
    detection: DetectionRules = Field(default_factory=DetectionRules)
    freshness: FreshnessRules = Field(default_factory=FreshnessRules)
    copy_: CopyRules = Field(default_factory=CopyRules, alias="copy")
    render: RenderRules = Field(default_factory=RenderRules)

    model_config = {"populate_by_name": True, "extra": "forbid"}


def preset_dirs() -> list[Path]:
    """User presets first, so a local file overrides a bundled one."""
    return [paths.root() / "presets", BUNDLED_DIR]


def _find(key: str) -> Path | None:
    for directory in preset_dirs():
        path = directory / f"{key}.yaml"
        if path.is_file():
            return path
    return None


def load_preset(key: str) -> CampaignPreset:
    if not _KEY.match(key or ""):
        raise PresetError(f"Invalid preset name {key!r}.")
    path = _find(key)
    if path is None:
        available = ", ".join(p["key"] for p in list_presets()) or "none"
        raise PresetError(f"No campaign preset named {key!r}. Available: {available}.")
    return _load_file(path, path.stat().st_mtime)


@functools.lru_cache(maxsize=16)
def _load_file(path: Path, _mtime: float) -> CampaignPreset:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        preset = CampaignPreset.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise PresetError(f"Campaign preset {path} is invalid: {exc}") from exc
    if preset.key != path.stem:
        raise PresetError(f"Preset {path} declares key {preset.key!r}; it must match the filename.")
    if preset.detection.min_duration_s >= preset.detection.max_duration_s:
        raise PresetError(f"Preset {path}: min_duration_s must be below max_duration_s.")
    return preset


def list_presets() -> list[dict[str, str]]:
    """Every loadable preset as ``{"key", "name", "platform"}``, sorted by name."""
    seen: dict[str, dict[str, str]] = {}
    for directory in preset_dirs():
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.yaml")):
            if path.stem in seen:
                continue
            try:
                preset = _load_file(path, path.stat().st_mtime)
            except PresetError as exc:
                log.warning("%s", exc)
                continue
            seen[path.stem] = {"key": preset.key, "name": preset.name, "platform": preset.platform}
    return sorted(seen.values(), key=lambda p: p["name"])


def rules_of(settings: Settings) -> CampaignPreset | None:
    """The campaign a job runs under, from its settings snapshot, or None."""
    campaign = settings.campaign
    if not campaign.enabled or campaign.rules is None:
        return None
    return CampaignPreset.model_validate(campaign.rules)


def apply(settings: Settings) -> Settings:
    """Settings with the campaign's preset laid over Viral Hook Mode.

    Returns ``settings`` itself when no campaign is on, so the default path is
    untouched. Idempotent: once the preset is snapshotted into
    ``campaign.rules`` (at job creation), later calls change nothing, and a
    retried job keeps the rules it started with.
    """
    campaign = settings.campaign
    if not campaign.enabled or campaign.rules is not None:
        return settings

    preset = load_preset(campaign.preset)
    merged = settings.model_copy(deep=True)
    merged.campaign.rules = preset.model_dump(mode="json", by_alias=True)

    detection, render = preset.detection, preset.render
    viral = merged.viral_hook
    viral.enabled = True
    viral.prompt_version = detection.prompt
    viral.top_n = detection.top_n
    viral.min_duration_s = detection.min_duration_s
    viral.max_duration_s = detection.max_duration_s
    viral.speaker_facts = preset.copy_.speaker_facts
    # The caption footer comes from the preset's lines instead.
    viral.caption_handle = ""
    viral.fixed_hashtags = []
    viral.layout = render.layout
    viral.video_fit = render.video_fit
    viral.title_font = render.title_font
    viral.highlight_title_keyword = render.highlight_title_keyword
    viral.caption_keyword_colour = render.caption_keyword_colour

    merged.clips.min_score = detection.min_score
    merged.hook_title.enabled = True
    if detection.diarization:
        merged.whisper.diarization = True
    return merged


def preset_summary(preset: CampaignPreset) -> dict[str, Any]:
    """What the UI shows about a preset."""
    return {
        "key": preset.key,
        "name": preset.name,
        "platform": preset.platform,
        "notes": preset.notes,
        "channel_url": preset.source.channel_url,
        "latest_episodes": preset.source.latest_episodes,
        "allow_uploads": preset.source.allow_uploads,
        "host": preset.host.name,
        "caption_lines": preset.copy_.caption_lines,
        "min_duration_s": preset.detection.min_duration_s,
        "max_duration_s": preset.detection.max_duration_s,
        "top_n": preset.detection.top_n,
        "min_score": preset.detection.min_score,
    }
