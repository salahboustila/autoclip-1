"""Podcast Campaign Mode: per-campaign presets on top of Viral Hook Mode.

A campaign is a YAML preset (``autoclip/presets/<key>.yaml``) describing one
clipping campaign's rules: where clips may come from, what every clip must
open with, the caption tags, and the render. Turning a campaign on switches
Viral Hook Mode on with the preset's values and adds the campaign-only rules
from this package. With it off, nothing here runs.

Kept import-light on purpose: ``config`` and the API import from here.
"""

from .preset import (
    CampaignPreset,
    PresetError,
    apply,
    list_presets,
    load_preset,
    rules_of,
)

__all__ = [
    "CampaignPreset",
    "PresetError",
    "apply",
    "list_presets",
    "load_preset",
    "rules_of",
]
