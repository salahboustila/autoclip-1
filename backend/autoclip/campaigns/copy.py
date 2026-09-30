"""Campaign copy: what the hook-title writer may say about the guest.

Hook titles are limited to what is said in the clip (``pipeline.hookcopy``).
The "authority + shocking claim" pattern also needs who is speaking. For a
podcast episode, the channel's own title names the guest ("Ex-CIA Spy: ...",
"$120M CEO: ...", "... | Alex Hormozi x Jack Neel"), so that descriptor is
passed to the writer as a fact, together with the preset's own facts.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..db.models import Source
    from .preset import CampaignPreset

_QUOTES = "\"“”'‘’("
MAX_DESCRIPTOR_WORDS = 6


def guest_descriptor(title: str, host_aliases: list[str] | None = None) -> str:
    """The guest as the episode title describes them, or "" when it doesn't."""
    title = " ".join((title or "").split())
    head, sep, _ = title.partition(":")
    if sep and head and head[0] not in _QUOTES and len(head.split()) <= MAX_DESCRIPTOR_WORDS:
        return head.strip()
    for alias in sorted(host_aliases or [], key=len, reverse=True):
        match = re.search(
            rf"(?:^|[|│]\s*)([^|│:]+?)\s+x\s+{re.escape(alias)}\b", title, flags=re.IGNORECASE
        )
        if match and len(match.group(1).split()) <= MAX_DESCRIPTOR_WORDS:
            return match.group(1).strip()
    return ""


def campaign_facts(preset: CampaignPreset, source: Source) -> str:
    facts = [preset.copy_.speaker_facts.strip()]
    descriptor = guest_descriptor(source.title, preset.host.aliases)
    if descriptor:
        facts.append(f"The episode title describes the guest as: {descriptor}.")
    return " ".join(f for f in facts if f)
