"""Source rules: clips may only come from the channel's latest episodes.

A YouTube link is checked against the channel's newest ``latest_episodes``
uploads (the Videos tab, so Shorts never count). An uploaded file can't be
checked; it is allowed when the preset allows uploads, and marked
"not verified" everywhere it shows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

from . import youtube

if TYPE_CHECKING:
    from ..config import IngestSettings
    from ..db.models import Source
    from .preset import CampaignPreset

Status = Literal["verified", "not_verified", "rejected"]


@dataclass
class SourceCheck:
    status: Status
    message: str
    episode: dict[str, Any] | None = None

    @property
    def allowed(self) -> bool:
        return self.status != "rejected"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def latest_episodes(
    preset: CampaignPreset, ingest_settings: IngestSettings | None = None
) -> list[dict[str, Any]]:
    """The channel's newest episodes, newest first. Raises when YouTube can't be read."""
    return youtube.channel_videos(
        preset.source.channel_url, preset.source.latest_episodes, settings=ingest_settings
    )


def check_url(
    preset: CampaignPreset, url: str, ingest_settings: IngestSettings | None = None
) -> SourceCheck:
    count = preset.source.latest_episodes
    vid = youtube.video_id(url)
    if vid is None:
        return SourceCheck("rejected", "That isn't a YouTube video link.")
    try:
        episodes = latest_episodes(preset, ingest_settings)
    except Exception as exc:
        return SourceCheck(
            "not_verified",
            f"Couldn't read the channel's latest episodes, so this link is not verified ({exc}).",
        )
    for position, episode in enumerate(episodes, start=1):
        if episode["id"] == vid:
            return SourceCheck(
                "verified",
                f"Episode {position} of {preset.host.name or 'the channel'}'s latest {count}.",
                episode,
            )
    return SourceCheck(
        "rejected",
        f"This video is not one of {preset.host.name or 'the channel'}'s latest {count} episodes, "
        f"so clips from it don't qualify for the {preset.name} campaign.",
    )


def check_source(
    preset: CampaignPreset, source: Source, ingest_settings: IngestSettings | None = None
) -> SourceCheck:
    if source.type == "youtube" and source.url:
        return check_url(preset, source.url, ingest_settings)
    if not preset.source.allow_uploads:
        return SourceCheck("rejected", f"The {preset.name} campaign doesn't allow uploaded files.")
    return SourceCheck(
        "not_verified",
        f"Uploaded file: source not verified. Make sure it is one of "
        f"{preset.host.name or 'the channel'}'s latest {preset.source.latest_episodes} episodes.",
    )
