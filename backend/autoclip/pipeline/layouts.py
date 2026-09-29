"""The Podcast Hook layout (Viral Hook Mode), 1080×1920.

    ┌──────────────┐  0
    │  HOOK TITLE  │  title band: 12% down to the top of the video
    │ ┌──────────┐ │
    │ │  video   │ │  full width, centred vertically
    │ └──────────┘ │
    │   captions   │  caption band: under the video, above the platform UI
    └──────────────┘  1920

Black background, no watermark, no channel handle. Everything is placed from
the video box, so a wide 16:9 source and a square tracked crop both lay out
without the title or captions touching the picture.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass

from .captions import CaptionStyle
from .reframe.croppath import CropPath, centre_crop
from .viral import mentions_number

WIDTH, HEIGHT = 1080, 1920
#: Keep captions clear of TikTok/Reels' bottom UI (caption text, buttons).
BOTTOM_SAFE_RATIO = 0.09
#: Space between the video and the title or captions.
GAP_PX = 36

KEYWORD_COLOURS = {"green": "#31E981", "yellow": "#FFE500"}


def _even(value: float) -> int:
    return max(2, int(round(value / 2)) * 2)


@dataclass(frozen=True)
class PodcastHookLayout:
    #: Where the video sits: (x, y, w, h). x is always 0 — full width.
    video_box: tuple[int, int, int, int]
    #: "full_width" shows the whole source frame; "tracked" uses the face-
    #: tracked square crop from the reframe stage.
    fit: str
    width: int = WIDTH
    height: int = HEIGHT

    @property
    def video_top(self) -> int:
        return self.video_box[1]

    @property
    def video_bottom(self) -> int:
        return self.video_box[1] + self.video_box[3]

    @property
    def title_limit_px(self) -> int:
        """Lowest pixel the title may reach."""
        return self.video_top - GAP_PX

    @property
    def caption_band(self) -> tuple[int, int]:
        return self.video_bottom + GAP_PX, round(self.height * (1 - BOTTOM_SAFE_RATIO))

    @property
    def crop_aspect(self) -> tuple[int, int]:
        """Aspect of the source region that fills the video box."""
        return self.video_box[2], self.video_box[3]


def podcast_hook_layout(
    source_w: int, source_h: int, *, fit: str = "full_width"
) -> PodcastHookLayout:
    """Place the video for a source of the given size.

    ``full_width`` keeps the source's aspect up to square: a 16:9 podcast
    becomes a 1080×608 strip. A source taller than square (already vertical)
    is centre-cropped to square so there's still room for the title and
    captions. ``tracked`` is always a 1080×1080 square.
    """
    if fit not in ("full_width", "tracked"):
        raise ValueError(f"Unknown fit {fit!r}; expected 'full_width' or 'tracked'.")
    if fit == "tracked" or not source_w or not source_h:
        box_h = WIDTH
    else:
        box_h = min(_even(WIDTH * source_h / source_w), WIDTH)
    top = _even((HEIGHT - box_h) / 2)
    return PodcastHookLayout(video_box=(0, top, WIDTH, box_h), fit=fit)


def caption_style_for(
    layout: PodcastHookLayout, base: CaptionStyle, *, keyword: str = "green"
) -> CaptionStyle:
    """The chosen caption preset, adapted to sit centred in the caption band.

    One line per group (at most three words), the spoken word pops, and one
    keyword per line is coloured.
    """
    band_top, band_bottom = layout.caption_band
    font_px = layout.height * base.size_ratio
    growth = base.scale_percent / 100 if base.animation == "scale" else 1.0
    line_px = font_px * 1.2 * growth
    # Bottom-anchored: put the single line's centre in the middle of the band.
    centre = (band_top + band_bottom) / 2
    bottom = min(band_bottom, centre + line_px / 2)
    return dataclasses.replace(
        base,
        margin_v_ratio=(layout.height - bottom) / layout.height,
        max_words=min(base.max_words, 3),
        keyword_colour=KEYWORD_COLOURS.get(keyword, keyword),
        # The keyword owns the colour; the active word only grows.
        accent="#FFFFFF" if base.accent else None,
    )


def title_style_key(font: str) -> str:
    return "podcast_montserrat" if font == "montserrat" else "podcast_anton"


def title_keyword(title: str) -> list[str]:
    """The word to paint yellow: a number if there is one, else the longest
    content word. Returns [] when nothing qualifies."""
    from .hookcopy import FRAMING_WORDS

    words = [w for w in re.split(r"\s+", title) if re.search(r"\w", w)]
    for word in words:
        if mentions_number(word):
            return [word]
    candidates = [
        w
        for w in words
        if len(re.sub(r"[^\w]", "", w)) >= 4
        and re.sub(r"[^\w']", "", w.lower()) not in FRAMING_WORDS
    ]
    if not candidates:
        return []
    return [max(candidates, key=lambda w: len(re.sub(r"[^\w]", "", w)))]


@dataclass(frozen=True)
class PodcastOptions:
    """Everything export needs to render one clip in the Podcast Hook layout."""

    layout: PodcastHookLayout
    title_style: str = "podcast_anton"
    highlight_title_keyword: bool = True
    caption_keyword_colour: str = "green"


def podcast_options(settings, source_w: int | None, source_h: int | None) -> PodcastOptions | None:
    """The layout to use for a job's settings, or None for the standard one."""
    viral = settings.viral_hook
    if not (viral.enabled and viral.layout == "podcast_hook"):
        return None
    return PodcastOptions(
        layout=podcast_hook_layout(source_w or 1920, source_h or 1080, fit=viral.video_fit),
        title_style=title_style_key(viral.title_font),
        highlight_title_keyword=viral.highlight_title_keyword,
        caption_keyword_colour=viral.caption_keyword_colour,
    )


def podcast_crop_path(
    options: PodcastOptions,
    source_w: int,
    source_h: int,
    duration_s: float,
    tracked: CropPath | None = None,
) -> CropPath:
    """The source region that fills the video box.

    ``tracked`` (a square crop path from the reframe stage) is used as is in
    tracked mode; otherwise the centre of the frame at the box's aspect, which
    for a 16:9 source in full-width mode is simply the whole frame.
    """
    if options.layout.fit == "tracked" and tracked is not None:
        return tracked
    box_w, box_h = options.layout.crop_aspect
    return centre_crop(source_w, source_h, duration_s, aspect_w=box_w, aspect_h=box_h)
