"""Hook title — a headline held on screen for the whole clip.

libass and ffmpeg's ``drawtext`` cannot draw colour emoji, and a hook title
without its emoji reads as flat. So the title is drawn with Pillow onto a
transparent full-frame PNG, with text and emoji split into runs (Montserrat
for text, the bundled Noto Color Emoji for emoji), and ffmpeg overlays that
PNG over every frame.

The layout guarantees three things regardless of title length: at most two
lines, never higher than :data:`MIN_TOP_PCT` of the frame, and never reaching
down into the caption area. A long title shrinks to fit rather than breaking
any of them.
"""

from __future__ import annotations

import functools
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .captions import FONT_DIR, CaptionStyle

log = logging.getLogger(__name__)

EMOJI_FONT = FONT_DIR / "emoji" / "NotoColorEmoji.ttf"
#: Noto Color Emoji is a bitmap (CBDT) font with a single strike; FreeType
#: refuses any other size, so emoji are drawn at this size and then scaled.
EMOJI_STRIKE = 109

#: The title never starts higher than this, as a percentage of frame height.
#: Above it, platform UI (the TikTok tabs, the Reels header) covers the text.
MIN_TOP_PCT = 12.0
DEFAULT_TOP_PCT = 13.0

#: Font sizes are configured against a 1080-wide frame and scaled to the output.
REFERENCE_WIDTH = 1080
DEFAULT_FONT_SIZE = 64
#: A title shrinks to fit, but never below this fraction of its configured size.
MIN_SHRINK = 0.5

#: Clear space kept between the bottom of the title and the top of the captions.
CAPTION_GAP_RATIO = 0.02


class HookTitleError(RuntimeError):
    """The title could not be laid out."""


@dataclass(frozen=True)
class HookTitleStyle:
    key: str
    font_file: str = "Montserrat-ExtraBold.ttf"
    text_colour: str = "#000000"
    #: Background box colour; None draws text straight onto the frame.
    box_colour: str | None = "#FFFFFF"
    #: Colour for highlighted key words; None disables highlighting.
    highlight_colour: str | None = None
    all_caps: bool = False
    max_lines: int = 2
    #: Widest a line may be, as a fraction of frame width (box included).
    max_width_ratio: float = 0.88
    #: Distance between baselines, in ems.
    line_spacing: float = 1.18
    #: Box padding and corner radius, in ems.
    pad_x: float = 0.55
    pad_y: float = 0.38
    radius: float = 0.38
    #: Emoji height relative to the font size.
    emoji_scale: float = 0.98

    @property
    def font_path(self) -> Path:
        return FONT_DIR / self.font_file


STYLES: dict[str, HookTitleStyle] = {
    # Default mode: black bold text in a white rounded box.
    "boxed": HookTitleStyle(key="boxed"),
    # Podcast Hook layout: bold white type straight on the black band, with the
    # key word in yellow.
    "podcast_anton": HookTitleStyle(
        key="podcast_anton",
        font_file="Anton-Regular.ttf",
        text_colour="#FFFFFF",
        box_colour=None,
        highlight_colour="#FFE500",
        all_caps=True,
        max_width_ratio=0.92,
        line_spacing=1.1,
        pad_x=0.2,
        pad_y=0.15,
    ),
    "podcast_montserrat": HookTitleStyle(
        key="podcast_montserrat",
        font_file="Montserrat-ExtraBold.ttf",
        text_colour="#FFFFFF",
        box_colour=None,
        highlight_colour="#FFE500",
        max_width_ratio=0.92,
        line_spacing=1.15,
        pad_x=0.2,
        pad_y=0.15,
    ),
}


def get_style(key: str) -> HookTitleStyle:
    style = STYLES.get(key)
    if style is None:
        raise ValueError(f"Unknown hook title style {key!r}. Available: {', '.join(STYLES)}")
    return style


# --------------------------------------------------------------------------
# Text / emoji runs
# --------------------------------------------------------------------------

_EMOJI_BASE = (
    "\U0001f000-\U0001faff"  # pictographs, emoticons, transport, symbols & pictographs ext.
    "☀-➿"  # misc symbols, dingbats
    "⌀-⏿"  # misc technical (⌚, ⏰)
    "⬀-⯿"  # arrows, stars (⭐, ⬆)
    "←-⇿"  # arrows
    "⤴⤵〰〽㊗㊙‼⁉"
)
_EMOJI_MODIFIERS = "️︎⃣\U0001f3fb-\U0001f3ff"
#: One emoji cluster: a flag pair, a keycap, or a base with modifiers, joined
#: by zero-width joiners into sequences like 👨‍👩‍👧.
EMOJI_RE = re.compile(
    "(?:"
    "[\U0001f1e6-\U0001f1ff]{2}"
    "|[0-9#*]️?⃣"
    f"|[{_EMOJI_BASE}][{_EMOJI_MODIFIERS}]*(?:‍[{_EMOJI_BASE}][{_EMOJI_MODIFIERS}]*)*"
    ")"
)


def split_runs(text: str) -> list[tuple[str, bool]]:
    """Split text into ``(run, is_emoji)`` pieces, one piece per emoji cluster."""
    runs: list[tuple[str, bool]] = []
    cursor = 0
    for match in EMOJI_RE.finditer(text):
        if match.start() > cursor:
            runs.append((text[cursor : match.start()], False))
        runs.append((match.group(), True))
        cursor = match.end()
    if cursor < len(text):
        runs.append((text[cursor:], False))
    return runs


def has_emoji(text: str) -> bool:
    return EMOJI_RE.search(text) is not None


# --------------------------------------------------------------------------
# Glyph drawing
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


@functools.lru_cache(maxsize=256)
def _emoji_bitmap(cluster: str) -> Image.Image:
    """One emoji drawn at the font's native strike, cropped to its ink."""
    font = _font(str(EMOJI_FONT), EMOJI_STRIKE)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    left, top, right, bottom = probe.textbbox((0, 0), cluster, font=font, embedded_color=True)
    image = Image.new("RGBA", (max(1, right - left), max(1, bottom - top)), (0, 0, 0, 0))
    ImageDraw.Draw(image).text((-left, -top), cluster, font=font, embedded_color=True)
    return image


def _emoji_image(cluster: str, height: int) -> Image.Image:
    bitmap = _emoji_bitmap(cluster)
    width = max(1, round(bitmap.width * height / bitmap.height))
    return bitmap.resize((width, height), Image.LANCZOS)


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------


@dataclass
class _Word:
    text: str
    runs: list[tuple[str, bool]]
    width: float
    highlight: bool = False


@dataclass
class TitleLayout:
    lines: list[list[_Word]]
    font_size: int
    #: Box in frame pixels: (left, top, right, bottom).
    box: tuple[int, int, int, int]
    truncated: bool = False

    @property
    def text_lines(self) -> list[str]:
        return [" ".join(word.text for word in line) for line in self.lines]


def _normalise(word: str) -> str:
    return re.sub(r"[^\w%$]", "", word).lower()


def _measure(
    words: list[str], size: int, style: HookTitleStyle, highlights: set[str]
) -> tuple[list[_Word], float]:
    font = _font(str(style.font_path), size)
    emoji_h = round(size * style.emoji_scale)
    measured: list[_Word] = []
    for word in words:
        runs = split_runs(word)
        width = 0.0
        for run, is_emoji in runs:
            if is_emoji:
                bitmap = _emoji_bitmap(run)
                width += bitmap.width * emoji_h / bitmap.height
            else:
                width += font.getlength(run)
        measured.append(
            _Word(word, runs, width, highlight=_normalise(word) in highlights and bool(highlights))
        )
    return measured, font.getlength(" ")


def _line_width(words: list[_Word], space: float) -> float:
    return sum(w.width for w in words) + space * max(0, len(words) - 1)


def _wrap(words: list[_Word], space: float, max_width: float, max_lines: int):
    """Balanced wrap into at most ``max_lines`` lines, or None if it can't fit.

    Two lines are split where the longer of the two is shortest, which is what
    a designer does by hand — greedy wrapping leaves one word on its own line.
    """
    if not words:
        return []
    if _line_width(words, space) <= max_width:
        return [words]
    if max_lines < 2 or len(words) < 2:
        return None

    best = None
    best_width = float("inf")
    for split in range(1, len(words)):
        first, second = words[:split], words[split:]
        widest = max(_line_width(first, space), _line_width(second, space))
        if widest <= max_width and widest < best_width:
            best, best_width = [first, second], widest
    return best


def caption_top_px(style: CaptionStyle, height: int, *, lines: int = 2) -> int:
    """Highest pixel a caption group can reach, assuming ``lines`` wrapped lines.

    Mirrors how the ASS style is built: bottom-anchored at ``margin_v_ratio``,
    with the active word allowed to grow to ``scale_percent``.
    """
    font_px = height * style.size_ratio
    growth = style.scale_percent / 100 if style.animation == "scale" else 1.0
    return round(height * (1 - style.margin_v_ratio) - lines * font_px * 1.2 * growth)


def layout_title(
    text: str,
    *,
    width: int,
    height: int,
    style: HookTitleStyle,
    font_size: int = DEFAULT_FONT_SIZE,
    position_pct: float = DEFAULT_TOP_PCT,
    bottom_limit_px: int | None = None,
    highlight_words: list[str] | None = None,
    centre_in_band: bool = False,
) -> TitleLayout:
    """Fit a title into at most ``style.max_lines`` lines above ``bottom_limit_px``.

    Shrinks the font step by step; if even the smallest size can't hold it,
    trailing words are dropped and an ellipsis added (logged — the review page
    shows the rendered result so the user can shorten it).
    """
    text = " ".join(text.split())
    if not text:
        raise HookTitleError("The hook title is empty.")
    if style.all_caps:
        text = text.upper()

    top = round(height * max(MIN_TOP_PCT, position_pct) / 100)
    limit = bottom_limit_px if bottom_limit_px is not None else height
    limit -= round(height * CAPTION_GAP_RATIO)
    max_width = width * style.max_width_ratio
    highlights = {_normalise(w) for w in (highlight_words or []) if _normalise(w)}

    base = font_size * width / REFERENCE_WIDTH
    words = text.split(" ")
    truncated = False

    while True:
        size = base
        while size >= base * MIN_SHRINK:
            candidate = _try_layout(words, round(size), style, highlights, top, limit, max_width)
            if candidate is not None:
                if centre_in_band:
                    # Sit midway between the top line and the limit.
                    left, box_top, right, box_bottom = candidate.box
                    shift = max(0, (limit - box_bottom) // 2)
                    candidate.box = (left, box_top + shift, right, box_bottom + shift)
                candidate.truncated = truncated
                if truncated:
                    log.warning("Hook title %r did not fit and was shortened.", text)
                return candidate
            size *= 0.94
        if len(words) <= 1:
            raise HookTitleError(f"The hook title {text!r} cannot fit in the frame.")
        words = [*words[:-2], words[-2].rstrip(",.;:!?") + "…"]
        truncated = True


def _try_layout(
    words: list[str],
    size: int,
    style: HookTitleStyle,
    highlights: set[str],
    top: int,
    limit: int,
    max_width: float,
) -> TitleLayout | None:
    pad_x, pad_y = size * style.pad_x, size * style.pad_y
    measured, space = _measure(words, size, style, highlights)
    lines = _wrap(measured, space, max_width - 2 * pad_x, style.max_lines)
    if lines is None:
        return None

    block_h = size + size * style.line_spacing * (len(lines) - 1)
    box_h = block_h + 2 * pad_y
    if top + box_h > limit:
        return None
    box_w = max(_line_width(line, space) for line in lines) + 2 * pad_x
    return TitleLayout(lines=lines, font_size=size, box=(0, top, round(box_w), round(top + box_h)))


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _rgba(colour: str) -> tuple[int, int, int, int]:
    colour = colour.lstrip("#")
    return (int(colour[0:2], 16), int(colour[2:4], 16), int(colour[4:6], 16), 255)


def render_title(
    text: str,
    *,
    width: int,
    height: int,
    style: HookTitleStyle | str = "boxed",
    font_size: int = DEFAULT_FONT_SIZE,
    position_pct: float = DEFAULT_TOP_PCT,
    bottom_limit_px: int | None = None,
    highlight_words: list[str] | None = None,
    centre_in_band: bool = False,
) -> tuple[Image.Image, TitleLayout]:
    """Draw the title onto a transparent ``width``×``height`` RGBA image."""
    if isinstance(style, str):
        style = get_style(style)
    layout = layout_title(
        text,
        width=width,
        height=height,
        style=style,
        font_size=font_size,
        position_pct=position_pct,
        bottom_limit_px=bottom_limit_px,
        highlight_words=highlight_words,
        centre_in_band=centre_in_band,
    )

    size = layout.font_size
    _, top, box_w, bottom = layout.box
    left = (width - box_w) // 2
    layout.box = (left, top, left + box_w, bottom)

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    if style.box_colour:
        draw.rounded_rectangle(
            layout.box, radius=round(size * style.radius), fill=_rgba(style.box_colour)
        )

    font = _font(str(style.font_path), size)
    space = font.getlength(" ")
    emoji_h = round(size * style.emoji_scale)
    text_colour = _rgba(style.text_colour)
    highlight_colour = _rgba(style.highlight_colour) if style.highlight_colour else text_colour

    for index, line in enumerate(layout.lines):
        line_w = _line_width(line, space)
        x = (width - line_w) / 2
        # "lm" anchors on the vertical middle of the font's em box, which puts
        # cap-height text visually centred on the same line as the emoji.
        centre_y = top + size * style.pad_y + size / 2 + index * size * style.line_spacing
        for word in line:
            colour = highlight_colour if word.highlight else text_colour
            for run, is_emoji in word.runs:
                if is_emoji:
                    glyph = _emoji_image(run, emoji_h)
                    image.alpha_composite(glyph, (round(x), round(centre_y - glyph.height / 2)))
                    x += glyph.width
                else:
                    draw.text((x, centre_y), run, font=font, fill=colour, anchor="lm")
                    x += font.getlength(run)
            x += space

    return image, layout


def write_title_png(path: Path, text: str, **kwargs) -> TitleLayout:
    """Render the title and save it as a PNG. Returns the layout used."""
    image, layout = render_title(text, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)
    return layout
