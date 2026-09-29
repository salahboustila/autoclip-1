"""The Podcast Hook layout: geometry, keyword picking, filtergraph, runner, render."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from autoclip import config
from autoclip.config import HookTitleSettings
from autoclip.db import store
from autoclip.db.models import Clip, Job, Source, new_id
from autoclip.pipeline import captions, export, ffmpeg, hooktitle, layouts
from autoclip.pipeline import runner as runner_module
from autoclip.pipeline.runner import PipelineRunner
from autoclip.pipeline.transcript import Transcript, Word
from PIL import Image

BOLD_POP = captions.get_style("bold_pop")
SPOKEN = "the average man makes fifty thousand dollars a year"


def viral_settings(**viral) -> config.Settings:
    settings = config.Settings()
    settings.viral_hook.enabled = True
    for key, value in viral.items():
        setattr(settings.viral_hook, key, value)
    return settings


class TestGeometry:
    def test_16x9_source_fills_the_width_centred(self) -> None:
        layout = layouts.podcast_hook_layout(1920, 1080)
        assert layout.video_box == (0, 656, 1080, 608)

    def test_4x3_source(self) -> None:
        x, y, w, h = layouts.podcast_hook_layout(1440, 1080).video_box
        assert (x, w, h) == (0, 1080, 810)
        assert y + h / 2 == pytest.approx(960, abs=2)

    def test_vertical_source_is_capped_at_square(self) -> None:
        assert layouts.podcast_hook_layout(1080, 1920).video_box == (0, 420, 1080, 1080)

    def test_tracked_is_square(self) -> None:
        layout = layouts.podcast_hook_layout(1920, 1080, fit="tracked")
        assert layout.video_box == (0, 420, 1080, 1080)

    def test_unknown_fit_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            layouts.podcast_hook_layout(1920, 1080, fit="zoomed")

    @pytest.mark.parametrize("fit", ["full_width", "tracked"])
    def test_bands_never_touch_the_video(self, fit: str) -> None:
        layout = layouts.podcast_hook_layout(1920, 1080, fit=fit)
        band_top, band_bottom = layout.caption_band
        assert layout.title_limit_px < layout.video_top
        assert band_top > layout.video_bottom
        assert band_bottom < layout.height


class TestCaptionStyle:
    @pytest.mark.parametrize("fit", ["full_width", "tracked"])
    @pytest.mark.parametrize("key", list(captions.PRESETS))
    def test_one_line_sits_inside_the_caption_band(self, fit: str, key: str) -> None:
        layout = layouts.podcast_hook_layout(1920, 1080, fit=fit)
        base = captions.get_style(key)
        style = layouts.caption_style_for(layout, base)
        band_top, band_bottom = layout.caption_band
        bottom = layout.height * (1 - style.margin_v_ratio)
        growth = base.scale_percent / 100 if base.animation == "scale" else 1
        top = bottom - layout.height * base.size_ratio * 1.2 * growth
        assert bottom <= band_bottom + 1
        if fit == "full_width":
            assert top >= band_top - 1

    def test_keyword_colour_and_short_lines(self) -> None:
        layout = layouts.podcast_hook_layout(1920, 1080)
        green = layouts.caption_style_for(layout, BOLD_POP)
        yellow = layouts.caption_style_for(layout, BOLD_POP, keyword="yellow")
        assert green.keyword_colour == "#31E981"
        assert yellow.keyword_colour == "#FFE500"
        assert green.max_words == 3
        assert green.accent == "#FFFFFF"  # the spoken word pops without recolouring

    def test_presets_are_untouched(self) -> None:
        assert all(style.keyword_colour is None for style in captions.PRESETS.values())


class TestKeywords:
    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("men earn fifty", "fifty"),
            ("about 69% of women", "69%"),
            ("the relationship was over", "relationship"),
        ],
    )
    def test_caption_keyword(self, line: str, expected: str) -> None:
        words = [Word(w, i, i + 0.5) for i, w in enumerate(line.split())]
        assert words[captions.pick_keyword(words)].text == expected

    def test_filler_only_line_has_no_keyword(self) -> None:
        words = [Word(w, i, i + 0.5) for i, w in enumerate(["yeah", "and", "the"])]
        assert captions.pick_keyword(words) is None

    def test_keyword_is_coloured_in_every_word_event(self) -> None:
        layout = layouts.podcast_hook_layout(1920, 1080)
        style = layouts.caption_style_for(layout, BOLD_POP)
        words = [Word("men", 0, 0.3), Word("earn", 0.3, 0.6), Word("fifty", 0.6, 0.9)]
        events = captions.build_ass(words, style, width=1080, height=1920).events
        assert len(events) == 3
        assert all("{\\c&H81E931&" in e.text and "FIFTY" in e.text for e in events)

    def test_default_captions_are_unchanged(self) -> None:
        words = [Word("men", 0, 0.3), Word("earn", 0.3, 0.6), Word("fifty", 0.6, 0.9)]
        events = captions.build_ass(words, BOLD_POP, width=1080, height=1920).events
        assert "81E931" not in "".join(e.text for e in events)

    @pytest.mark.parametrize(
        ("title", "expected"),
        [
            ("69% Voted Not Guilty For This?! 😱", ["69%"]),
            ("How Much Does the Average Man Make? 🤔", ["Average"]),
            ("If You Want It, Learn This", []),
        ],
    )
    def test_title_keyword(self, title: str, expected: list[str]) -> None:
        assert layouts.title_keyword(title) == expected


class TestTitle:
    def test_podcast_title_is_white_with_a_yellow_keyword_and_no_box(self) -> None:
        layout = layouts.podcast_hook_layout(1920, 1080)
        image, placed = hooktitle.render_title(
            "How Much Does the Average Man Make? 🤔",
            width=1080,
            height=1920,
            style="podcast_anton",
            font_size=90,
            position_pct=12,
            bottom_limit_px=layout.title_limit_px,
            centre_in_band=True,
            highlight_words=["Average"],
        )
        assert placed.text_lines[0].isupper()
        assert placed.box[1] >= round(1920 * 0.12)
        assert placed.box[3] <= layout.title_limit_px
        pixels = list(image.crop(placed.box).getdata())
        assert any(p[:3] == (255, 229, 0) and p[3] == 255 for p in pixels)  # yellow
        assert any(p[:3] == (255, 255, 255) and p[3] == 255 for p in pixels)  # white
        assert image.getpixel((placed.box[0] + 2, placed.box[1] + 2))[3] == 0  # no box

    def test_centred_in_the_band(self) -> None:
        _, placed = hooktitle.render_title(
            "Short Title",
            width=1080,
            height=1920,
            style="podcast_anton",
            position_pct=12,
            bottom_limit_px=620,
            centre_in_band=True,
        )
        top, bottom = placed.box[1], placed.box[3]
        above = top - round(1920 * 0.12)
        below = (620 - round(1920 * hooktitle.CAPTION_GAP_RATIO)) - bottom
        assert abs(above - below) <= 2


class TestFiltergraph:
    def _request(self, **kwargs) -> export.ExportRequest:
        return export.ExportRequest(
            source=Path("in.mp4"),
            destination=Path("out.mp4"),
            start_s=0.0,
            end_s=5.0,
            crop_path=layouts.podcast_crop_path(_options(), 1920, 1080, 5.0),
            words=[],
            style=BOLD_POP,
            **kwargs,
        )

    def test_scales_into_the_box_and_pads_onto_black(self) -> None:
        request = self._request(podcast=_options(), ratio="1:1")
        graph = export.build_video_filtergraph(request, subtitle_name="captions.ass")
        assert request.ratio == "9:16"
        assert "scale=1080:608" in graph
        assert "pad=1080:1920:0:656:color=black" in graph
        assert graph.index("pad=") < graph.index("ass=")

    def test_full_width_crop_is_the_whole_16x9_frame(self) -> None:
        segment = layouts.podcast_crop_path(_options(), 1920, 1080, 5.0).segments[0]
        # 1080x608 is 16:9 to within even-pixel rounding: 1 px trimmed per side.
        assert segment.height == 1080
        assert 1916 <= segment.width <= 1920

    def test_standard_layout_has_no_pad(self) -> None:
        graph = export.build_video_filtergraph(
            export.ExportRequest(
                source=Path("in.mp4"),
                destination=Path("out.mp4"),
                start_s=0.0,
                end_s=5.0,
                crop_path=layouts.podcast_crop_path(_options(), 1920, 1080, 5.0),
                words=[],
                style=BOLD_POP,
            ),
            subtitle_name="captions.ass",
        )
        assert "pad=" not in graph


class TestOptions:
    def test_off_unless_viral_mode_and_podcast_layout(self) -> None:
        assert layouts.podcast_options(config.Settings(), 1920, 1080) is None
        assert layouts.podcast_options(viral_settings(layout="standard"), 1920, 1080) is None
        options = layouts.podcast_options(
            viral_settings(title_font="montserrat", caption_keyword_colour="yellow"), 1920, 1080
        )
        assert options.title_style == "podcast_montserrat"
        assert options.caption_keyword_colour == "yellow"

    def test_full_width_skips_the_reframe_stage(
        self, initialised_db, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_tracking(*_a, **_k):
            raise AssertionError("face tracking should not run for full-width")

        monkeypatch.setattr(runner_module, "build_crop_path", no_tracking)
        source = store.create_source(
            Source(id=new_id(), type="upload", path="x.mp4", width=1920, height=1080)
        )
        job = store.create_job(Job(id=new_id(), source_id=source.id))
        clip = Clip(id=new_id(), job_id=job.id, start_s=0, end_s=30)
        runner = PipelineRunner(job, source, settings=viral_settings())

        assert runner._stage_reframe([clip], Transcript(words=[])) == {}

    def test_tracked_reframes_to_square(
        self, initialised_db, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = {}

        def capture(*_a, config, **_k):
            seen["aspect"] = (config.aspect_w, config.aspect_h)
            return layouts.podcast_crop_path(_options(), 1920, 1080, 30.0)

        monkeypatch.setattr(runner_module, "build_crop_path", capture)
        source = store.create_source(
            Source(id=new_id(), type="upload", path="x.mp4", width=1920, height=1080)
        )
        job = store.create_job(Job(id=new_id(), source_id=source.id))
        clip = Clip(id=new_id(), job_id=job.id, start_s=0, end_s=30)
        runner = PipelineRunner(job, source, settings=viral_settings(video_fit="tracked"))

        runner._stage_reframe([clip], Transcript(words=[]))

        assert seen["aspect"] == (1, 1)


def _options(fit: str = "full_width") -> layouts.PodcastOptions:
    return layouts.podcast_options(viral_settings(video_fit=fit), 1920, 1080)


@pytest.mark.slow
class TestRender:
    @pytest.fixture(scope="class")
    def source(self, tmp_path_factory) -> Path:
        path = tmp_path_factory.mktemp("media") / "pod.mp4"
        subprocess.run(
            [ffmpeg.ffmpeg_path(), "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             "testsrc2=size=1920x1080:rate=30:duration=6", "-f", "lavfi", "-i",
             "sine=frequency=440:duration=6", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-shortest", str(path)],
            check=True, capture_output=True,
        )  # fmt: skip
        return path

    @pytest.mark.parametrize("fit", ["full_width", "tracked"])
    def test_renders_the_layout(self, source: Path, tmp_path: Path, fit: str) -> None:
        options = _options(fit)
        words = [
            Word(text=w, start=1 + i * 0.4, end=1 + i * 0.4 + 0.38)
            for i, w in enumerate(SPOKEN.split())
        ]
        destination = tmp_path / "out.mp4"
        request = export.ExportRequest(
            source=source,
            destination=destination,
            start_s=1.0,
            end_s=5.0,
            crop_path=layouts.podcast_crop_path(options, 1920, 1080, 4.0),
            words=words,
            style=BOLD_POP,
            hook_title="How Much Does the Average Man Make? 💰",
            hook_title_settings=HookTitleSettings(),
            podcast=options,
        )

        export.export_clip(request, work_dir=tmp_path / "work")

        info = ffmpeg.probe(destination)
        assert (info.width, info.height) == (1080, 1920)
        frame = tmp_path / "frame.png"
        subprocess.run(
            [ffmpeg.ffmpeg_path(), "-loglevel", "error", "-y", "-ss", "0.9", "-i",
             str(destination), "-frames:v", "1", str(frame)],
            check=True, capture_output=True,
        )  # fmt: skip
        layout = options.layout
        with Image.open(frame) as image:
            rgb = image.convert("RGB")

            def band(top: int, bottom: int) -> list[tuple[int, int, int]]:
                return [
                    rgb.getpixel((x, y)) for y in range(top, bottom, 6) for x in range(0, 1080, 6)
                ]

            def bright(pixels) -> int:
                return sum(1 for p in pixels if max(p) > 200)

            # Black canvas: the very top and bottom strips are empty.
            assert bright(band(0, 150)) == 0
            assert bright(band(1880, 1920)) == 0
            # The title in its band, the video in its box, captions in theirs.
            assert bright(band(round(1920 * 0.12), layout.video_top)) > 50
            assert bright(band(layout.video_top + 10, layout.video_bottom - 10)) > 1000
            assert bright(band(*layout.caption_band)) > 50
            # Yellow title keyword somewhere in the title band.
            title = band(round(1920 * 0.12), layout.video_top)
            assert any(r > 220 and g > 190 and b < 80 for r, g, b in title)
