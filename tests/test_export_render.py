"""End-to-end render tests against real ffmpeg.

These are the tests that catch what unit tests structurally cannot: a
filtergraph that parses but produces the wrong thing, a font libass can't
resolve, a path escaping bug that only appears on Windows, an encoder flag the
local build rejects. Marked ``slow`` — they shell out and encode video.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from autoclip import paths
from autoclip.config import ExportSettings, WatermarkSettings
from autoclip.pipeline import captions, export, ffmpeg
from autoclip.pipeline.reframe.croppath import (
    CropKeyframe,
    CropPath,
    CropSegment,
    Strategy,
    centre_crop,
)
from autoclip.pipeline.transcript import Word

pytestmark = pytest.mark.slow

SOURCE_W, SOURCE_H = 1280, 720
SOURCE_DURATION = 10.0


@pytest.fixture(scope="module")
def source_video(tmp_path_factory) -> Path:
    """A synthetic 1280x720 test clip with a tone, generated once per session."""
    path = tmp_path_factory.mktemp("media") / "source.mp4"
    subprocess.run(
        [
            ffmpeg.ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={SOURCE_W}x{SOURCE_H}:rate=30:duration={SOURCE_DURATION}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={SOURCE_DURATION}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture
def words() -> list[Word]:
    """Words spanning 2.0-7.0s of the source, matching the clip under test."""
    texts = [
        "this",
        "is",
        "a",
        "test",
        "of",
        "the",
        "caption",
        "rendering",
        "pipeline",
        "right",
        "now",
    ]
    step = 5.0 / len(texts)
    return [
        Word(text=text, start=2.0 + i * step, end=2.0 + i * step + step * 0.85)
        for i, text in enumerate(texts)
    ]


@pytest.fixture(scope="module")
def watermark_source(tmp_path_factory) -> Path:
    """A small solid-colour PNG, generated once per session.

    Built with the same ffmpeg-lavfi idiom as `source_video` above, rather
    than pulling in Pillow or numpy just to make a test fixture — there's
    otherwise no reason for this suite to need either.
    """
    path = tmp_path_factory.mktemp("watermark") / "logo.png"
    subprocess.run(
        [
            ffmpeg.ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=200x100",
            "-frames:v",
            "1",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _watermarked_settings(watermark_source: Path, **overrides) -> ExportSettings:
    """Copy the fixture image into this test's AUTOCLIP_HOME and enable it.

    A real copy into `paths.watermarks_dir()`, not a mock: `export_clip` reads
    the watermark straight off disk, so this exercises the same path a real
    upload would leave behind.
    """
    directory = paths.ensure_watermarks_dir()
    destination = directory / "watermark.png"
    shutil.copyfile(watermark_source, destination)
    return ExportSettings(
        watermark=WatermarkSettings(enabled=True, filename=destination.name, **overrides)
    )


def make_request(source: Path, destination: Path, crop_path: CropPath, words, **kwargs):
    return export.ExportRequest(
        source=source,
        destination=destination,
        start_s=2.0,
        end_s=7.0,
        crop_path=crop_path,
        words=words,
        style=kwargs.pop("style", captions.get_style("bold_pop")),
        **kwargs,
    )


class TestSingleSegmentRender:
    def test_renders_a_vertical_clip(self, source_video, words, tmp_path) -> None:
        destination = tmp_path / "out.mp4"
        request = make_request(
            source_video, destination, centre_crop(SOURCE_W, SOURCE_H, 5.0), words
        )

        export.export_clip(request, work_dir=tmp_path / "work")

        assert destination.exists()
        info = ffmpeg.probe(destination)
        assert (info.width, info.height) == (1080, 1920)
        assert info.has_audio
        assert info.duration_s == pytest.approx(5.0, abs=0.35)

    @pytest.mark.parametrize("ratio", ["9:16", "1:1", "16:9"])
    def test_every_ratio_renders(self, source_video, words, tmp_path, ratio: str) -> None:
        destination = tmp_path / f"out_{ratio.replace(':', 'x')}.mp4"
        aspect = tuple(int(part) for part in ratio.split(":"))
        crop_path = centre_crop(SOURCE_W, SOURCE_H, 5.0, aspect_w=aspect[0], aspect_h=aspect[1])

        export.export_clip(
            make_request(source_video, destination, crop_path, words, ratio=ratio),
            work_dir=tmp_path / "work",
        )

        info = ffmpeg.probe(destination)
        assert (info.width, info.height) == export.ratio_dimensions(ratio)

    @pytest.mark.parametrize("style_key", list(captions.PRESETS))
    def test_every_caption_style_burns_in(
        self, source_video, words, tmp_path, style_key: str
    ) -> None:
        # A style whose font libass can't resolve still "succeeds" but renders
        # in a substituted face, so this checks the render completes for each.
        destination = tmp_path / f"out_{style_key}.mp4"
        request = make_request(
            source_video,
            destination,
            centre_crop(SOURCE_W, SOURCE_H, 5.0),
            words,
            style=captions.get_style(style_key),
        )

        export.export_clip(request, work_dir=tmp_path / "work")

        assert destination.stat().st_size > 1000

    def test_captions_visibly_change_the_output(self, source_video, words, tmp_path) -> None:
        """Burned captions must actually alter the pixels.

        Without this, a silently-failing `ass` filter would leave every other
        assertion passing while shipping clips with no captions on them.
        """
        with_captions = tmp_path / "with.mp4"
        without_captions = tmp_path / "without.mp4"

        export.export_clip(
            make_request(source_video, with_captions, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=tmp_path / "work",
        )
        export.export_clip(
            make_request(
                source_video,
                without_captions,
                centre_crop(SOURCE_W, SOURCE_H, 5.0),
                words,
                burn_captions=False,
            ),
            work_dir=tmp_path / "work",
        )

        assert _frame_signature(with_captions, 2.5) != _frame_signature(without_captions, 2.5)


class TestMultiSegmentRender:
    def test_concatenated_segments_render(self, source_video, words, tmp_path) -> None:
        crop_w, crop_h = 404, 720
        crop_path = CropPath(
            source_width=SOURCE_W,
            source_height=SOURCE_H,
            segments=[
                CropSegment(
                    start_s=0.0,
                    end_s=2.5,
                    width=crop_w,
                    height=crop_h,
                    keyframes=[CropKeyframe(0.0, 100.0, 0.0)],
                    strategy=Strategy.TRACK,
                ),
                CropSegment(
                    start_s=2.5,
                    end_s=5.0,
                    width=crop_w,
                    height=crop_h,
                    keyframes=[CropKeyframe(2.5, 700.0, 0.0)],
                    strategy=Strategy.TRACK,
                ),
            ],
        )
        destination = tmp_path / "multi.mp4"

        export.export_clip(
            make_request(source_video, destination, crop_path, words),
            work_dir=tmp_path / "work",
        )

        info = ffmpeg.probe(destination)
        assert (info.width, info.height) == (1080, 1920)
        assert info.duration_s == pytest.approx(5.0, abs=0.35)

    def test_segments_actually_show_different_regions(self, source_video, words, tmp_path) -> None:
        """A crop that doesn't move would make the two segments identical."""
        crop_w, crop_h = 404, 720
        crop_path = CropPath(
            source_width=SOURCE_W,
            source_height=SOURCE_H,
            segments=[
                CropSegment(
                    start_s=0.0,
                    end_s=2.5,
                    width=crop_w,
                    height=crop_h,
                    keyframes=[CropKeyframe(0.0, 0.0, 0.0)],
                ),
                CropSegment(
                    start_s=2.5,
                    end_s=5.0,
                    width=crop_w,
                    height=crop_h,
                    keyframes=[CropKeyframe(2.5, 876.0, 0.0)],
                ),
            ],
        )
        destination = tmp_path / "regions.mp4"

        export.export_clip(
            make_request(source_video, destination, crop_path, words, burn_captions=False),
            work_dir=tmp_path / "work",
        )

        assert _frame_signature(destination, 1.0) != _frame_signature(destination, 4.0)

    def test_animated_crop_expression_renders(self, source_video, words, tmp_path) -> None:
        crop_path = CropPath(
            source_width=SOURCE_W,
            source_height=SOURCE_H,
            segments=[
                CropSegment(
                    start_s=0.0,
                    end_s=5.0,
                    width=404,
                    height=720,
                    keyframes=[
                        CropKeyframe(0.0, 0.0, 0.0),
                        CropKeyframe(2.5, 400.0, 0.0),
                        CropKeyframe(5.0, 876.0, 0.0),
                    ],
                    strategy=Strategy.TRACK,
                )
            ],
        )
        destination = tmp_path / "animated.mp4"

        export.export_clip(
            make_request(source_video, destination, crop_path, words, burn_captions=False),
            work_dir=tmp_path / "work",
        )

        # A broken expression would evaluate to a constant, leaving these equal.
        assert _frame_signature(destination, 0.5) != _frame_signature(destination, 4.5)

    def test_fit_segment_renders_with_blurred_background(
        self, source_video, words, tmp_path
    ) -> None:
        crop_path = CropPath(
            source_width=SOURCE_W,
            source_height=SOURCE_H,
            segments=[
                CropSegment(
                    start_s=0.0,
                    end_s=5.0,
                    width=SOURCE_W,
                    height=SOURCE_H,
                    keyframes=[CropKeyframe(0.0, 0.0, 0.0)],
                    strategy=Strategy.WIDE,
                    fit=True,
                )
            ],
        )
        destination = tmp_path / "fit.mp4"

        export.export_clip(
            make_request(source_video, destination, crop_path, words),
            work_dir=tmp_path / "work",
        )

        info = ffmpeg.probe(destination)
        assert (info.width, info.height) == (1080, 1920)


class TestEncoding:
    def test_software_encoder_path(self, source_video, words, tmp_path) -> None:
        settings = ExportSettings(prefer_hardware_encoder=False)
        destination = tmp_path / "x264.mp4"

        export.export_clip(
            make_request(source_video, destination, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=tmp_path / "work",
            settings=settings,
        )

        assert ffmpeg.probe(destination).video_codec == "h264"

    def test_output_is_yuv420p(self, source_video, words, tmp_path) -> None:
        # Anything else fails to play on a surprising number of phones.
        destination = tmp_path / "pixfmt.mp4"
        export.export_clip(
            make_request(source_video, destination, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=tmp_path / "work",
        )

        result = subprocess.run(
            [
                ffmpeg.ffprobe_path(),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=pix_fmt",
                "-of",
                "csv=p=0",
                str(destination),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        assert result.stdout.strip() == "yuv420p"

    def test_srt_sidecar_is_written_when_requested(self, source_video, words, tmp_path) -> None:
        settings = ExportSettings(write_srt=True)
        destination = tmp_path / "sidecar.mp4"

        export.export_clip(
            make_request(source_video, destination, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=tmp_path / "work",
            settings=settings,
        )

        assert destination.with_suffix(".srt").exists()

    def test_progress_reaches_completion(self, source_video, words, tmp_path) -> None:
        seen: list[float] = []

        export.export_clip(
            make_request(
                source_video, tmp_path / "progress.mp4", centre_crop(SOURCE_W, SOURCE_H, 5.0), words
            ),
            work_dir=tmp_path / "work",
            on_progress=seen.append,
        )

        assert seen
        assert seen[-1] == 1.0
        assert all(0.0 <= value <= 1.0 for value in seen)


class TestPathsWithSpecialCharacters:
    def test_directory_with_spaces_and_punctuation(self, source_video, words, tmp_path) -> None:
        """The escaping test that actually matters on Windows.

        A drive letter plus a space plus an apostrophe is the combination that
        breaks naive filtergraph quoting.
        """
        awkward = tmp_path / "My Clips (2026)" / "it's here"
        awkward.mkdir(parents=True)
        destination = awkward / "out.mp4"

        export.export_clip(
            make_request(source_video, destination, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=awkward / "work",
        )

        assert destination.exists()
        assert ffmpeg.probe(destination).width == 1080


class TestWatermark:
    def test_visibly_changes_the_output(
        self, source_video, words, tmp_path, watermark_source
    ) -> None:
        plain = tmp_path / "plain.mp4"
        marked = tmp_path / "marked.mp4"

        export.export_clip(
            make_request(source_video, plain, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=tmp_path / "work-plain",
        )
        export.export_clip(
            make_request(source_video, marked, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=tmp_path / "work-marked",
            settings=_watermarked_settings(watermark_source, scale_pct=25.0, opacity_pct=100.0),
        )

        # Regression: looping the image input (-loop 1) once made the
        # watermarked render run forever, stretching the clip's last frame.
        assert ffmpeg.probe(marked).duration_s == pytest.approx(5.0, abs=0.35)
        assert _frame_signature(plain, 3.0) != _frame_signature(marked, 3.0)

    def test_disabled_renders_exactly_like_no_watermark_at_all(
        self, source_video, words, tmp_path, watermark_source
    ) -> None:
        """A file sitting on disk but the toggle off must be indistinguishable
        from never having uploaded one — the "no behaviour change" contract."""
        # Copies the fixture image into place and enables it, then flips it
        # back off — so the file genuinely exists on disk for this assertion,
        # not just an absent one that _resolve_watermark would skip anyway.
        enabled = _watermarked_settings(watermark_source)
        settings_off = ExportSettings(
            watermark=enabled.watermark.model_copy(update={"enabled": False})
        )

        without = tmp_path / "without.mp4"
        off = tmp_path / "off.mp4"
        export.export_clip(
            make_request(source_video, without, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=tmp_path / "work-without",
        )
        export.export_clip(
            make_request(source_video, off, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=tmp_path / "work-off",
            settings=settings_off,
        )

        assert _frame_signature(without, 3.0) == _frame_signature(off, 3.0)

    def test_a_missing_file_degrades_to_no_watermark_instead_of_failing(
        self, source_video, words, tmp_path
    ) -> None:
        settings = ExportSettings(watermark=WatermarkSettings(enabled=True, filename="ghost.png"))

        without = tmp_path / "without.mp4"
        ghost = tmp_path / "ghost.mp4"
        export.export_clip(
            make_request(source_video, without, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=tmp_path / "work-without",
        )
        export.export_clip(
            make_request(source_video, ghost, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=tmp_path / "work-ghost",
            settings=settings,
        )

        assert _frame_signature(without, 3.0) == _frame_signature(ghost, 3.0)

    def test_position_places_it_in_the_requested_corner(
        self, source_video, words, tmp_path, watermark_source
    ) -> None:
        top_left = tmp_path / "top_left.mp4"
        bottom_right = tmp_path / "bottom_right.mp4"

        export.export_clip(
            make_request(source_video, top_left, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=tmp_path / "work-tl",
            settings=_watermarked_settings(
                watermark_source, position="top-left", scale_pct=20.0, opacity_pct=100.0
            ),
        )
        export.export_clip(
            make_request(source_video, bottom_right, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=tmp_path / "work-br",
            settings=_watermarked_settings(
                watermark_source, position="bottom-right", scale_pct=20.0, opacity_pct=100.0
            ),
        )

        # Same crop window (the top-left 200x200 square) on both renders: it
        # holds the logo in one and untouched background in the other.
        top_left_corner = "200:200:0:0"
        assert _region_signature(top_left, 3.0, top_left_corner) != _region_signature(
            bottom_right, 3.0, top_left_corner
        )

    def test_opacity_changes_the_blend(
        self, source_video, words, tmp_path, watermark_source
    ) -> None:
        faint = tmp_path / "faint.mp4"
        solid = tmp_path / "solid.mp4"

        export.export_clip(
            make_request(source_video, faint, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=tmp_path / "work-faint",
            settings=_watermarked_settings(watermark_source, opacity_pct=15.0),
        )
        export.export_clip(
            make_request(source_video, solid, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=tmp_path / "work-solid",
            settings=_watermarked_settings(watermark_source, opacity_pct=100.0),
        )

        assert _frame_signature(faint, 3.0) != _frame_signature(solid, 3.0)


def _frame_signature(video: Path, timestamp: float) -> str:
    """Hash one frame's pixels, for comparing rendered output."""
    return _region_signature(video, timestamp, crop=None)


def _region_signature(video: Path, timestamp: float, crop: str | None) -> str:
    """Hash one frame's pixels, optionally within an ffmpeg ``crop=w:h:x:y``
    region — for comparing one corner of a rendered frame against another."""
    args = [
        ffmpeg.ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(timestamp),
        "-i",
        str(video),
    ]
    if crop is not None:
        args += ["-vf", f"crop={crop}"]
    args += ["-frames:v", "1", "-f", "hash", "-hash", "md5", "-"]

    result = subprocess.run(args, capture_output=True, text=True, check=True)
    return result.stdout.strip()
