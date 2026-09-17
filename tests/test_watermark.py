"""Watermark filtergraph construction and resolution — the pure-Python logic.

No ffmpeg binary needed here: these test the strings and arguments
``export.py`` builds, not what ffmpeg does with them. The real-render proof
that the overlay lands on the right pixels, and that the clip still ends on
time, lives in ``test_export_render.py`` (marked slow).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from autoclip import paths
from autoclip.config import ExportSettings, WatermarkSettings
from autoclip.pipeline import captions, export
from autoclip.pipeline.reframe.croppath import centre_crop


def _request(ratio: str = "9:16") -> export.ExportRequest:
    return export.ExportRequest(
        source=Path("source.mp4"),
        destination=Path("out.mp4"),
        start_s=0.0,
        end_s=5.0,
        crop_path=centre_crop(1920, 1080, 5.0),
        words=[],
        style=captions.get_style("bold_pop"),
        ratio=ratio,
    )


class TestPositionExpressions:
    """W/H are the base frame, w/h the scaled watermark — both resolved by
    ffmpeg itself from the real streams; only the margin is precomputed."""

    @pytest.mark.parametrize(
        ("position", "expected"),
        [
            ("top-left", "10:10"),
            ("top-right", "W-w-10:10"),
            ("bottom-left", "10:H-h-10"),
            ("bottom-right", "W-w-10:H-h-10"),
            ("center", "(W-w)/2:(H-h)/2"),
        ],
    )
    def test_each_preset(self, position: str, expected: str) -> None:
        assert export._watermark_position_expr(position, margin_px=10) == expected

    def test_unknown_position_is_a_clear_error(self) -> None:
        with pytest.raises(export.ExportError, match="diagonal"):
            export._watermark_position_expr("diagonal", margin_px=10)


class TestResolveWatermark:
    """The one place "enabled, but broken somehow" degrades to "off" rather
    than failing an export that would otherwise have succeeded."""

    def test_disabled_is_none_even_with_a_real_file(self, autoclip_home: Path) -> None:
        path = autoclip_home / "watermarks" / "watermark.png"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"fake-png")

        watermark = WatermarkSettings(enabled=False, filename="watermark.png")

        assert export._resolve_watermark(watermark) is None

    def test_enabled_with_no_filename_is_none(self) -> None:
        assert export._resolve_watermark(WatermarkSettings(enabled=True, filename="")) is None

    def test_enabled_but_the_file_was_deleted_externally_is_none(
        self, autoclip_home: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        watermark = WatermarkSettings(enabled=True, filename="gone.png")

        with caplog.at_level("WARNING"):
            resolved = export._resolve_watermark(watermark)

        assert resolved is None
        assert "gone.png" in caplog.text

    def test_enabled_with_the_file_present_resolves_to_it(self, autoclip_home: Path) -> None:
        path = autoclip_home / "watermarks" / "logo.png"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"fake-png")

        watermark = WatermarkSettings(enabled=True, filename="logo.png")

        assert export._resolve_watermark(watermark) == path


class TestFiltergraphWithoutAWatermark:
    """The regression contract: nothing about the existing graph changes for a
    clip that isn't watermarked, whether that's because the argument was
    omitted or explicitly passed as None."""

    def test_omitting_and_passing_none_produce_the_same_graph(self) -> None:
        request = _request()

        omitted = export.build_video_filtergraph(request, subtitle_name=None)
        explicit = export.build_video_filtergraph(request, subtitle_name=None, watermark=None)

        assert omitted == explicit

    def test_no_watermark_input_or_overlay_appears(self) -> None:
        graph = export.build_video_filtergraph(_request(), subtitle_name=None)

        assert "[1:v]" not in graph
        assert "overlay" not in graph
        assert graph.endswith("[vout]")
        assert graph.count("[vout]") == 1


class TestFiltergraphWithAWatermark:
    def test_adds_a_second_input_scale_opacity_and_overlay_stage(self) -> None:
        watermark = WatermarkSettings(
            enabled=True, filename="w.png", position="top-left", scale_pct=20.0, opacity_pct=50.0
        )

        graph = export.build_video_filtergraph(_request(), subtitle_name=None, watermark=watermark)

        # 9:16 output is 1080 wide: 20% -> 216px scale, 4% margin -> 43px.
        assert "[1:v]scale=216:-2" in graph
        assert "colorchannelmixer=aa=0.500" in graph
        assert "[vpre][wm]overlay=43:43:eof_action=repeat,format=yuv420p[vout]" in graph
        assert graph.count("[vout]") == 1

    def test_captions_still_burn_in_ahead_of_the_watermark(self) -> None:
        """The watermark composites onto [vpre], the captions stage's own
        output — so captions render *underneath* the logo, not the other way
        around, which would let a caption cover a logo instead."""
        watermark = WatermarkSettings(enabled=True, filename="w.png")

        graph = export.build_video_filtergraph(
            _request(), subtitle_name="captions.ass", fonts_name="fonts", watermark=watermark
        )

        assert "ass=filename=captions.ass:fontsdir=fonts[vpre]" in graph
        assert "[vpre][wm]overlay=" in graph

    def test_scale_percentage_is_relative_to_output_width_not_the_source(self) -> None:
        watermark = WatermarkSettings(enabled=True, filename="w.png", scale_pct=10.0)

        def graph_for(ratio: str) -> str:
            return export.build_video_filtergraph(
                _request(ratio), subtitle_name=None, watermark=watermark
            )

        # 9:16 and 1:1 are both 1080 wide; 16:9 is 1920 wide.
        assert "scale=108:-2" in graph_for("9:16")
        assert "scale=108:-2" in graph_for("1:1")
        assert "scale=192:-2" in graph_for("16:9")


class TestFfmpegArguments:
    """What export_clip hands ffmpeg, captured without running it."""

    @pytest.fixture
    def captured(self, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
        calls: list[list[str]] = []

        def fake_run(args, **kwargs) -> None:
            calls.append(list(args))
            # export_clip checks that the output exists and isn't empty.
            Path(args[-1]).write_bytes(b"rendered")

        monkeypatch.setattr(export.ffmpeg, "run", fake_run)
        return calls

    def _export(self, tmp_path: Path, watermark: WatermarkSettings) -> None:
        request = _request()
        request.destination = tmp_path / "out.mp4"
        export.export_clip(
            request,
            work_dir=tmp_path / "work",
            # Software encoding, so the test never probes the machine for NVENC.
            settings=ExportSettings(prefer_hardware_encoder=False, watermark=watermark),
        )

    @staticmethod
    def _inputs(args: list[str]) -> list[str]:
        return [args[i + 1] for i, arg in enumerate(args) if arg == "-i"]

    def test_without_a_watermark_there_is_only_the_source_input(
        self, tmp_path: Path, captured: list[list[str]]
    ) -> None:
        self._export(tmp_path, WatermarkSettings())

        assert self._inputs(captured[0]) == [str(Path("source.mp4"))]

    def test_the_image_is_a_plain_single_frame_input_not_a_loop(
        self, tmp_path: Path, captured: list[list[str]]
    ) -> None:
        # Looping the image (-loop 1) never lets its stream end, and overlay
        # then stretches the clip to match: the render ran until it was killed.
        image = paths.ensure_watermarks_dir() / "watermark.png"
        image.write_bytes(b"png")

        self._export(tmp_path, WatermarkSettings(enabled=True, filename="watermark.png"))

        args = captured[0]
        assert "-loop" not in args
        assert self._inputs(args) == [str(Path("source.mp4")), str(image)]
