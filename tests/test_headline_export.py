"""Headline resolution and filtergraph gating — the pure-Python logic.

Mirrors test_watermark.py's split: no ffmpeg binary needed here, just the
strings and objects export.py builds. The real-render proof that the
overlay lands on the right pixels lives in test_export_render.py (marked
slow).
"""

from __future__ import annotations

from pathlib import Path

from autoclip.config import HeadlineSettings
from autoclip.pipeline import captions, export
from autoclip.pipeline.reframe.croppath import centre_crop


def _request(**overrides) -> export.ExportRequest:
    defaults = {
        "source": Path("source.mp4"),
        "destination": Path("out.mp4"),
        "start_s": 0.0,
        "end_s": 5.0,
        "crop_path": centre_crop(1920, 1080, 5.0),
        "words": [],
        "style": captions.get_style("bold_pop"),
    }
    return export.ExportRequest(**{**defaults, **overrides})


class TestResolveHeadline:
    """The one place "enabled, but nothing to show" degrades to "off" rather
    than failing an export that would otherwise succeed — same contract as
    _resolve_watermark."""

    def test_disabled_globally_is_none_even_with_text(self) -> None:
        settings = HeadlineSettings(enabled=False)

        assert export._resolve_headline(settings, "A real headline") is None

    def test_enabled_with_empty_text_is_none(self) -> None:
        settings = HeadlineSettings(enabled=True)

        assert export._resolve_headline(settings, "") is None

    def test_enabled_with_only_whitespace_text_is_none(self) -> None:
        settings = HeadlineSettings(enabled=True)

        assert export._resolve_headline(settings, "   ") is None

    def test_enabled_with_text_resolves_to_a_style(self) -> None:
        settings = HeadlineSettings(
            enabled=True, text_color="#00FF00", position="upper-center", max_lines=2
        )

        style = export._resolve_headline(settings, "A real headline")

        assert style is not None
        assert style.text == "A real headline"
        assert style.text_color == "#00FF00"
        assert style.position == "upper-center"
        assert style.max_lines == 2


class TestFiltergraphWithHeadlineButCaptionsOff:
    """Regression test for the bug caught while wiring this feature in:
    build_video_filtergraph's ass= gate used to also require
    request.burn_captions, which meant a headline-only clip with word
    captions off silently lost the headline too."""

    def test_ass_filter_still_applies_when_burn_captions_is_false(self) -> None:
        request = _request(burn_captions=False)

        graph = export.build_video_filtergraph(request, subtitle_name="captions.ass")

        assert "ass=filename=captions.ass" in graph

    def test_null_passthrough_only_when_there_is_truly_no_subtitle_file(self) -> None:
        request = _request(burn_captions=False)

        graph = export.build_video_filtergraph(request, subtitle_name=None)

        assert "null[vout]" in graph
        assert "ass=" not in graph


class TestExportClipResolvesTheHeadline:
    """export_clip is the one place that decides whether an .ass file needs
    writing at all — for word captions, the headline, or both."""

    def test_writes_ass_for_a_headline_even_with_no_words_and_captions_off(
        self, tmp_path, monkeypatch
    ) -> None:
        written: list[dict] = []
        real_write_ass = captions.write_ass

        def spy_write_ass(*args, **kwargs):
            written.append(kwargs)
            return real_write_ass(*args, **kwargs)

        monkeypatch.setattr(export.captions_module, "write_ass", spy_write_ass)

        def fake_run(args, **kwargs) -> None:
            Path(args[-1]).write_bytes(b"rendered")

        monkeypatch.setattr(export.ffmpeg, "run", fake_run)

        from autoclip.config import ExportSettings

        request = _request(
            destination=tmp_path / "out.mp4",
            burn_captions=False,
            headline_text="A generated headline",
        )
        export.export_clip(
            request,
            work_dir=tmp_path / "work",
            settings=ExportSettings(prefer_hardware_encoder=False),
        )

        assert len(written) == 1
        assert written[0]["headline"] is not None
        assert written[0]["headline"].text == "A generated headline"

    def test_no_ass_file_written_when_neither_captions_nor_headline_are_wanted(
        self, tmp_path, monkeypatch
    ) -> None:
        written: list[dict] = []
        monkeypatch.setattr(
            export.captions_module,
            "write_ass",
            lambda *a, **k: written.append(k) or Path(a[0]),
        )

        def fake_run(args, **kwargs) -> None:
            Path(args[-1]).write_bytes(b"rendered")

        monkeypatch.setattr(export.ffmpeg, "run", fake_run)

        from autoclip.config import ExportSettings

        request = _request(
            destination=tmp_path / "out.mp4", burn_captions=False, headline_text=""
        )
        export.export_clip(
            request,
            work_dir=tmp_path / "work",
            settings=ExportSettings(prefer_hardware_encoder=False),
        )

        assert written == []
