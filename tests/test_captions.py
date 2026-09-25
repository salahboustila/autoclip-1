"""Caption grouping and ASS generation."""

from __future__ import annotations

import pysubs2
import pytest
from autoclip.pipeline import captions
from autoclip.pipeline.transcript import Word


def words_from(spec: list[tuple[str, float, float]]) -> list[Word]:
    return [Word(text=t, start=s, end=e) for t, s, e in spec]


def evenly_spaced(texts: list[str], *, step: float = 0.4) -> list[Word]:
    return [
        Word(text=text, start=i * step, end=i * step + step * 0.8) for i, text in enumerate(texts)
    ]


class TestGrouping:
    def test_respects_the_word_ceiling(self) -> None:
        groups = captions.group_words(evenly_spaced([f"w{i}" for i in range(12)]), max_words=4)

        assert all(len(g.words) <= 4 for g in groups)
        assert sum(len(g.words) for g in groups) == 12

    def test_breaks_on_a_long_pause(self) -> None:
        words = words_from([("one", 0.0, 0.3), ("two", 0.3, 0.6), ("three", 2.0, 2.3)])

        groups = captions.group_words(words, max_words=10, max_gap_s=0.4)

        assert len(groups) == 2
        assert groups[1].words[0].text == "three"

    def test_breaks_after_sentence_punctuation(self) -> None:
        words = words_from([("Hello.", 0.0, 0.3), ("Next", 0.4, 0.7), ("thing", 0.8, 1.0)])

        groups = captions.group_words(words, max_words=10)

        assert len(groups) == 2
        assert groups[0].text == "Hello."

    def test_trailing_sentence_end_does_not_make_an_empty_group(self) -> None:
        words = words_from([("Hello", 0.0, 0.3), ("world.", 0.4, 0.7)])

        groups = captions.group_words(words, max_words=10)

        assert len(groups) == 1
        assert all(g.words for g in groups)

    def test_every_word_appears_exactly_once(self) -> None:
        words = evenly_spaced([f"w{i}" for i in range(37)])

        grouped = [w for g in captions.group_words(words, max_words=4) for w in g.words]

        assert [w.text for w in grouped] == [w.text for w in words]

    def test_empty_input_gives_no_groups(self) -> None:
        assert captions.group_words([]) == []

    def test_group_times_span_its_words(self) -> None:
        groups = captions.group_words(evenly_spaced(["a", "b", "c"]), max_words=3)

        assert groups[0].start == 0.0
        assert groups[0].end == pytest.approx(0.8 + 0.32, abs=0.01)


class TestColour:
    def test_hex_converts_to_rgb(self) -> None:
        colour = captions.hex_to_ass("#FFE500")

        assert (colour.r, colour.g, colour.b) == (255, 229, 0)

    def test_leading_hash_is_optional(self) -> None:
        assert captions.hex_to_ass("FFFFFF").r == 255

    def test_bad_hex_raises(self) -> None:
        with pytest.raises(ValueError):
            captions.hex_to_ass("#FFF")


class TestPresets:
    def test_all_four_presets_exist(self) -> None:
        assert set(captions.PRESETS) == {
            "bold_pop",
            "karaoke_fill",
            "clean_lower",
            "boxed",
        }

    @pytest.mark.parametrize("key", list(captions.PRESETS))
    def test_every_preset_ships_its_font(self, key: str) -> None:
        # A missing font file means libass silently substitutes, and the caption
        # looks nothing like the preview.
        assert captions.get_style(key).font_path.exists()

    def test_unknown_style_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown caption style"):
            captions.get_style("neon-explosion")


class TestAssGeneration:
    @pytest.fixture
    def words(self) -> list[Word]:
        return evenly_spaced(["the", "quick", "brown", "fox", "jumps", "over."])

    @pytest.mark.parametrize("key", list(captions.PRESETS))
    def test_every_preset_produces_events(self, key: str, words: list[Word]) -> None:
        subs = captions.build_ass(words, captions.get_style(key), width=1080, height=1920)

        assert len(subs.events) > 0
        assert captions.STYLE_NAME in subs.styles

    def test_resolution_is_recorded(self, words: list[Word]) -> None:
        subs = captions.build_ass(words, captions.get_style("bold_pop"), width=1080, height=1920)

        assert subs.info["PlayResX"] == "1080"
        assert subs.info["PlayResY"] == "1920"

    def test_font_size_scales_with_height(self, words: list[Word]) -> None:
        tall = captions.build_ass(words, captions.get_style("bold_pop"), width=1080, height=1920)
        square = captions.build_ass(words, captions.get_style("bold_pop"), width=1080, height=1080)

        assert (
            tall.styles[captions.STYLE_NAME].fontsize > square.styles[captions.STYLE_NAME].fontsize
        )

    def test_time_offset_rebases_to_clip_relative(self, words: list[Word]) -> None:
        offset = captions.build_ass(
            words,
            captions.get_style("clean_lower"),
            width=1080,
            height=1920,
            time_offset_s=100.0,
        )

        # Absolute times were 0-2.7s, so everything clamps to zero rather than
        # going negative.
        assert offset.events[0].start == 0

    def test_offset_preserves_relative_spacing(self) -> None:
        words = [Word(text=t, start=100.0 + i, end=100.5 + i) for i, t in enumerate("abc")]

        subs = captions.build_ass(
            words,
            captions.get_style("clean_lower"),
            width=1080,
            height=1920,
            time_offset_s=100.0,
        )

        assert subs.events[0].start == pytest.approx(0, abs=20)

    def test_all_caps_is_applied(self, words: list[Word]) -> None:
        subs = captions.build_ass(words, captions.get_style("bold_pop"), width=1080, height=1920)

        assert "QUICK" in " ".join(e.text for e in subs.events)

    def test_clean_lower_preserves_case(self, words: list[Word]) -> None:
        subs = captions.build_ass(words, captions.get_style("clean_lower"), width=1080, height=1920)

        assert "quick" in " ".join(e.text for e in subs.events)

    def test_karaoke_emits_kf_tags(self, words: list[Word]) -> None:
        subs = captions.build_ass(
            words, captions.get_style("karaoke_fill"), width=1080, height=1920
        )

        assert any("\\kf" in event.text for event in subs.events)

    def test_bold_pop_emits_one_event_per_word(self, words: list[Word]) -> None:
        # Per-word highlighting has no per-glyph timeline in libass, so each
        # word needs its own event showing the whole group.
        subs = captions.build_ass(words, captions.get_style("bold_pop"), width=1080, height=1920)

        assert len(subs.events) == len(words)

    def test_clean_lower_emits_one_event_per_group(self, words: list[Word]) -> None:
        style = captions.get_style("clean_lower")
        groups = captions.group_words(words, max_words=style.max_words)

        subs = captions.build_ass(words, style, width=1080, height=1920)

        assert len(subs.events) == len(groups)

    def test_boxed_uses_an_opaque_border_style(self, words: list[Word]) -> None:
        subs = captions.build_ass(words, captions.get_style("boxed"), width=1080, height=1920)

        assert subs.styles[captions.STYLE_NAME].borderstyle == 3

    def test_events_never_start_before_zero(self, words: list[Word]) -> None:
        subs = captions.build_ass(
            words,
            captions.get_style("bold_pop"),
            width=1080,
            height=1920,
            time_offset_s=1.0,
        )

        assert all(event.start >= 0 for event in subs.events)

    def test_written_file_is_valid_ass(self, words: list[Word], tmp_path) -> None:
        path = captions.write_ass(
            tmp_path / "out.ass",
            words,
            captions.get_style("bold_pop"),
            width=1080,
            height=1920,
        )

        reloaded = pysubs2.load(str(path), encoding="utf-8")

        assert len(reloaded.events) > 0

    def test_srt_sidecar_is_written(self, words: list[Word], tmp_path) -> None:
        path = captions.write_srt(tmp_path / "out.srt", words)

        assert path.exists()
        assert len(pysubs2.load(str(path), encoding="utf-8").events) > 0


class TestHeadlineWrap:
    def test_short_text_fits_on_one_line(self) -> None:
        result = captions._wrap_headline("SHORT TEXT", max_chars_per_line=40, max_lines=3)

        assert r"\N" not in result

    def test_wraps_at_a_word_boundary(self) -> None:
        result = captions._wrap_headline(
            "ONE TWO THREE FOUR", max_chars_per_line=8, max_lines=3
        )

        # Never split a word itself, whatever the line breaks land on.
        assert set(result.replace(r"\N", " ").split(" ")) == {"ONE", "TWO", "THREE", "FOUR"}

    def test_never_exceeds_max_lines(self) -> None:
        text = "ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN"

        result = captions._wrap_headline(text, max_chars_per_line=4, max_lines=3)

        assert result.count(r"\N") <= 2  # at most 3 lines = 2 breaks

    def test_overflow_words_land_on_the_last_line_rather_than_vanishing(self) -> None:
        """No word the AI wrote is ever silently dropped, even under an
        unreasonably tight budget — see _wrap_headline's own docstring."""
        text = "ONE TWO THREE FOUR FIVE SIX"

        result = captions._wrap_headline(text, max_chars_per_line=3, max_lines=2)

        assert set(result.replace(r"\N", " ").split(" ")) == set(text.split(" "))

    def test_empty_text_gives_empty_result(self) -> None:
        assert captions._wrap_headline("", max_chars_per_line=40, max_lines=3) == ""

    def test_single_long_word_is_not_split(self) -> None:
        result = captions._wrap_headline("SUPERCALIFRAGILISTIC", max_chars_per_line=5, max_lines=3)

        assert result == "SUPERCALIFRAGILISTIC"


class TestHeadlineOverlay:
    def test_no_headline_leaves_events_untouched(self) -> None:
        subs = pysubs2.SSAFile()
        subs.info["PlayResX"] = "1080"
        subs.info["PlayResY"] = "1920"

        captions.add_headline(
            subs,
            captions.HeadlineStyle(text=""),
            width=1080,
            height=1920,
            duration_s=10.0,
        )

        assert len(subs.events) == 0

    def test_adds_exactly_one_event_for_the_whole_clip_duration(self) -> None:
        subs = pysubs2.SSAFile()

        captions.add_headline(
            subs,
            captions.HeadlineStyle(text="A real headline"),
            width=1080,
            height=1920,
            duration_s=12.5,
        )

        assert len(subs.events) == 1
        event = subs.events[0]
        assert event.start == 0
        assert event.end == pysubs2.make_time(s=12.5)

    def test_text_is_upper_cased(self) -> None:
        subs = pysubs2.SSAFile()

        captions.add_headline(
            subs,
            captions.HeadlineStyle(text="lower case words"),
            width=1080,
            height=1920,
            duration_s=5.0,
        )

        assert subs.events[0].text == subs.events[0].text.upper()

    def test_uses_an_opaque_border_style_for_the_background_box(self) -> None:
        subs = pysubs2.SSAFile()

        captions.add_headline(
            subs,
            captions.HeadlineStyle(text="Boxed headline"),
            width=1080,
            height=1920,
            duration_s=5.0,
        )

        style = subs.styles[subs.events[0].style]
        assert style.borderstyle == 3

    def test_top_position_sits_closer_to_the_edge_than_upper_center(self) -> None:
        top = pysubs2.SSAFile()
        captions.add_headline(
            top,
            captions.HeadlineStyle(text="X", position="top"),
            width=1080,
            height=1920,
            duration_s=5.0,
        )
        upper_center = pysubs2.SSAFile()
        captions.add_headline(
            upper_center,
            captions.HeadlineStyle(text="X", position="upper-center"),
            width=1080,
            height=1920,
            duration_s=5.0,
        )

        top_margin = top.styles[top.events[0].style].marginv
        upper_center_margin = upper_center.styles[upper_center.events[0].style].marginv
        assert top_margin < upper_center_margin

    def test_full_opacity_and_zero_opacity_produce_different_backcolor_alpha(self) -> None:
        opaque = pysubs2.SSAFile()
        captions.add_headline(
            opaque,
            captions.HeadlineStyle(text="X", bg_opacity_pct=100.0),
            width=1080,
            height=1920,
            duration_s=5.0,
        )
        transparent = pysubs2.SSAFile()
        captions.add_headline(
            transparent,
            captions.HeadlineStyle(text="X", bg_opacity_pct=0.0),
            width=1080,
            height=1920,
            duration_s=5.0,
        )

        opaque_alpha = opaque.styles[opaque.events[0].style].backcolor.a
        transparent_alpha = transparent.styles[transparent.events[0].style].backcolor.a
        # ASS alpha is inverted: 0 = fully opaque, 255 = fully transparent.
        assert opaque_alpha < transparent_alpha

    def test_text_colour_is_applied(self) -> None:
        subs = pysubs2.SSAFile()

        captions.add_headline(
            subs,
            captions.HeadlineStyle(text="X", text_color="#FF0000"),
            width=1080,
            height=1920,
            duration_s=5.0,
        )

        colour = subs.styles[subs.events[0].style].primarycolor
        assert (colour.r, colour.g, colour.b) == (255, 0, 0)

    def test_headline_style_does_not_collide_with_the_caption_style_name(self) -> None:
        """Both live in the same SSAFile when write_ass layers them together —
        distinct style names are what keeps libass from confusing them."""
        subs = pysubs2.SSAFile()
        subs.styles[captions.STYLE_NAME] = pysubs2.SSAStyle()

        captions.add_headline(
            subs,
            captions.HeadlineStyle(text="X"),
            width=1080,
            height=1920,
            duration_s=5.0,
        )

        assert subs.events[0].style != captions.STYLE_NAME
        assert captions.STYLE_NAME in subs.styles  # untouched, not overwritten


class TestWriteAssWithHeadline:
    @pytest.fixture
    def words(self) -> list[Word]:
        return evenly_spaced(["the", "quick", "brown", "fox", "jumps", "over."])

    def test_headline_layers_onto_existing_caption_events(
        self, words: list[Word], tmp_path
    ) -> None:
        path = captions.write_ass(
            tmp_path / "out.ass",
            words,
            captions.get_style("bold_pop"),
            width=1080,
            height=1920,
            headline=captions.HeadlineStyle(text="Breaking overlay news"),
            clip_duration_s=10.0,
        )

        reloaded = pysubs2.load(str(path), encoding="utf-8")
        # bold_pop emits one event per word (see TestAssGeneration above) plus
        # exactly one more for the headline.
        assert len(reloaded.events) == len(words) + 1

    def test_headline_works_with_no_word_captions_at_all(self, tmp_path) -> None:
        """Headline on, word captions off: write_ass must still produce a
        valid file with just the one headline event, not fail on empty words."""
        path = captions.write_ass(
            tmp_path / "headline_only.ass",
            [],
            captions.get_style("bold_pop"),
            width=1080,
            height=1920,
            headline=captions.HeadlineStyle(text="Headline only, no captions"),
            clip_duration_s=8.0,
        )

        reloaded = pysubs2.load(str(path), encoding="utf-8")
        assert len(reloaded.events) == 1

    def test_omitting_headline_is_unchanged_from_before_the_feature_existed(
        self, words: list[Word], tmp_path
    ) -> None:
        with_default = captions.write_ass(
            tmp_path / "a.ass", words, captions.get_style("bold_pop"), width=1080, height=1920
        )
        explicit_none = captions.write_ass(
            tmp_path / "b.ass",
            words,
            captions.get_style("bold_pop"),
            width=1080,
            height=1920,
            headline=None,
        )

        assert with_default.read_text(encoding="utf-8") == explicit_none.read_text(encoding="utf-8")
