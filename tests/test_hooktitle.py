"""Hook title: emoji splitting, layout rules, the export wiring, and the API."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from autoclip import app as app_module
from autoclip import config, db
from autoclip.app import create_app
from autoclip.config import HookTitleSettings
from autoclip.db import store
from autoclip.db.models import Clip, Job, Source, new_id
from autoclip.pipeline import captions, export, hooktitle
from autoclip.pipeline.reframe.croppath import centre_crop
from fastapi.testclient import TestClient
from PIL import Image

W, H = 1080, 1920
BOLD_POP = captions.get_style("bold_pop")


def render(text: str, **kwargs):
    kwargs.setdefault("bottom_limit_px", hooktitle.caption_top_px(BOLD_POP, H))
    return hooktitle.render_title(text, width=W, height=H, **kwargs)


class TestEmojiRuns:
    def test_text_only(self) -> None:
        assert hooktitle.split_runs("No emoji here") == [("No emoji here", False)]

    def test_trailing_emoji(self) -> None:
        assert hooktitle.split_runs("Wait for it 😱") == [("Wait for it ", False), ("😱", True)]

    def test_multi_codepoint_clusters_stay_whole(self) -> None:
        runs = hooktitle.split_runs("a👨‍👩‍👧b🇺🇸c👍🏽d❤️")
        emoji = [run for run, is_emoji in runs if is_emoji]
        assert emoji == ["👨‍👩‍👧", "🇺🇸", "👍🏽", "❤️"]

    def test_punctuation_numbers_and_symbols_are_text(self) -> None:
        text = "69% Voted Not Guilty For This?! $100k #1 10+"
        assert hooktitle.split_runs(text) == [(text, False)]


class TestLayout:
    def test_short_title_is_one_line(self) -> None:
        _, layout = render("Learn This")
        assert len(layout.lines) == 1

    def test_long_title_wraps_to_at_most_two_balanced_lines(self) -> None:
        _, layout = render("Multi-Millionaire Reveals Shocking Study on Women 💰")
        assert len(layout.lines) == 2
        first, second = (len(line) for line in layout.text_lines)
        assert abs(first - second) < 20

    def test_very_long_title_shrinks_instead_of_a_third_line(self) -> None:
        text = "If You Want a High-Value Man You Really Need to Learn This One Thing First"
        _, layout = render(text, font_size=90)
        assert len(layout.lines) <= 2
        assert layout.font_size < 90

    def test_absurd_title_is_truncated_not_overflowed(self) -> None:
        _, layout = render(" ".join(["Unbelievably"] * 30))
        assert layout.truncated
        assert len(layout.lines) <= 2
        assert layout.text_lines[-1].endswith("…")

    def test_default_top_is_13_percent(self) -> None:
        _, layout = render("Hello There")
        assert layout.box[1] == round(H * 0.13)

    @pytest.mark.parametrize("pct", [0, 5, 11.9])
    def test_never_above_12_percent(self, pct: float) -> None:
        _, layout = render("Hello There", position_pct=pct)
        assert layout.box[1] >= round(H * 0.12)

    @pytest.mark.parametrize("ratio", ["9:16", "1:1", "16:9"])
    @pytest.mark.parametrize("style_key", list(captions.PRESETS))
    def test_never_overlaps_captions(self, ratio: str, style_key: str) -> None:
        width, height = export.ratio_dimensions(ratio)
        limit = hooktitle.caption_top_px(captions.get_style(style_key), height)
        _, layout = hooktitle.render_title(
            "Multi-Millionaire Reveals Shocking Study on Women 💰",
            width=width,
            height=height,
            position_pct=40,
            bottom_limit_px=limit,
        )
        assert layout.box[3] <= limit

    def test_box_is_centred_and_inside_the_frame(self) -> None:
        _, layout = render("How Much Does the Average American Man Make? 🤔")
        left, _, right, _ = layout.box
        assert left > 0 and right < W
        assert abs(left - (W - right)) <= 1

    def test_empty_title_is_an_error(self) -> None:
        with pytest.raises(hooktitle.HookTitleError):
            render("   ")


class TestPixels:
    def test_box_is_white_text_is_black_and_the_rest_transparent(self) -> None:
        image, layout = render("Learn This")
        left, top, right, bottom = layout.box
        assert image.getpixel((left + 8, (top + bottom) // 2)) == (255, 255, 255, 255)
        assert image.getpixel((5, 5))[3] == 0
        box = image.crop(layout.box).convert("L")
        assert min(box.getdata()) < 40  # black text inside the box

    def test_emoji_draws_in_colour(self) -> None:
        image, layout = render("Money 💰")
        box = image.crop(layout.box).convert("RGB")
        coloured = [p for p in box.getdata() if max(p) - min(p) > 60]
        assert len(coloured) > 200

    def test_writes_a_full_frame_png(self, tmp_path: Path) -> None:
        path = tmp_path / "t.png"
        hooktitle.write_title_png(path, "Hello 🤔", width=W, height=H)
        with Image.open(path) as png:
            assert png.size == (W, H)
            assert png.mode == "RGBA"


class TestSettings:
    def test_defaults(self) -> None:
        settings = config.Settings().hook_title
        assert settings.enabled is True
        assert settings.position_pct == 13.0

    def test_position_is_clamped_to_12(self) -> None:
        assert HookTitleSettings(position_pct=3).position_pct == 12.0


class TestExportWiring:
    def _request(self, **kwargs) -> export.ExportRequest:
        return export.ExportRequest(
            source=Path("in.mp4"),
            destination=Path("out.mp4"),
            start_s=0.0,
            end_s=5.0,
            crop_path=centre_crop(1920, 1080, 5.0),
            words=[],
            style=BOLD_POP,
            **kwargs,
        )

    def test_no_title_leaves_the_filtergraph_unchanged(self) -> None:
        plain = export.build_video_filtergraph(self._request(), subtitle_name="captions.ass")
        assert "overlay" not in plain
        assert self._request(hook_title="").wants_hook_title is False
        assert self._request(hook_title="Hi").wants_hook_title is False  # no settings

    def test_disabled_setting_means_no_title(self) -> None:
        request = self._request(
            hook_title="Hi", hook_title_settings=HookTitleSettings(enabled=False)
        )
        assert request.wants_hook_title is False

    def test_title_overlays_before_captions(self) -> None:
        request = self._request(hook_title="Hi", hook_title_settings=HookTitleSettings())
        graph = export.build_video_filtergraph(request, subtitle_name="captions.ass", title_input=1)
        assert "[1:v]overlay=0:0[vtitle]" in graph
        assert graph.index("overlay") < graph.index("ass=")


class TestMigrationAndApi:
    def test_migration_adds_hook_title_with_empty_default(self, initialised_db) -> None:
        with db.connection() as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(clips)")}
        assert "hook_title" in columns

    @pytest.fixture
    def client(self, autoclip_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
        monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
        with TestClient(create_app()) as test_client:
            yield test_client

    def test_hook_title_round_trips_through_patch(self, client: TestClient) -> None:
        source = store.create_source(Source(id=new_id(), type="upload", path="x.mp4"))
        job = store.create_job(Job(id=new_id(), source_id=source.id, status="done"))
        clip = Clip(id=new_id(), job_id=job.id, start_s=0, end_s=30)
        store.replace_clips(job.id, [clip])

        assert client.get(f"/api/clips/{clip.id}").json()["hook_title"] == ""

        response = client.patch(
            f"/api/clips/{clip.id}", json={"hook_title": "  Learn   This 🤔 "}
        )
        assert response.status_code == 200
        assert response.json()["hook_title"] == "Learn This 🤔"
        assert store.get_clip(clip.id).hook_title == "Learn This 🤔"

        cleared = client.patch(f"/api/clips/{clip.id}", json={"hook_title": ""})
        assert cleared.json()["hook_title"] == ""

    def test_settings_api_exposes_hook_title(self, client: TestClient) -> None:
        body = client.put("/api/settings", json={"hook_title": {"position_pct": 20}}).json()
        assert body["hook_title"]["position_pct"] == 20
        assert config.load().hook_title.position_pct == 20
