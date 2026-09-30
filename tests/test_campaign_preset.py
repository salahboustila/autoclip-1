"""Podcast Campaign Mode: presets, settings, and job wiring."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from autoclip import app as app_module
from autoclip import campaigns, config, paths
from autoclip.app import create_app
from autoclip.campaigns.preset import BUNDLED_DIR
from autoclip.db import store
from autoclip.db.models import Job, Source, new_id
from autoclip.pipeline.runner import PipelineRunner
from fastapi.testclient import TestClient


@pytest.fixture
def client(autoclip_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
    with TestClient(create_app()) as test_client:
        yield test_client


class TestJackNeelPreset:
    def test_loads_with_the_campaign_rules(self) -> None:
        preset = campaigns.load_preset("jack_neel")
        assert preset.name == "Jack Neel Podcast"
        assert preset.source.channel_url == "https://www.youtube.com/@jackneel/videos"
        assert preset.source.latest_episodes == 20
        assert preset.copy_.caption_lines == ["@jackhneel", "@jackneel", "#jackneelpod"]
        assert (preset.detection.min_duration_s, preset.detection.max_duration_s) == (25, 50)
        assert preset.detection.top_n == 10
        assert preset.detection.fallback_pass is False
        assert preset.render.watermark is False

    def test_is_listed(self) -> None:
        assert {"key": "jack_neel", "name": "Jack Neel Podcast", "platform": "Content Rewards"} in (
            campaigns.list_presets()
        )


class TestPresetFiles:
    def _write(self, directory: Path, key: str, body: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{key}.yaml").write_text(body, encoding="utf-8")

    def test_a_user_preset_overrides_the_bundled_one(self, autoclip_home) -> None:
        bundled = (BUNDLED_DIR / "jack_neel.yaml").read_text(encoding="utf-8")
        self._write(paths.root() / "presets", "jack_neel", bundled.replace("top_n: 10", "top_n: 4"))
        assert campaigns.load_preset("jack_neel").detection.top_n == 4

    def test_a_new_campaign_is_just_a_file(self, autoclip_home) -> None:
        self._write(paths.root() / "presets", "other_pod", "key: other_pod\nname: Other Pod\n")
        preset = campaigns.load_preset("other_pod")
        assert preset.detection.top_n == 10  # defaults fill the rest
        assert "other_pod" in {p["key"] for p in campaigns.list_presets()}

    @pytest.mark.parametrize(
        "body",
        [
            "key: bad\nname: Bad\nrender: {watermark: true}\n",  # watermarks are never allowed
            "key: bad\nname: Bad\nunknown_section: 1\n",
            "key: wrong_key\nname: Bad\n",
            "key: bad\nname: Bad\ndetection: {min_duration_s: 60, max_duration_s: 30}\n",
            "key: bad\nname: [unclosed\n",
        ],
    )
    def test_invalid_presets_are_rejected(self, autoclip_home, body: str) -> None:
        self._write(paths.root() / "presets", "bad", body)
        with pytest.raises(campaigns.PresetError):
            campaigns.load_preset("bad")

    @pytest.mark.parametrize("key", ["missing", "../etc/passwd", ""])
    def test_unknown_or_unsafe_names(self, key: str) -> None:
        with pytest.raises(campaigns.PresetError):
            campaigns.load_preset(key)


class TestApply:
    def test_off_returns_the_same_settings_object(self) -> None:
        settings = config.Settings()
        assert campaigns.apply(settings) is settings
        assert campaigns.rules_of(settings) is None

    def test_on_lays_the_preset_over_viral_hook_mode(self) -> None:
        settings = config.Settings()
        settings.campaign.enabled = True
        merged = campaigns.apply(settings)

        assert settings.viral_hook.enabled is False  # the input is not mutated
        viral = merged.viral_hook
        assert viral.enabled is True
        assert viral.prompt_version == "campaign_question_first_v1"
        assert (viral.top_n, viral.min_duration_s, viral.max_duration_s) == (10, 25, 50)
        assert viral.layout == "podcast_hook"
        assert viral.caption_handle == "" and viral.fixed_hashtags == []
        assert merged.clips.min_score == 70
        assert merged.whisper.diarization is True
        assert campaigns.rules_of(merged).key == "jack_neel"

    def test_is_idempotent(self) -> None:
        settings = config.Settings()
        settings.campaign.enabled = True
        once = campaigns.apply(settings)
        assert campaigns.apply(once) is once

    def test_keeps_plaintext_fallback_secrets(self) -> None:
        settings = config.Settings()
        settings.campaign.enabled = True
        settings._fallback_secrets["anthropic"] = "sk"
        assert campaigns.apply(settings)._fallback_secrets == {"anthropic": "sk"}

    def test_runner_applies_it_for_cli_runs(self, initialised_db) -> None:
        settings = config.Settings()
        settings.campaign.enabled = True
        source = store.create_source(Source(id=new_id(), type="upload", path="x.mp4"))
        job = store.create_job(Job(id=new_id(), source_id=source.id))
        runner = PipelineRunner(job, source, settings=settings)
        assert runner.settings.viral_hook.enabled is True
        assert campaigns.rules_of(runner.settings) is not None


class TestJobs:
    def _source(self) -> Source:
        return store.create_source(Source(id=new_id(), type="upload", path="x.mp4"))

    def test_campaign_job_snapshots_the_preset(self, client: TestClient) -> None:
        body = client.post(
            "/api/jobs",
            json={"source_id": self._source().id, "settings": {"campaign_preset": "jack_neel"}},
        ).json()
        settings = config.Settings.model_validate(store.get_job(body["id"]).settings)
        assert settings.campaign.enabled is True
        assert settings.campaign.rules["key"] == "jack_neel"
        assert settings.viral_hook.enabled is True

    def test_per_job_overrides_beat_the_preset(self, client: TestClient) -> None:
        body = client.post(
            "/api/jobs",
            json={
                "source_id": self._source().id,
                "settings": {"campaign_preset": "jack_neel", "max_clips": 3},
            },
        ).json()
        settings = config.Settings.model_validate(store.get_job(body["id"]).settings)
        assert settings.viral_hook.top_n == 3

    def test_default_jobs_are_unchanged(self, client: TestClient) -> None:
        body = client.post("/api/jobs", json={"source_id": self._source().id}).json()
        settings = config.Settings.model_validate(store.get_job(body["id"]).settings)
        assert settings.campaign.enabled is False
        assert settings.campaign.rules is None
        assert settings.viral_hook.enabled is False
        assert settings.clips.min_score == 50

    def test_unknown_preset_is_a_400(self, client: TestClient) -> None:
        response = client.post(
            "/api/jobs",
            json={"source_id": self._source().id, "settings": {"campaign_preset": "nope"}},
        )
        assert response.status_code == 400
        assert "nope" in response.json()["detail"]

    def test_empty_string_turns_a_saved_default_off(self, client: TestClient) -> None:
        saved = config.Settings()
        saved.campaign.enabled = True
        config.save(saved)
        body = client.post(
            "/api/jobs",
            json={"source_id": self._source().id, "settings": {"campaign_preset": ""}},
        ).json()
        settings = config.Settings.model_validate(store.get_job(body["id"]).settings)
        assert settings.campaign.enabled is False


class TestApi:
    def test_lists_and_describes_presets(self, client: TestClient) -> None:
        assert client.get("/api/campaigns").json()[0]["key"] == "jack_neel"
        detail = client.get("/api/campaigns/jack_neel").json()
        assert detail["caption_lines"] == ["@jackhneel", "@jackneel", "#jackneelpod"]
        assert client.get("/api/campaigns/nope").status_code == 404

    def test_settings_round_trip_never_saves_a_snapshot(self, client: TestClient) -> None:
        body = client.put(
            "/api/settings", json={"campaign": {"enabled": True, "rules": {"key": "x"}}}
        ).json()
        assert body["campaign"]["enabled"] is True
        assert body["campaign"]["rules"] is None
        assert config.load().campaign.rules is None

    def test_settings_reject_an_unknown_preset(self, client: TestClient) -> None:
        response = client.put(
            "/api/settings", json={"campaign": {"enabled": True, "preset": "nope"}}
        )
        assert response.status_code == 400
