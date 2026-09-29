"""API additions behind the Viral Hook Mode UI."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from autoclip import app as app_module
from autoclip import config, paths
from autoclip.app import create_app
from autoclip.db import store
from autoclip.db.models import Clip, Job, Source, new_id
from fastapi.testclient import TestClient


@pytest.fixture
def client(autoclip_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
    with TestClient(create_app()) as test_client:
        yield test_client


def _source(path: str = "missing.mp4") -> Source:
    return store.create_source(Source(id=new_id(), type="upload", path=path))


class TestJobFlags:
    def test_media_available_follows_the_file(self, client: TestClient) -> None:
        paths.ensure_layout()
        video = paths.media_dir() / "v.mp4"
        video.write_bytes(b"x")
        job = store.create_job(Job(id=new_id(), source_id=_source(str(video)).id))

        assert client.get(f"/api/jobs/{job.id}").json()["source"]["media_available"] is True
        video.unlink()
        assert client.get(f"/api/jobs/{job.id}").json()["source"]["media_available"] is False

    def test_viral_hook_flag(self, client: TestClient) -> None:
        source = _source()
        plain = client.post("/api/jobs", json={"source_id": source.id}).json()
        viral = client.post(
            "/api/jobs", json={"source_id": source.id, "settings": {"viral_hook": True}}
        ).json()
        assert plain["viral_hook"] is False
        assert viral["viral_hook"] is True


class TestOverridesFollowTheMode:
    def _job_settings(self, client: TestClient, **overrides) -> config.Settings:
        body = client.post(
            "/api/jobs", json={"source_id": _source().id, "settings": overrides}
        ).json()
        return config.Settings.model_validate(store.get_job(body["id"]).settings)

    def test_viral_mode_takes_count_and_lengths(self, client: TestClient) -> None:
        settings = self._job_settings(
            client, viral_hook=True, max_clips=6, min_duration_s=30, max_duration_s=45
        )
        viral = settings.viral_hook
        assert (viral.top_n, viral.min_duration_s, viral.max_duration_s) == (6, 30, 45)

    def test_viral_defaults_stand_without_overrides(self, client: TestClient) -> None:
        viral = self._job_settings(client, viral_hook=True).viral_hook
        assert (viral.top_n, viral.min_duration_s, viral.max_duration_s) == (10, 25, 50)

    def test_default_mode_is_unchanged(self, client: TestClient) -> None:
        settings = self._job_settings(client, max_clips=6)
        assert settings.clips.max_clips == 6
        assert settings.viral_hook.top_n == 10

    def test_inverted_viral_lengths_are_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/jobs",
            json={
                "source_id": _source().id,
                "settings": {"viral_hook": True, "min_duration_s": 60},
            },
        )
        assert response.status_code == 400


class TestAlternatives:
    @pytest.fixture
    def clip(self) -> Clip:
        job = store.create_job(Job(id=new_id(), source_id=_source().id))
        clip = Clip(
            id=new_id(), job_id=job.id, start_s=0, end_s=30,
            hook_title="Title A 😳", hook_title_alts=["Title B 💰", "Title C 🤔"],
        )  # fmt: skip
        store.replace_clips(job.id, [clip])
        return clip

    def test_swapping_in_an_alternative(self, client: TestClient, clip: Clip) -> None:
        body = client.patch(
            f"/api/clips/{clip.id}",
            json={"hook_title": "Title B 💰", "hook_title_alts": ["Title A 😳", "Title C 🤔"]},
        ).json()
        assert body["hook_title"] == "Title B 💰"
        assert body["hook_title_alts"] == ["Title A 😳", "Title C 🤔"]

    def test_typing_an_alternative_removes_it_from_the_list(
        self, client: TestClient, clip: Clip
    ) -> None:
        body = client.patch(f"/api/clips/{clip.id}", json={"hook_title": "Title C 🤔"}).json()
        assert body["hook_title"] == "Title C 🤔"
        assert body["hook_title_alts"] == ["Title B 💰"]

    def test_a_new_title_leaves_the_alternatives(self, client: TestClient, clip: Clip) -> None:
        body = client.patch(f"/api/clips/{clip.id}", json={"hook_title": "Brand New 👀"}).json()
        assert body["hook_title_alts"] == ["Title B 💰", "Title C 🤔"]
