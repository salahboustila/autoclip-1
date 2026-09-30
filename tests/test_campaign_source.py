"""Campaign source rules: only the channel's latest episodes; uploads unverified."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from autoclip import app as app_module
from autoclip import campaigns
from autoclip.app import create_app
from autoclip.campaigns import source as campaign_source
from autoclip.campaigns import youtube
from autoclip.cli import app as cli_app
from autoclip.db import store
from autoclip.db.models import Job, Source, new_id
from fastapi.testclient import TestClient
from typer.testing import CliRunner

PRESET = campaigns.load_preset("jack_neel")
LATEST = [f"vid{n:08d}" for n in range(20)]  # 11-character ids
OLD = "oldvideo123"


@pytest.fixture
def channel(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A fake @jackneel/videos tab: 20 episodes, newest first."""

    def fake(url, *, flat, limit=None, **_kw):
        assert flat and "@jackneel/videos" in url
        return {"entries": [{"id": vid, "title": f"Episode {vid}"} for vid in LATEST][:limit]}

    monkeypatch.setattr(youtube, "_extract", fake)
    return LATEST


def url(vid: str) -> str:
    return f"https://www.youtube.com/watch?v={vid}"


class TestCheckUrl:
    def test_one_of_the_latest_20_is_verified(self, autoclip_home, channel) -> None:
        check = campaign_source.check_url(PRESET, url(LATEST[3]))
        assert check.status == "verified"
        assert check.message == "Episode 4 of Jack Neel's latest 20."

    def test_an_older_episode_is_rejected(self, autoclip_home, channel) -> None:
        check = campaign_source.check_url(PRESET, url(OLD))
        assert check.status == "rejected"
        assert "not one of Jack Neel's latest 20 episodes" in check.message

    def test_a_non_youtube_link_is_rejected(self, autoclip_home, channel) -> None:
        assert campaign_source.check_url(PRESET, "https://vimeo.com/1").status == "rejected"

    def test_an_unreadable_channel_leaves_it_unverified(self, autoclip_home) -> None:
        # conftest makes every lookup fail.
        check = campaign_source.check_url(PRESET, url(LATEST[0]))
        assert check.status == "not_verified"
        assert check.allowed


class TestUploads:
    def test_uploads_are_allowed_but_not_verified(self) -> None:
        upload = Source(id="s", type="upload", path="x.mp4", title="ep.mp4")
        check = campaign_source.check_source(PRESET, upload)
        assert check.status == "not_verified"
        assert check.message.startswith("Uploaded file: source not verified.")

    def test_a_preset_can_forbid_uploads(self) -> None:
        strict = PRESET.model_copy(deep=True)
        strict.source.allow_uploads = False
        upload = Source(id="s", type="upload", path="x.mp4")
        assert campaign_source.check_source(strict, upload).status == "rejected"


@pytest.fixture
def client(autoclip_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
    with TestClient(create_app()) as test_client:
        yield test_client


class TestApi:
    def _job(self, client: TestClient, src: Source):
        return client.post(
            "/api/jobs", json={"source_id": src.id, "settings": {"campaign_preset": "jack_neel"}}
        )

    def test_a_job_from_an_old_episode_is_refused(self, client: TestClient, channel) -> None:
        src = store.create_source(Source(id=new_id(), type="youtube", path="x", url=url(OLD)))
        response = self._job(client, src)
        assert response.status_code == 400
        assert "latest 20" in response.json()["detail"]

    def test_a_job_from_a_latest_episode_records_the_check(
        self, client: TestClient, channel
    ) -> None:
        src = store.create_source(Source(id=new_id(), type="youtube", path="x", url=url(LATEST[0])))
        body = self._job(client, src).json()
        assert body["campaign"]["source_check"]["status"] == "verified"
        assert body["campaign"]["name"] == "Jack Neel Podcast"
        assert body["campaign"]["host_check"] is None  # not run yet

    def test_an_upload_job_is_marked_not_verified(self, client: TestClient) -> None:
        src = store.create_source(Source(id=new_id(), type="upload", path="x.mp4"))
        body = self._job(client, src).json()
        assert body["campaign"]["source_check"]["status"] == "not_verified"

    def test_default_jobs_skip_the_check(self, client: TestClient) -> None:
        src = store.create_source(Source(id=new_id(), type="youtube", path="x", url=url(OLD)))
        body = client.post("/api/jobs", json={"source_id": src.id}).json()
        assert body["campaign"] is None

    def test_check_url_endpoint(self, client: TestClient, channel) -> None:
        verified = client.post("/api/campaigns/jack_neel/check-url", json={"url": url(LATEST[0])})
        rejected = client.post("/api/campaigns/jack_neel/check-url", json={"url": url(OLD)})
        assert verified.json()["status"] == "verified"
        assert rejected.json()["status"] == "rejected"

    def test_episodes_endpoint(self, client: TestClient, channel) -> None:
        episodes = client.get("/api/campaigns/jack_neel/episodes").json()
        assert [e["id"] for e in episodes] == LATEST

    def test_episodes_endpoint_reports_an_unreadable_channel(self, client: TestClient) -> None:
        response = client.get("/api/campaigns/jack_neel/episodes")
        assert response.status_code == 502

    def test_campaign_report_endpoint(self, client: TestClient) -> None:
        src = store.create_source(Source(id=new_id(), type="upload", path="x.mp4"))
        job = store.create_job(Job(id=new_id(), source_id=src.id))
        assert client.get(f"/api/jobs/{job.id}/campaign-report").status_code == 404


class TestCli:
    def test_run_refuses_an_old_episode_before_downloading(
        self, autoclip_home, channel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from autoclip.pipeline import ingest

        def never(*_a, **_k):
            raise AssertionError("must not download a rejected episode")

        monkeypatch.setattr(ingest, "ingest_youtube", never)
        result = CliRunner().invoke(cli_app, ["campaign", "run", "jack_neel", url(OLD)])
        assert result.exit_code == 1
        assert "latest 20" in result.output

    def test_list(self, autoclip_home) -> None:
        result = CliRunner().invoke(cli_app, ["campaign", "list"])
        assert result.exit_code == 0
        assert "Jack Neel Podcast" in result.output
