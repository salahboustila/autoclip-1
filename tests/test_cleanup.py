"""Job deletion, the download-all zip, and post-export auto-cleanup."""

from __future__ import annotations

import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from autoclip import app as app_module
from autoclip import cleanup, config, paths
from autoclip.app import create_app
from autoclip.db import store
from autoclip.db.models import Clip, Export, Job, Source, new_id
from autoclip.pipeline.runner import JobWorkspace, PipelineRunner
from fastapi.testclient import TestClient


@pytest.fixture
def client(autoclip_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
    with TestClient(create_app()) as test_client:
        yield test_client


def make_job(*, status: str = "done", clips: int = 2, exported: bool = True) -> tuple[Job, Source]:
    """A job with real files on disk: source video, work files, exports."""
    paths.ensure_layout()
    source_id = new_id()
    media = paths.source_media_dir(source_id)
    media.mkdir(parents=True)
    video = media / "video.mp4"
    video.write_bytes(b"source")
    source = store.create_source(
        Source(id=source_id, type="upload", path=str(video), title="t", duration_s=60.0)
    )
    job = store.create_job(Job(id=new_id(), source_id=source.id, status=status))

    work = JobWorkspace(job.id).root
    (work / "audio.wav").write_bytes(b"audio")

    out = paths.exports_dir() / job.id
    out.mkdir(parents=True, exist_ok=True)
    created = [
        Clip(id=new_id(), job_id=job.id, start_s=i * 10.0, end_s=i * 10.0 + 9, rank=i)
        for i in range(clips)
    ]
    store.replace_clips(job.id, created)
    if exported:
        for i, clip in enumerate(created):
            mp4 = out / f"clip-{i}_9x16.mp4"
            mp4.write_bytes(b"mp4" * 10)
            store.create_export(Export(id=new_id(), clip_id=clip.id, path=str(mp4)))
        (out / "clip_01_caption.txt").write_text("caption one")
    return job, source


class TestDeleteJob:
    def test_removes_files_and_rows(self, client: TestClient) -> None:
        job, source = make_job()

        assert client.delete(f"/api/jobs/{job.id}").status_code == 204

        assert store.get_job(job.id) is None
        assert store.list_clips(job.id) == []
        assert store.get_source(source.id) is None
        assert not Path(source.path).exists()
        assert not paths.job_work_dir(job.id).exists()
        assert not (paths.exports_dir() / job.id).exists()

    def test_keeps_a_source_another_job_uses(self, client: TestClient) -> None:
        job, source = make_job()
        other = store.create_job(Job(id=new_id(), source_id=source.id, status="done"))

        assert client.delete(f"/api/jobs/{job.id}").status_code == 204

        assert store.get_job(other.id) is not None
        assert store.get_source(source.id) is not None
        assert Path(source.path).exists()

    def test_refuses_a_running_job(self, client: TestClient) -> None:
        job, source = make_job(status="running")

        assert client.delete(f"/api/jobs/{job.id}").status_code == 409
        assert store.get_job(job.id) is not None
        assert Path(source.path).exists()

    def test_unknown_job_is_404(self, client: TestClient) -> None:
        assert client.delete("/api/jobs/nope").status_code == 404

    def test_never_deletes_a_file_outside_the_media_directory(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        outside = tmp_path / "precious.mp4"
        outside.write_bytes(b"mine")
        source = store.create_source(
            Source(id=new_id(), type="upload", path=str(outside), title="t")
        )
        job = store.create_job(Job(id=new_id(), source_id=source.id, status="done"))

        assert client.delete(f"/api/jobs/{job.id}").status_code == 204
        assert outside.read_bytes() == b"mine"


class TestDownloadAll:
    def test_zips_every_clip_and_caption_file(self, client: TestClient, tmp_path: Path) -> None:
        job, _ = make_job()

        response = client.get(f"/api/jobs/{job.id}/download-all")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"
        archive = tmp_path / "out.zip"
        archive.write_bytes(response.content)
        with zipfile.ZipFile(archive) as zf:
            assert sorted(zf.namelist()) == [
                "clip-0_9x16.mp4",
                "clip-1_9x16.mp4",
                "clip_01_caption.txt",
            ]

    def test_uses_only_the_newest_export_of_a_clip(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        job, _ = make_job(clips=1)
        clip = store.list_clips(job.id)[0]
        newer = paths.exports_dir() / job.id / "clip-0-v2_9x16.mp4"
        newer.write_bytes(b"new")
        store.create_export(
            Export(id=new_id(), clip_id=clip.id, path=str(newer), created_at="2999-01-01T00:00:00")
        )

        archive = cleanup.build_zip(job.id)

        assert archive is not None
        with zipfile.ZipFile(archive) as zf:
            assert "clip-0-v2_9x16.mp4" in zf.namelist()
            assert "clip-0_9x16.mp4" not in zf.namelist()
        archive.unlink()

    def test_no_exports_is_404(self, client: TestClient) -> None:
        job, _ = make_job(exported=False)

        assert client.get(f"/api/jobs/{job.id}/download-all").status_code == 404


class TestAutoCleanup:
    def _runner(self, job: Job, source: Source, *, enabled: bool) -> PipelineRunner:
        settings = config.Settings()
        settings.cleanup.auto_delete_sources = enabled
        return PipelineRunner(job, source, settings=settings)

    def test_default_is_on(self) -> None:
        assert config.Settings().cleanup.auto_delete_sources is True

    def test_keeps_only_the_final_clips(self, autoclip_home: Path, initialised_db) -> None:
        job, source = make_job()

        self._runner(job, source, enabled=True)._auto_cleanup()

        assert not Path(source.path).exists()
        assert not paths.job_work_dir(job.id).exists()
        assert len(list((paths.exports_dir() / job.id).glob("*.mp4"))) == 2
        assert store.get_job(job.id) is not None

    def test_disabled_leaves_everything(self, autoclip_home: Path, initialised_db) -> None:
        job, source = make_job()

        self._runner(job, source, enabled=False)._auto_cleanup()

        assert Path(source.path).exists()
        assert paths.job_work_dir(job.id).exists()

    def test_keeps_a_source_a_queued_job_still_needs(
        self, autoclip_home: Path, initialised_db
    ) -> None:
        job, source = make_job()
        store.create_job(Job(id=new_id(), source_id=source.id, status="queued"))

        self._runner(job, source, enabled=True)._auto_cleanup()

        assert Path(source.path).exists()
        assert not paths.job_work_dir(job.id).exists()

    def test_cleanup_failure_never_fails_the_job(
        self, autoclip_home: Path, initialised_db, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        job, source = make_job()

        def boom(*_a, **_k):
            raise OSError("disk on fire")

        monkeypatch.setattr(cleanup, "purge_after_export", boom)
        self._runner(job, source, enabled=True)._auto_cleanup()  # must not raise


class TestSettingsApi:
    def test_cleanup_setting_round_trips(self, client: TestClient) -> None:
        assert client.get("/api/settings").json()["cleanup"]["auto_delete_sources"] is True

        response = client.put("/api/settings", json={"cleanup": {"auto_delete_sources": False}})

        assert response.json()["cleanup"]["auto_delete_sources"] is False
        assert config.load().cleanup.auto_delete_sources is False


class TestRerenderAfterCleanup:
    def test_export_reports_a_deleted_source_clearly(self, client: TestClient) -> None:
        job, source = make_job()
        clip = store.list_clips(job.id)[0]
        cleanup.remove_source_media(source)

        response = client.post(
            f"/api/clips/{clip.id}/export", json={"ratio": "9:16", "style": "bold_pop"}
        )

        assert response.status_code == 410
