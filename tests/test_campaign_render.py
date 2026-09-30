"""Campaign render: the Podcast Hook layout, and nothing burned in but title and captions."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from autoclip import campaigns, config, paths
from autoclip.campaigns import report
from autoclip.db import store
from autoclip.db.models import Clip, Job, Source, new_id
from autoclip.pipeline import export, ffmpeg, layouts
from autoclip.pipeline.runner import PipelineRunner
from autoclip.pipeline.transcript import Transcript, Word
from PIL import Image

SPOKEN = (
    "How much money did you make last year? I made four million dollars from one deal "
    "and nobody believed me."
)


CAPTION = "He made four million from one deal.\nCould you?\n\n@jackhneel\n@jackneel\n#jackneelpod"


def campaign_settings() -> config.Settings:
    settings = config.Settings()
    settings.campaign.enabled = True
    settings.cleanup.auto_delete_sources = False
    return campaigns.apply(settings)


def words(pace: float = 0.3) -> list[Word]:
    return [Word(t, i * pace, i * pace + pace * 0.9) for i, t in enumerate(SPOKEN.split())]


class TestLayoutChoice:
    def test_campaign_jobs_render_the_podcast_hook_layout(self) -> None:
        options = layouts.podcast_options(campaign_settings(), 1920, 1080)
        assert options is not None
        assert options.layout.video_box == (0, 656, 1080, 608)
        assert options.title_style == "podcast_anton"
        assert options.highlight_title_keyword is True

    def test_default_jobs_do_not(self) -> None:
        assert layouts.podcast_options(config.Settings(), 1920, 1080) is None


class TestNothingElseBurnedIn:
    def test_the_ffmpeg_command_has_only_video_title_and_captions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict = {}

        def fake_run(args, **kwargs):
            captured["args"] = list(args)
            Path(args[-1]).write_bytes(b"x")

        monkeypatch.setattr(ffmpeg, "run", fake_run)
        settings = campaign_settings()
        options = layouts.podcast_options(settings, 1920, 1080)
        request = export.ExportRequest(
            source=tmp_path / "src.mp4",
            destination=tmp_path / "out.mp4",
            start_s=0.0,
            end_s=8.0,
            crop_path=layouts.podcast_crop_path(options, 1920, 1080, 8.0),
            words=words(),
            style=export.captions_module.get_style(settings.export.caption_style),
            hook_title="How Much Money Did You Make? 🤔",
            hook_title_settings=settings.hook_title,
            podcast=options,
        )

        export.export_clip(request, work_dir=tmp_path / "work", settings=settings.export)

        args = captured["args"]
        inputs = [args[i + 1] for i, a in enumerate(args) if a == "-i"]
        assert inputs == [str(tmp_path / "src.mp4"), str(tmp_path / "work/out/hook_title.png")]
        graph = args[args.index("-filter_complex") + 1]
        for forbidden in ("drawtext", "movie=", "logo", "watermark"):
            assert forbidden not in graph
        assert graph.count("overlay=") == 1  # the hook title, nothing else
        ass = (tmp_path / "work/out/captions.ass").read_text(encoding="utf-8")
        for handle in ("@jackhneel", "@jackneel", "#jackneelpod"):
            assert handle not in ass


@pytest.mark.slow
def test_campaign_export_end_to_end(initialised_db, tmp_path: Path) -> None:
    """Synthetic 16:9 episode → campaign clip, caption file and report."""
    video = tmp_path / "episode.mp4"
    subprocess.run(
        [ffmpeg.ffmpeg_path(), "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=1920x1080:rate=30:duration=8", "-f", "lavfi", "-i",
         "sine=frequency=330:duration=8", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(video)],
        check=True, capture_output=True,
    )  # fmt: skip
    settings = campaign_settings()
    source = store.create_source(
        Source(id=new_id(), type="upload", path=str(video), width=1920, height=1080, duration_s=8)
    )
    job = store.create_job(Job(id=new_id(), source_id=source.id))
    transcript = Transcript(words=words())
    clip = Clip(
        id=new_id(), job_id=job.id, rank=1, start_s=0.0, end_s=6.0, start_word=0,
        end_word=len(transcript.words) - 1, score=84, topic="Money", title="Income",
        question_text="How much money did you make last year?",
        hook_title="How Much Money Did You Make? 🤔",
        post_caption=CAPTION,
    )  # fmt: skip
    store.replace_clips(job.id, [clip])
    preset = campaigns.rules_of(settings)
    report.write(job.id, preset, [clip], host_check="no_speaker_labels")

    PipelineRunner(job, source, settings=settings)._stage_export([clip], transcript, {})

    out = paths.exports_dir() / job.id
    mp4 = out / "clip_01_income_9x16.mp4"
    assert ffmpeg.probe(mp4).width == 1080
    caption = (out / "clip_01_caption.txt").read_text(encoding="utf-8")
    assert caption.splitlines()[-3:] == ["@jackhneel", "@jackneel", "#jackneelpod"]
    data = json.loads((out / "selected_clips.json").read_text(encoding="utf-8"))
    assert data["clips"][0]["video_file"] == mp4.name
    assert data["clips"][0]["caption_file"] == "clip_01_caption.txt"

    frame = tmp_path / "frame.png"
    subprocess.run(
        [ffmpeg.ffmpeg_path(), "-loglevel", "error", "-y", "-ss", "3.5", "-i", str(mp4),
         "-frames:v", "1", str(frame)],
        check=True, capture_output=True,
    )  # fmt: skip
    with Image.open(frame) as image:
        rgb = image.convert("RGB")
        # Black top strip and bottom strip: no logo or handle anywhere.
        top = [rgb.getpixel((x, y)) for y in range(0, 150, 5) for x in range(0, 1080, 5)]
        bottom = [rgb.getpixel((x, y)) for y in range(1760, 1920, 5) for x in range(0, 1080, 5)]
        assert max(max(p) for p in top) < 40
        assert max(max(p) for p in bottom) < 40
