"""Campaign freshness: history, the host's Shorts, headline quote, replay peaks."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from autoclip import app as app_module
from autoclip import campaigns
from autoclip.app import create_app
from autoclip.campaigns import freshness, history, youtube
from autoclip.db import store
from autoclip.db.models import Clip, Job, Source, new_id
from autoclip.pipeline.boundaries import Boundary
from autoclip.pipeline.transcript import Transcript, Word
from fastapi.testclient import TestClient

PRESET = campaigns.load_preset("jack_neel")
TEXT = (
    "How much money did you make last year? I made four million dollars from one deal "
    "and the government took half of it in taxes before I saw a cent."
)


def transcript(text: str = TEXT) -> Transcript:
    return Transcript(words=[Word(t, i * 0.4, i * 0.4 + 0.35) for i, t in enumerate(text.split())])


def boundary(t: Transcript, start_s: float = 0.0) -> Boundary:
    return Boundary(start_s=start_s, end_s=start_s + 30, start_word=0, end_word=len(t.words) - 1)


EPISODE_TITLE = '$120M CEO: "You Can Make $1M in a Week!" The 3 Hidden AI Gold Rushes'


def source(url: str | None = "https://www.youtube.com/watch?v=pBsT6v-ciO8", **kw) -> Source:
    return Source(
        id=new_id(), type="youtube", path="x.mp4", url=url, title=kw.get("title", EPISODE_TITLE)
    )


class TestYoutubeHelpers:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.youtube.com/watch?v=pBsT6v-ciO8",
            "https://youtu.be/pBsT6v-ciO8?t=30",
            "https://www.youtube.com/shorts/pBsT6v-ciO8",
            "https://m.youtube.com/watch?v=pBsT6v-ciO8&list=x",
        ],
    )
    def test_video_id(self, url: str) -> None:
        assert youtube.video_id(url) == "pBsT6v-ciO8"

    @pytest.mark.parametrize("url", [None, "", "https://example.com/watch?v=pBsT6v-ciO8", "nope"])
    def test_no_video_id(self, url) -> None:
        assert youtube.video_id(url) is None

    @pytest.mark.parametrize(
        ("title", "quote"),
        [
            (
                '$120M CEO: "You Can Make $1M in a Week!" The 3 Hidden',
                "You Can Make $1M in a Week!",
            ),
            ("Ex-CIA Spy: “His Body Was Switched!” The 7 People", "His Body Was Switched!"),
            ("(NEW!) Andrew Tate's Emergency Interview", ""),
        ],
    )
    def test_headline_quote(self, title: str, quote: str) -> None:
        assert freshness.headline_quote(title) == quote

    def test_lookups_are_cached(self, autoclip_home, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []

        def fake(url, **_kw):
            calls.append(url)
            return {"entries": [{"id": "pBsT6v-ciO8", "title": "Ep", "duration": 60}]}

        monkeypatch.setattr(youtube, "_extract", fake)
        first = youtube.channel_videos("https://yt/c", 20)
        second = youtube.channel_videos("https://yt/c", 20)
        assert first == second == [
            {"id": "pBsT6v-ciO8", "title": "Ep", "duration_s": 60,
             "url": "https://www.youtube.com/watch?v=pBsT6v-ciO8"}
        ]  # fmt: skip
        assert len(calls) == 1


class TestHistory:
    def test_record_entries_and_forget(self, autoclip_home) -> None:
        src = source()
        clip = Clip(id="c1", job_id="job-a", start_s=100, end_s=130, question_text="How?")
        history.record("jack_neel", src, clip)
        history.record("jack_neel", src, clip)  # re-export replaces, never duplicates

        (entry,) = history.entries("jack_neel", "yt:pBsT6v-ciO8")
        assert (entry["job_id"], entry["start_s"], entry["end_s"]) == ("job-a", 100, 130)

        history.forget_job("job-a")
        assert history.entries("jack_neel") == []

    def test_uploads_are_keyed_by_title(self) -> None:
        assert history.episode_key(source(url=None, title="My Ep!")) == "upload:my-ep"


class TestAssess:
    def test_a_moment_already_clipped_is_blocked(self) -> None:
        t = transcript()
        fresh = freshness.Freshness(previous=[{"job_id": "abcdef123", "start_s": 5, "end_s": 35}])
        penalty, notes, block = fresh.assess(t, boundary(t))
        assert block == "already clipped in job abcdef12"

    def test_a_different_moment_passes(self) -> None:
        t = transcript()
        fresh = freshness.Freshness(previous=[{"job_id": "a", "start_s": 300, "end_s": 330}])
        assert fresh.assess(t, boundary(t)) == (0, [], None)

    def test_a_moment_matching_the_hosts_short_is_blocked(self) -> None:
        t = transcript()
        fresh = freshness.Freshness(shorts=["He Made Four Million Dollars From One Deal"])
        _, _, block = fresh.assess(t, boundary(t))
        assert block and "Short" in block

    def test_generic_words_alone_never_match(self) -> None:
        # Seen on the acceptance run: a real @jackneel Short title that only
        # shares "make" and "millions" with an unrelated money clip.
        t = transcript()
        fresh = freshness.Freshness(shorts=["How to Make Millions With AI Videos"])
        assert fresh.assess(t, boundary(t))[2] is None

    def test_a_loosely_related_short_is_not_a_match(self) -> None:
        t = transcript()
        fresh = freshness.Freshness(shorts=["Why Taxes Are Theft According To Economists"])
        assert fresh.assess(t, boundary(t))[2] is None

    def test_the_headline_quote_is_penalised(self) -> None:
        t = transcript("You can make a million dollars in a week if you stop selling your time.")
        fresh = freshness.Freshness(headline="You Can Make $1M in a Week!", headline_penalty=20)
        penalty, notes, block = fresh.assess(t, boundary(t))
        assert (penalty, block) == (20, None)
        assert "headline" in notes[0]

    def test_most_replayed_peaks_are_penalised(self) -> None:
        peaks = freshness._peaks(
            [
                {"start": 0, "end": 10, "value": 0.2},
                {"start": 10, "end": 20, "value": 0.9},
                {"start": 20, "end": 30, "value": 0.7},
            ]
        )
        assert peaks == [(10, 30)]
        t = transcript()
        fresh = freshness.Freshness(peaks=peaks, most_replayed_penalty=15)
        assert fresh.assess(t, boundary(t))[0] == 15
        assert fresh.assess(t, boundary(t, start_s=200))[0] == 0


class TestLoad:
    def test_gathers_signals_and_skips_its_own_job(
        self, autoclip_home, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = source()
        history.record("jack_neel", src, Clip(id="c1", job_id="old", start_s=1, end_s=30))
        history.record("jack_neel", src, Clip(id="c2", job_id="this", start_s=40, end_s=70))

        def fake(url, *, flat, **_kw):
            if flat:
                return {"entries": [{"id": "aaaaaaaaaaa", "title": "Short one"}]}
            return {"heatmap": [{"start_time": 0, "end_time": 5, "value": 1.0}]}

        monkeypatch.setattr(youtube, "_extract", fake)
        fresh = freshness.load(PRESET, src, "this")

        assert [e["job_id"] for e in fresh.previous] == ["old"]
        assert fresh.shorts == ["Short one"]
        assert fresh.headline == "You Can Make $1M in a Week!"
        assert fresh.peaks == [(0, 5)]
        assert fresh.warnings == []

    def test_failed_lookups_become_warnings(self, autoclip_home) -> None:
        # conftest makes every lookup fail, as an offline pod would.
        fresh = freshness.load(PRESET, source(), "job")
        assert len(fresh.warnings) == 2
        assert fresh.shorts == [] and fresh.peaks == []

    def test_uploads_skip_the_heatmap(self, autoclip_home) -> None:
        fresh = freshness.load(PRESET, source(url=None), "job")
        assert not any("most-replayed" in w for w in fresh.warnings)


class TestDeleteForgets:
    @pytest.fixture
    def client(self, autoclip_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
        monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
        with TestClient(create_app()) as test_client:
            yield test_client

    def test_deleting_a_job_forgets_its_clips(self, client: TestClient) -> None:
        src = store.create_source(source())
        job = store.create_job(Job(id=new_id(), source_id=src.id, status="done"))
        history.record("jack_neel", src, Clip(id="c", job_id=job.id, start_s=0, end_s=30))

        assert client.delete(f"/api/jobs/{job.id}").status_code == 204
        assert history.entries("jack_neel") == []


class TestWithTheQuestionFirstPolicy:
    def test_a_second_job_never_recuts_the_same_moment(self) -> None:
        from autoclip.campaigns.question_first import QuestionFirstPolicy
        from autoclip.pipeline import highlights
        from autoclip.providers import DetectionConfig
        from autoclip.providers.base import ClipCandidate

        answer = "I made four million dollars from one deal and it took ten years. " * 6
        t = transcript("How much money did you make last year? " + answer)
        fresh = freshness.Freshness(previous=[{"job_id": "earlier1", "start_s": 0, "end_s": 30}])
        policy = QuestionFirstPolicy(PRESET, freshness=fresh)
        config = DetectionConfig(
            min_duration_s=25, max_duration_s=50, viral_hook=True, clip_policy=policy
        )
        candidate = ClipCandidate(
            start_word_index=0, end_word_index=len(t.words) - 1, score=90,
            question_hook=9, controversy=9, number_stat=10, clarity=9,
        )  # fmt: skip

        assert highlights.build_clips(t, [candidate], config, job_id="j", min_score=70) == []
        assert policy.rejections[-1]["reason"] == "already clipped in job earlier1"
