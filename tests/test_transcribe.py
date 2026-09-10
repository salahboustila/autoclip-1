"""Transcription fallbacks.

Silero VAD can classify an entire track as non-speech — a music bed under the
voice, heavy compression — which used to fail the job with "No speech was
detected" on audio that plainly has speech. These tests pin the no-VAD retry
without loading a real Whisper model.
"""

from __future__ import annotations

from types import SimpleNamespace

import faster_whisper
import pytest
from autoclip.config import WhisperSettings
from autoclip.pipeline import transcribe as transcribe_module
from autoclip.pipeline.transcribe import TranscriptionError, transcribe

SPEECH = [(" hello", 0.0, 0.5), (" world.", 0.5, 1.0)]


def _segments(words: list[tuple[str, float, float]]) -> list[SimpleNamespace]:
    if not words:
        return []
    fake_words = [SimpleNamespace(word=w, start=s, end=e) for w, s, e in words]
    return [
        SimpleNamespace(
            words=fake_words,
            text="".join(w for w, _, _ in words),
            start=words[0][1],
            end=words[-1][2],
        )
    ]


@pytest.fixture
def fake_whisper(monkeypatch: pytest.MonkeyPatch):
    """Install a WhisperModel whose output depends on whether VAD is on.

    Returns a function that sets what each pass hears, and the list of
    ``vad_filter`` values the model was called with.
    """
    calls: list[bool] = []
    heard: dict[bool, list[tuple[str, float, float]]] = {True: [], False: []}

    class FakeModel:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def transcribe(self, audio: str, **kwargs):
            vad = kwargs["vad_filter"]
            calls.append(vad)
            info = SimpleNamespace(language="en", duration=1.0)
            return iter(_segments(heard[vad])), info

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    monkeypatch.setattr(transcribe_module, "resolve_compute", lambda settings: ("cpu", "int8"))

    def configure(*, with_vad, without_vad):
        heard[True] = with_vad
        heard[False] = without_vad
        return calls

    return configure


def test_unfiltered_pass_recovers_speech_vad_dropped(fake_whisper, tmp_path) -> None:
    calls = fake_whisper(with_vad=[], without_vad=SPEECH)

    result = transcribe(tmp_path / "audio.wav", WhisperSettings())

    assert [w.text for w in result.words] == ["hello", "world."]
    assert calls == [True, False]


def test_vad_pass_is_kept_when_it_finds_speech(fake_whisper, tmp_path) -> None:
    calls = fake_whisper(with_vad=SPEECH, without_vad=[])

    result = transcribe(tmp_path / "audio.wav", WhisperSettings())

    assert [w.text for w in result.words] == ["hello", "world."]
    assert calls == [True]


def test_silence_on_both_passes_still_fails(fake_whisper, tmp_path) -> None:
    calls = fake_whisper(with_vad=[], without_vad=[])

    with pytest.raises(TranscriptionError, match="No speech was detected"):
        transcribe(tmp_path / "audio.wav", WhisperSettings())

    assert calls == [True, False]
