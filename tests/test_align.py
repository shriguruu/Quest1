"""Tests for dialogue_locator.stages.align.

Ground truth for a speech onset is hard to come by -- nobody labels real
media to the millisecond -- so we construct it.  A TTS clip is trimmed to
its first audible sample and spliced into digital silence at a known
offset, which makes the true onset exact *by construction*: if speech
begins at sample ``2.500 * 16000``, it begins at 2.500s, and any
disagreement belongs to the aligner.

**Why the fixture says "topic" and not "mind".**  Measuring across initial
phonemes shows the aligner's error is always positive and tracks how
abruptly the sound starts:

    plosive /k/    +7ms      fricative /s/   +66ms
    plosive /t/   +23ms      nasal /m/       +68ms
    plosive /p/   +24ms

A plosive opens with a burst, so an energy threshold and a CTC model agree
on where it begins.  A nasal or fricative ramps up, so a 2%-of-peak
threshold fires early on the ramp while the model waits for enough
acoustic evidence.  That gap is a disagreement about what "onset" *means*
for a gradual sound, not aligner error, and a fixture built on one would
be measuring the threshold rather than the thing under test.  So the tight
assertion uses a plosive, and the continuant behaviour is documented in
its own test rather than tuned away.
"""

from __future__ import annotations

import array
import shutil
import subprocess
import wave
from fractions import Fraction
from pathlib import Path

import pytest

from dialogue_locator.config import Settings
from dialogue_locator.models import (
    Candidate,
    MatchTier,
    MediaInfo,
    Transcript,
    Word,
)
from dialogue_locator.stages.align import (
    FORCED_UNCERTAINTY_S,
    WHISPER_UNCERTAINTY_S,
    _prepare_tokens,
    _read_window,
    align,
)

pytest.importorskip("torch", reason="forced alignment requires torch")
pytest.importorskip("torchaudio", reason="forced alignment requires torchaudio")

SAMPLE_RATE = 16000
TRUE_ONSET_S = 2.500
TOLERANCE_S = 0.030

# Plosive-initial: sharp burst, so ground truth is unambiguous.
PLOSIVE_TEXT = "topic centres on careful timing"
# Nasal-initial: gradual ramp, used only to document the effect.
NASAL_TEXT = "mind rebels against dull routine"

# Whisper's stand-in estimate, deliberately early by a realistic DTW error.
WHISPER_ONSET_S = 2.320

_HAVE_TOOLS = (
    shutil.which("powershell") is not None and shutil.which("ffmpeg") is not None
)
requires_tts = pytest.mark.skipif(
    not _HAVE_TOOLS,
    reason="needs Windows SAPI (powershell) and ffmpeg to synthesise speech",
)


# ---------------------------------------------------------------------------
# Fixture construction
# ---------------------------------------------------------------------------


def _synthesize(text: str, out_path: Path) -> Path:
    """Speak *text* to a WAV via Windows SAPI."""
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SetOutputToWaveFile('{out_path.as_posix()}'); "
        f"$s.Speak('{text}'); $s.Dispose()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        check=True,
        capture_output=True,
    )
    return out_path


def _to_16k_mono(src: Path, dest: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src),
         "-ar", str(SAMPLE_RATE), "-ac", "1", str(dest)],
        check=True,
        capture_output=True,
    )
    return dest


def _read_samples(path: Path) -> array.array:
    with wave.open(str(path), "rb") as wf:
        samples = array.array("h")
        samples.frombytes(wf.readframes(wf.getnframes()))
    return samples


def _write_samples(path: Path, samples: array.array) -> Path:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())
    return path


def _trim_silence(samples: array.array, frac_of_peak: float = 0.02) -> array.array:
    """Drop leading and trailing silence, defining onset by energy."""
    peak = max(abs(v) for v in samples)
    threshold = peak * frac_of_peak
    first = next(i for i, v in enumerate(samples) if abs(v) > threshold)
    last = len(samples) - next(
        i for i, v in enumerate(reversed(samples)) if abs(v) > threshold
    )
    return samples[first:last]


def _build_spliced(text: str, workdir: Path, tag: str) -> tuple[Path, float]:
    """Speech beginning at exactly TRUE_ONSET_S. Returns (path, duration)."""
    raw = _synthesize(text, workdir / f"raw_{tag}.wav")
    resampled = _to_16k_mono(raw, workdir / f"16k_{tag}.wav")
    speech = _trim_silence(_read_samples(resampled))

    lead = array.array("h", [0] * int(TRUE_ONSET_S * SAMPLE_RATE))
    tail = array.array("h", [0] * int(1.5 * SAMPLE_RATE))
    combined = lead + speech + tail

    path = _write_samples(workdir / f"spliced_{tag}.wav", combined)
    return path, len(combined) / SAMPLE_RATE


@pytest.fixture(scope="session")
def workdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("align")


@pytest.fixture(scope="session")
def plosive_clip(workdir: Path) -> tuple[Path, float]:
    return _build_spliced(PLOSIVE_TEXT, workdir, "plosive")


@pytest.fixture(scope="session")
def nasal_clip(workdir: Path) -> tuple[Path, float]:
    return _build_spliced(NASAL_TEXT, workdir, "nasal")


@pytest.fixture(scope="session")
def silent_clip(workdir: Path) -> Path:
    return _write_samples(
        workdir / "silent.wav", array.array("h", [0] * (SAMPLE_RATE * 6))
    )


# ---------------------------------------------------------------------------
# Small builders
# ---------------------------------------------------------------------------


def make_media(duration_s: float) -> MediaInfo:
    return MediaInfo(
        url="test://synthetic",
        video_path=Path("synthetic.mp4"),
        audio_path=None,
        duration_s=duration_s,
        fps=Fraction(24000, 1001),
        is_vfr=False,
        total_frames=int(duration_s * 24),
        width=960,
        height=720,
        video_start_time_s=0.0,
        audio_start_time_s=0.0,
        av_offset_s=0.0,
        video_sha256="synthetic",
    )


def make_transcript(words: list[str], first_start_s: float, step: float = 0.30):
    return Transcript(
        words=[
            Word(
                text=w,
                start_s=first_start_s + i * step,
                end_s=first_start_s + i * step + 0.28,
                probability=0.9,
                segment_id=0,
            )
            for i, w in enumerate(words)
        ],
        language="en",
        model_name="test",
        source="asr",
    )


def make_candidate(n_words: int, coarse_start_s: float) -> Candidate:
    return Candidate(
        word_index_start=0,
        word_index_end=n_words - 1,
        matched_text="synthetic",
        score=95.0,
        tier=MatchTier.FUZZY,
        coarse_start_s=coarse_start_s,
    )


def run_align(clip: Path, duration_s: float, text: str, whisper_onset: float):
    words = text.split()
    return align(
        clip,
        make_media(duration_s),
        make_candidate(len(words), whisper_onset),
        make_transcript(words, whisper_onset),
        Settings(),
    )


# ---------------------------------------------------------------------------
# The headline requirement
# ---------------------------------------------------------------------------


@requires_tts
class TestForcedAlignmentAccuracy:
    def test_onset_within_30ms_of_truth(
        self, plosive_clip: tuple[Path, float]
    ) -> None:
        """Speech starts at exactly 2.500s; the aligner must find it."""
        clip, duration = plosive_clip
        result = run_align(clip, duration, PLOSIVE_TEXT, WHISPER_ONSET_S)

        assert result.method == "forced", (
            f"expected forced alignment, got fallback: {result.warnings}"
        )
        error_s = abs(result.onset_s - TRUE_ONSET_S)
        assert error_s <= TOLERANCE_S, (
            f"onset {result.onset_s:.4f}s is {error_s * 1000:.1f}ms from the "
            f"true {TRUE_ONSET_S}s (tolerance {TOLERANCE_S * 1000:.0f}ms)"
        )

    def test_reports_20ms_uncertainty_and_vad_confirmation(
        self, plosive_clip: tuple[Path, float]
    ) -> None:
        clip, duration = plosive_clip
        result = run_align(clip, duration, PLOSIVE_TEXT, WHISPER_ONSET_S)

        assert result.method == "forced"
        assert result.uncertainty_s == FORCED_UNCERTAINTY_S
        assert result.vad_confirmed is True
        assert result.warnings == []

    def test_beats_the_whisper_estimate(
        self, plosive_clip: tuple[Path, float]
    ) -> None:
        """The entire justification for this stage."""
        clip, duration = plosive_clip
        result = run_align(clip, duration, PLOSIVE_TEXT, WHISPER_ONSET_S)

        whisper_error = abs(WHISPER_ONSET_S - TRUE_ONSET_S)
        aligned_error = abs(result.onset_s - TRUE_ONSET_S)
        assert aligned_error < whisper_error, (
            f"alignment ({aligned_error * 1000:.1f}ms) did not improve on "
            f"whisper ({whisper_error * 1000:.1f}ms)"
        )

    def test_per_word_timings_are_monotonic(
        self, plosive_clip: tuple[Path, float]
    ) -> None:
        clip, duration = plosive_clip
        result = run_align(clip, duration, PLOSIVE_TEXT, WHISPER_ONSET_S)

        assert len(result.per_word) == len(PLOSIVE_TEXT.split())
        starts = [w.start_s for w in result.per_word]
        assert starts == sorted(starts)
        assert result.onset_s == pytest.approx(starts[0], abs=1e-6)
        assert all(w.end_s >= w.start_s for w in result.per_word)


@requires_tts
class TestOnsetDependsOnInitialPhoneme:
    """Document, rather than hide, the continuant-onset effect."""

    def test_nasal_onset_lands_late_but_bounded(
        self, nasal_clip: tuple[Path, float]
    ) -> None:
        clip, duration = nasal_clip
        result = run_align(clip, duration, NASAL_TEXT, WHISPER_ONSET_S)

        assert result.method == "forced"
        error_s = result.onset_s - TRUE_ONSET_S
        # Late, never early: the model waits for acoustic evidence.
        assert error_s > 0
        # Still far better than Whisper's +/-300ms, and under 2 video frames.
        assert error_s < 0.100, f"nasal onset drifted {error_s * 1000:.1f}ms"

    def test_still_beats_whisper_on_a_nasal(
        self, nasal_clip: tuple[Path, float]
    ) -> None:
        clip, duration = nasal_clip
        result = run_align(clip, duration, NASAL_TEXT, WHISPER_ONSET_S)

        assert abs(result.onset_s - TRUE_ONSET_S) < abs(
            WHISPER_ONSET_S - TRUE_ONSET_S
        )


# ---------------------------------------------------------------------------
# Sanity checks must demote, never lie
# ---------------------------------------------------------------------------


class TestFallbackBehaviour:
    def test_silent_audio_falls_back(self, silent_clip: Path) -> None:
        """Nothing to align against at all."""
        result = run_align(silent_clip, 6.0, PLOSIVE_TEXT, 2.5)

        assert result.method == "whisper_fallback"
        assert result.uncertainty_s == WHISPER_UNCERTAINTY_S
        assert result.warnings, "a fallback must explain itself"

    def test_fallback_preserves_whisper_timings(self, silent_clip: Path) -> None:
        words = PLOSIVE_TEXT.split()
        transcript = make_transcript(words, 2.5)
        result = align(
            silent_clip,
            make_media(6.0),
            make_candidate(len(words), 2.5),
            transcript,
            Settings(),
        )
        assert [w.start_s for w in result.per_word] == [
            w.start_s for w in transcript.words
        ]

    @requires_tts
    def test_wrong_region_falls_back(
        self, plosive_clip: tuple[Path, float]
    ) -> None:
        """Whisper points at a stretch of silence far from the speech."""
        clip, duration = plosive_clip
        result = run_align(clip, duration, PLOSIVE_TEXT, whisper_onset=6.4)

        assert result.method == "whisper_fallback"
        assert result.warnings

    def test_unalignable_text_falls_back(self, silent_clip: Path) -> None:
        """Punctuation-only words leave nothing to align."""
        transcript = make_transcript(["...", "---"], 2.5)
        result = align(
            silent_clip,
            make_media(6.0),
            make_candidate(2, 2.5),
            transcript,
            Settings(),
        )
        assert result.method == "whisper_fallback"
        assert "alignable" in " ".join(result.warnings)

    def test_empty_candidate_span_raises(self, silent_clip: Path) -> None:
        from dialogue_locator.errors import AlignmentError

        transcript = make_transcript(PLOSIVE_TEXT.split(), 2.5)
        bad = Candidate(
            word_index_start=99,
            word_index_end=110,
            matched_text="nope",
            score=90.0,
            tier=MatchTier.FUZZY,
            coarse_start_s=2.5,
        )
        with pytest.raises(AlignmentError, match="spans no words"):
            align(silent_clip, make_media(6.0), bad, transcript, Settings())


# ---------------------------------------------------------------------------
# Token preparation
# ---------------------------------------------------------------------------


class TestPrepareTokens:
    def test_uppercases_and_strips_punctuation(self) -> None:
        words = make_transcript(["My", "mind,", "rebels!"], 0.0).words
        indices, tokens = _prepare_tokens(words)
        assert tokens == ["MY", "MIND", "REBELS"]
        assert indices == [0, 1, 2]

    def test_drops_punctuation_only_words_keeping_index_map(self) -> None:
        """A dropped word must not desynchronise the mapping back."""
        words = make_transcript(["mind", "--", "rebels"], 0.0).words
        indices, tokens = _prepare_tokens(words)
        assert tokens == ["MIND", "REBELS"]
        assert indices == [0, 2], "index 1 must be skipped, not renumbered"

    def test_digits_are_spelled_out(self) -> None:
        """wav2vec2's label set has no digits, so they must become letters."""
        words = make_transcript(["221", "Baker"], 0.0).words
        _, tokens = _prepare_tokens(words)
        assert all(c.isalpha() or c == "'" for t in tokens for c in t)
        assert "TWO" in tokens[0]

    def test_output_is_within_the_label_alphabet(self) -> None:
        words = make_transcript(["Don't", "stop—", "3 things"], 0.0).words
        _, tokens = _prepare_tokens(words)
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ'")
        assert all(c in allowed for t in tokens for c in t)


# ---------------------------------------------------------------------------
# Window reading
# ---------------------------------------------------------------------------


class TestReadWindow:
    def test_reads_expected_sample_count(self, silent_clip: Path) -> None:
        samples, rate = _read_window(silent_clip, 1.0, 3.0)
        assert rate == SAMPLE_RATE
        assert len(samples) == pytest.approx(2 * SAMPLE_RATE, abs=2)

    def test_clamps_to_file_bounds(self, silent_clip: Path) -> None:
        samples, _ = _read_window(silent_clip, 0.0, 9999.0)
        assert len(samples) > 0

    def test_empty_window_raises(self, silent_clip: Path) -> None:
        from dialogue_locator.errors import AlignmentError

        with pytest.raises(AlignmentError, match="Empty alignment window"):
            _read_window(silent_clip, 3.0, 3.0)
