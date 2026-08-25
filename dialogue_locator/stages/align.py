"""Alignment stage: turn a coarse match into a frame-accurate onset.

Whisper does not measure when a word begins.  Its word timestamps are
derived from cross-attention DTW -- a byproduct of where the decoder was
attending -- and land within roughly 100-300ms of the truth.  At 24fps
that is 3-8 frames of slop, which is fatal for a tool whose entire output
is a single frame number.

Forced alignment solves a strictly easier problem.  Recognition asks "what
words are in this audio?"; alignment already knows the words and asks only
"where does each one sit?"  Constraining the search to a known token
sequence collapses the ambiguity, and a CTC model resolves onsets to one
emission frame -- measured at 20.03ms on this pipeline, comfortably under
a single video frame.

**Why torchaudio rather than whisperx.**  ``whisperx.align`` wraps roughly
this same wav2vec2 CTC procedure, but it arrives with a large transitive
dependency tree and pins ``faster-whisper`` versions, which risks
destabilising a transcription stage that already works.  ``torchaudio``
exposes :func:`torchaudio.functional.forced_align` directly -- a
Viterbi-over-CTC-emissions primitive and nothing else -- so the whole
stage is about thirty lines of real logic with no hidden behaviour and no
version coupling.  When a dependency's only advantage is convenience, and
the thing it wraps is this small, wrapping it ourselves is cheaper than
owning the dependency.

**Aligning the transcript, not the query.**  CTC forced alignment cannot
refuse.  Hand it a token sequence that is not in the audio and it will
still produce a path, smearing those tokens across whatever sound is
present and reporting confident nonsense.  The user's query is only
*approximately* what was said -- that is the entire premise of the fuzzy
matcher upstream -- so we align the words the ASR actually heard.  The
query decides *which span*; the transcript decides *what text* is aligned
inside it.

**Trusting the result.**  Three independent checks guard the output, and
any failure demotes the answer back to Whisper's timestamp with a stated
reason.  A known-imprecise answer is far more useful than a precise-looking
wrong one, so nothing here fails silently.
"""

from __future__ import annotations

import array
import gc
import wave
from dataclasses import dataclass
from pathlib import Path

import structlog

from dialogue_locator.config import Settings
from dialogue_locator.errors import AlignmentError
from dialogue_locator.models import (
    AlignmentResult,
    Candidate,
    MediaInfo,
    Transcript,
    Word,
)
from dialogue_locator.stages.match import normalize

logger = structlog.get_logger()

# One CTC emission frame is ~20ms for wav2vec2 base at 16kHz.
FORCED_UNCERTAINTY_S = 0.02
# What we claim when we fall back to Whisper's own timestamps.
WHISPER_UNCERTAINTY_S = 0.3

# An aligned onset further than this from Whisper's estimate means the two
# models disagree about *which* utterance this is, not merely about its
# edge. That is a different kind of error, and not one to paper over.
MAX_DRIFT_S = 2.0
# Mean CTC posterior below this indicates the token sequence does not
# really fit the audio.
MIN_CTC_SCORE = 0.30

# wav2vec2's LibriSpeech label set is uppercase A-Z plus an apostrophe.
_ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ'")


@dataclass(frozen=True)
class _AlignedWord:
    """One word located by the aligner, in absolute media time."""

    word_index: int
    start_s: float
    end_s: float
    score: float


# ---------------------------------------------------------------------------
# Audio access
# ---------------------------------------------------------------------------


def _read_window(
    audio_path: Path, start_s: float, end_s: float
) -> tuple[list[float], int]:
    """Return mono float samples in ``[start_s, end_s)`` plus the rate."""
    try:
        with wave.open(str(audio_path), "rb") as wf:
            rate = wf.getframerate()
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            total = wf.getnframes()

            if width != 2:
                raise AlignmentError(
                    f"Expected 16-bit PCM audio, got {width * 8}-bit"
                )

            first = max(0, int(start_s * rate))
            last = min(total, int(end_s * rate))
            if last <= first:
                raise AlignmentError(
                    f"Empty alignment window [{start_s:.3f}, {end_s:.3f})"
                )
            wf.setpos(first)
            raw = wf.readframes(last - first)
    except wave.Error as e:
        raise AlignmentError(f"Could not read audio {audio_path}: {e}") from e
    except OSError as e:
        raise AlignmentError(f"Could not open audio {audio_path}: {e}") from e

    samples = array.array("h")
    samples.frombytes(raw)
    if channels > 1:
        samples = array.array("h", samples[::channels])

    return [s / 32768.0 for s in samples], rate


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------


def _prepare_tokens(words: list[Word]) -> tuple[list[int], list[str]]:
    """Map transcript words onto wav2vec2-compatible uppercase tokens.

    Returns the indices (into *words*) that survived, alongside their token
    strings.  Words that reduce to nothing -- pure punctuation -- are
    dropped here rather than silently desynchronising the mapping back from
    alignment output to transcript words.
    """
    indices: list[int] = []
    tokens: list[str] = []

    for i, word in enumerate(words):
        # normalize() already expands contractions, spells out digits (the
        # label set has none) and strips punctuation.
        cleaned = "".join(
            c for c in normalize(word.text).upper() if c in _ALLOWED
        )
        if not cleaned:
            continue
        indices.append(i)
        tokens.append(cleaned)

    return indices, tokens


# ---------------------------------------------------------------------------
# Core forced alignment
# ---------------------------------------------------------------------------


def _run_forced_alignment(
    samples: list[float],
    rate: int,
    window_start_s: float,
    tokens: list[str],
) -> tuple[list[tuple[float, float, float]], float]:
    """Align *tokens* against *samples*; return per-word spans and mean score.

    Spans are ``(start_s, end_s, score)`` in absolute media time, one per
    entry of *tokens*.
    """
    try:
        import torch
        import torchaudio
        import torchaudio.functional as AF
    except ImportError as e:
        raise AlignmentError(
            "Forced alignment needs torch and torchaudio. Run: uv sync"
        ) from e

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    if rate != bundle.sample_rate:
        raise AlignmentError(
            f"Audio is {rate}Hz but the aligner expects {bundle.sample_rate}Hz"
        )

    model = None
    try:
        model = bundle.get_model()
        model.eval()

        labels = bundle.get_labels()
        dictionary = {label: i for i, label in enumerate(labels)}
        separator = dictionary["|"]

        transcript_chars = "|".join(tokens)
        unknown = {c for c in transcript_chars if c not in dictionary}
        if unknown:
            raise AlignmentError(
                f"Characters outside the aligner's alphabet: {sorted(unknown)}"
            )

        waveform = torch.tensor([samples], dtype=torch.float32)

        with torch.inference_mode():
            emission, _ = model(waveform)
            log_probs = torch.log_softmax(emission, dim=-1)

            targets = torch.tensor(
                [[dictionary[c] for c in transcript_chars]], dtype=torch.int32
            )
            aligned, scores = AF.forced_align(log_probs, targets, blank=0)
            token_spans = AF.merge_tokens(aligned[0], scores[0].exp())

        # Seconds represented by one emission frame.
        seconds_per_frame = waveform.shape[1] / log_probs.shape[1] / rate

        # Regroup token spans into words on the separator token.
        grouped: list[list] = []
        current: list = []
        for span in token_spans:
            if span.token == separator:
                if current:
                    grouped.append(current)
                    current = []
            else:
                current.append(span)
        if current:
            grouped.append(current)

        if len(grouped) != len(tokens):
            raise AlignmentError(
                f"Aligner returned {len(grouped)} words for {len(tokens)} "
                f"expected; alignment is not trustworthy"
            )

        spans: list[tuple[float, float, float]] = []
        total_weight = 0.0
        total_score = 0.0
        for group in grouped:
            start_s = window_start_s + group[0].start * seconds_per_frame
            end_s = window_start_s + group[-1].end * seconds_per_frame
            weight = sum(s.end - s.start for s in group) or 1
            score = sum(s.score * (s.end - s.start) for s in group) / weight
            spans.append((start_s, end_s, float(score)))
            total_weight += weight
            total_score += float(score) * weight

        mean_score = total_score / total_weight if total_weight else 0.0
        return spans, mean_score

    finally:
        # wav2vec2 is the only model resident at this point; drop it before
        # returning so nothing accumulates across a batch of queries.
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Voice-activity confirmation
# ---------------------------------------------------------------------------


def _onset_in_speech(
    samples: list[float], rate: int, window_start_s: float, onset_s: float
) -> bool:
    """True when *onset_s* falls inside a Silero-detected speech region.

    A word onset that lands in silence means the alignment slid off the
    utterance, which is precisely the failure a confident-looking wrong
    answer would hide.
    """
    try:
        import numpy as np
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError:  # pragma: no cover - bundled with faster-whisper
        logger.warning("Silero VAD unavailable; skipping onset confirmation")
        return True

    try:
        audio = np.asarray(samples, dtype=np.float32)
        regions = get_speech_timestamps(
            audio,
            VadOptions(min_speech_duration_ms=100, min_silence_duration_ms=100),
            sampling_rate=rate,
        )
    except Exception as e:
        logger.warning("Silero VAD failed; skipping confirmation", error=str(e))
        return True

    relative = onset_s - window_start_s
    # A small tolerance: Silero trims low-energy consonant onsets, so an
    # alignment landing just before a detected region is still correct.
    tolerance_s = 0.12
    for region in regions:
        start = region["start"] / rate
        end = region["end"] / rate
        if start - tolerance_s <= relative <= end:
            return True
    return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def align(
    audio_path: Path,
    media: MediaInfo,
    candidate: Candidate,
    transcript: Transcript,
    settings: Settings,
) -> AlignmentResult:
    """Locate *candidate*'s first word precisely within *audio_path*.

    Falls back to Whisper's own timestamps, with an explanatory warning,
    whenever forced alignment cannot be trusted.  Never raises for an
    alignment failure -- only for a genuinely unusable input.
    """
    matched = transcript.words[
        candidate.word_index_start : candidate.word_index_end + 1
    ]
    if not matched:
        raise AlignmentError(
            f"Candidate spans no words "
            f"({candidate.word_index_start}..{candidate.word_index_end})"
        )

    whisper_onset_s = matched[0].start_s
    whisper_offset_s = matched[-1].end_s

    def _fallback(reason: str, vad_confirmed: bool = False) -> AlignmentResult:
        logger.warning("Falling back to Whisper timestamps", reason=reason)
        return AlignmentResult(
            onset_s=max(0.0, whisper_onset_s),
            offset_s=max(whisper_offset_s, whisper_onset_s),
            method="whisper_fallback",
            uncertainty_s=WHISPER_UNCERTAINTY_S,
            vad_confirmed=vad_confirmed,
            per_word=matched,
            warnings=[f"Forced alignment rejected: {reason}"],
        )

    indices, tokens = _prepare_tokens(matched)
    if not tokens:
        return _fallback("matched text contains no alignable characters")

    # Two attempts: the configured window, then double-width if the first
    # attempt's onset landed in silence (the usual cause is an utterance
    # that starts outside the window entirely).
    attempts = (settings.align_window_s, settings.align_window_s * 2)
    last_reason = "unknown"

    for attempt, pad_s in enumerate(attempts, start=1):
        start_s = max(0.0, whisper_onset_s - pad_s)
        end_s = min(media.duration_s, whisper_offset_s + pad_s)

        try:
            samples, rate = _read_window(audio_path, start_s, end_s)
            spans, mean_score = _run_forced_alignment(
                samples, rate, start_s, tokens
            )
        except AlignmentError as e:
            last_reason = str(e)
            logger.warning("Alignment attempt failed", attempt=attempt, error=str(e))
            continue

        onset_s, _, _ = spans[0]
        drift_s = abs(onset_s - whisper_onset_s)

        logger.info(
            "Forced alignment attempt",
            attempt=attempt,
            window=f"[{start_s:.2f}, {end_s:.2f}]",
            onset_s=round(onset_s, 3),
            whisper_onset_s=round(whisper_onset_s, 3),
            drift_ms=round(drift_s * 1000, 1),
            mean_score=round(mean_score, 3),
        )

        # --- Check 1: does the path actually fit the audio? ---
        if mean_score < MIN_CTC_SCORE:
            return _fallback(
                f"CTC path likelihood {mean_score:.2f} below "
                f"{MIN_CTC_SCORE:.2f}; the text may not match this audio"
            )

        # --- Check 2: does it agree with Whisper about which utterance? ---
        if drift_s > MAX_DRIFT_S:
            return _fallback(
                f"aligned onset is {drift_s:.2f}s from Whisper's estimate "
                f"(limit {MAX_DRIFT_S:.1f}s); the two disagree about which "
                f"utterance this is"
            )

        # --- Check 3: is the onset inside actual speech? ---
        if _onset_in_speech(samples, rate, start_s, onset_s):
            per_word = _rebuild_words(matched, indices, spans)
            logger.info(
                "Forced alignment accepted",
                onset_s=round(onset_s, 3),
                correction_ms=round((onset_s - whisper_onset_s) * 1000, 1),
            )
            return AlignmentResult(
                onset_s=max(0.0, onset_s),
                offset_s=max(spans[-1][1], onset_s),
                method="forced",
                uncertainty_s=FORCED_UNCERTAINTY_S,
                vad_confirmed=True,
                per_word=per_word,
                warnings=[],
            )

        last_reason = "aligned onset landed inside detected silence"
        logger.warning(
            "Onset landed in silence",
            attempt=attempt,
            onset_s=round(onset_s, 3),
            retrying=attempt < len(attempts),
        )

    return _fallback(last_reason, vad_confirmed=False)


def _rebuild_words(
    matched: list[Word],
    indices: list[int],
    spans: list[tuple[float, float, float]],
) -> list[Word]:
    """Return *matched* with aligned timings substituted in.

    Words that were dropped during token preparation keep their original
    Whisper timings, so the returned list always matches the candidate's
    span one-for-one.
    """
    corrected = list(matched)
    for word_index, (start_s, end_s, score) in zip(indices, spans):
        original = matched[word_index]
        corrected[word_index] = original.model_copy(
            update={
                "start_s": start_s,
                "end_s": end_s,
                "probability": score,
            }
        )
    return corrected
