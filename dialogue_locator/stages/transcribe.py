"""Transcription stage: audio -> word-level :class:`Transcript`.

This stage produces one thing and nothing else: a flat, continuous stream
of timestamped words covering the whole audio file.  It does no matching,
no searching, and makes no decisions about relevance.

**Why the word stream is flat.**  Whisper emits *segments*, and its
segment boundaries are an artefact of its decoding window, not of the
speech.  A target phrase lands across a boundary often enough that
searching within segments would miss it -- "My mind rebels" can easily end
one segment and "at stagnation" begin the next.  So segments are flattened
into a single ``list[Word]`` immediately, with ``segment_id`` recorded on
each word for provenance.  Every downstream stage matches on the flat
stream and never on segments.

**Why it is cached.**  ASR dominates the runtime of the whole pipeline --
minutes, on long media.  Transcripts are keyed by a hash of the audio
content plus every parameter that could change the output, so re-running
the tool against the same media is effectively free.
"""

from __future__ import annotations

import gc
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import structlog

from dialogue_locator.config import Settings
from dialogue_locator.errors import TranscriptionError
from dialogue_locator.models import Transcript, Word

logger = structlog.get_logger()

# How much *audio time* to cover between progress log lines.
_PROGRESS_INTERVAL_S = 30.0


# ---------------------------------------------------------------------------
# Cache keying
# ---------------------------------------------------------------------------


def _hash_file(path: Path) -> str:
    """Full SHA-256 of a file, read in chunks."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _cache_key(audio_path: Path, model_name: str, params: dict) -> str:
    """Key a transcript by audio content plus every output-affecting param.

    The audio is hashed by content rather than by path so that the same
    media acquired twice, or moved, still hits the cache -- and so that a
    changed file never silently reuses a stale transcript.
    """
    audio_sha = _hash_file(audio_path)
    param_blob = json.dumps(params, sort_keys=True)
    param_sha = hashlib.sha256(param_blob.encode("utf-8")).hexdigest()
    safe_model = model_name.replace("/", "_")
    return f"{audio_sha[:16]}.{safe_model}.{param_sha[:8]}"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model(model_name: str, device: str, compute_type: str):
    """Load a faster-whisper model on a specific device. Raises on failure."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:  # pragma: no cover - dependency is declared
        raise TranscriptionError(
            "faster-whisper is not installed. Run: uv sync"
        ) from e

    logger.info(
        "Loading ASR model",
        model=model_name,
        device=device,
        compute_type=compute_type,
    )
    try:
        return WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as e:
        raise TranscriptionError(
            f"Could not load ASR model {model_name!r} on {device!r} "
            f"with compute_type {compute_type!r}: {e}"
        ) from e


def _release(model) -> None:
    """Free the model's memory before returning.

    Later stages load their own models.  On a small card, leaving a large
    Whisper model resident is the difference between the next stage running
    and the next stage OOMing, so the handle is dropped explicitly rather
    than left to the garbage collector's discretion.
    """
    del model
    gc.collect()
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        # CTranslate2 does not depend on torch; it releases its own device
        # memory when the model object is collected.
        pass


# ---------------------------------------------------------------------------
# Core transcription
# ---------------------------------------------------------------------------


def _run_asr(
    audio_path: Path,
    *,
    model_name: str,
    device: str,
    compute_type: str,
    language: str | None,
    beam_size: int,
    vad_filter: bool,
    min_silence_duration_ms: int,
    time_offset_s: float,
) -> Transcript:
    """Run the model and flatten its segments into one word stream."""
    model = _load_model(model_name, device, compute_type)

    try:
        vad_parameters = (
            {"min_silence_duration_ms": min_silence_duration_ms}
            if vad_filter
            else None
        )

        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            task="transcribe",
            beam_size=beam_size,
            word_timestamps=True,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            condition_on_previous_text=False,
        )

        words: list[Word] = []
        started = time.perf_counter()
        next_report = _PROGRESS_INTERVAL_S

        # `segments` is a generator: the model does not actually run until
        # we iterate it, which is also what lets us report progress.
        for seg_id, segment in enumerate(segments):
            for w in segment.words or []:
                text = w.word.strip()
                if not text:
                    continue
                words.append(
                    Word(
                        text=text,
                        start_s=float(w.start) + time_offset_s,
                        end_s=float(w.end) + time_offset_s,
                        probability=(
                            float(w.probability)
                            if w.probability is not None
                            else None
                        ),
                        segment_id=seg_id,
                    )
                )

            if segment.end >= next_report:
                elapsed = max(time.perf_counter() - started, 1e-6)
                logger.info(
                    "Transcribing",
                    audio_s=round(float(segment.end), 1),
                    elapsed_s=round(elapsed, 1),
                    rate_x=round(float(segment.end) / elapsed, 2),
                    words=len(words),
                )
                while next_report <= segment.end:
                    next_report += _PROGRESS_INTERVAL_S

        detected_language = getattr(info, "language", None) or language or "unknown"
    except Exception as e:
        _release(model)
        raise TranscriptionError(f"Transcription failed: {e}") from e

    _release(model)

    if not words:
        raise TranscriptionError(
            f"Transcription produced no words for {audio_path}. "
            "The audio may be silent, or the wrong language may be configured."
        )

    logger.info(
        "Transcription complete",
        words=len(words),
        language=detected_language,
        device=device,
        compute_type=compute_type,
    )

    return Transcript(
        words=words,
        language=detected_language,
        model_name=model_name,
        source="asr",
    )


def transcribe(
    audio_path: Path,
    settings: Settings,
    cache_dir: Path,
    force: bool = False,
) -> Transcript:
    """Transcribe *audio_path* into a flat, word-level :class:`Transcript`.

    Parameters
    ----------
    settings:
        Supplies the model name, device, compute type, language, beam size
        and VAD threshold.
    cache_dir:
        Where the serialised transcript is cached.
    force:
        Skip the cache read and re-run the model.  The fresh result is
        still written back.
    """
    return _transcribe_cached(
        audio_path,
        cache_dir=cache_dir,
        force=force,
        model_name=settings.whisper_model,
        device=settings.device,
        compute_type=settings.whisper_compute_type,
        cpu_compute_type=settings.cpu_compute_type,
        language=settings.language,
        beam_size=settings.beam_size,
        vad_filter=True,
        min_silence_duration_ms=settings.min_silence_duration_ms,
        time_offset_s=0.0,
    )


def transcribe_window(
    audio_path: Path,
    settings: Settings,
    cache_dir: Path,
    *,
    model_name: str,
    vad_filter: bool = False,
    time_offset_s: float = 0.0,
    force: bool = False,
) -> Transcript:
    """Transcribe a slice of audio with an explicitly chosen model.

    Same contract as :func:`transcribe`, but lets the caller override the
    model and shift the emitted timestamps.  *time_offset_s* is added to
    every timestamp so that a slice cut out of a longer file reports times
    in the original file's domain.
    """
    return _transcribe_cached(
        audio_path,
        cache_dir=cache_dir,
        force=force,
        model_name=model_name,
        device=settings.device,
        compute_type=settings.whisper_compute_type,
        cpu_compute_type=settings.cpu_compute_type,
        language=settings.language,
        beam_size=settings.beam_size,
        vad_filter=vad_filter,
        min_silence_duration_ms=settings.min_silence_duration_ms,
        time_offset_s=time_offset_s,
    )


def _transcribe_cached(
    audio_path: Path,
    *,
    cache_dir: Path,
    force: bool,
    model_name: str,
    device: str,
    compute_type: str,
    cpu_compute_type: str,
    language: str | None,
    beam_size: int,
    vad_filter: bool,
    min_silence_duration_ms: int,
    time_offset_s: float,
) -> Transcript:
    if not audio_path.is_file():
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    cache_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "language": language,
        "beam_size": beam_size,
        "vad_filter": vad_filter,
        "min_silence_duration_ms": min_silence_duration_ms,
        "word_timestamps": True,
        "condition_on_previous_text": False,
        "task": "transcribe",
        "time_offset_s": time_offset_s,
    }
    key = _cache_key(audio_path, model_name, params)
    cache_path = cache_dir / f"{key}.transcript.json"

    if cache_path.is_file() and not force:
        try:
            cached = Transcript.model_validate_json(cache_path.read_text())
            logger.info(
                "Transcript cache hit",
                path=str(cache_path),
                words=len(cached.words),
            )
            return cached
        except Exception:
            logger.warning(
                "Discarding unreadable transcript cache", path=str(cache_path)
            )

    kwargs = {
        "model_name": model_name,
        "language": language,
        "beam_size": beam_size,
        "vad_filter": vad_filter,
        "min_silence_duration_ms": min_silence_duration_ms,
        "time_offset_s": time_offset_s,
    }

    # The CUDA fallback has to wrap the *whole run*, not just the model
    # load.  CTranslate2 constructs a CUDA model happily even when the
    # cuBLAS/cuDNN DLLs are missing; the failure only surfaces on the
    # first encode() call, which happens lazily part-way through iterating
    # the segment generator.  Catching it at load time would never fire.
    if device != "cpu":
        try:
            transcript = _run_asr(
                audio_path, device=device, compute_type=compute_type, **kwargs
            )
        except TranscriptionError as e:
            logger.warning(
                "GPU transcription failed; retrying on CPU",
                device=device,
                error=str(e),
                fallback_compute_type=cpu_compute_type,
            )
            transcript = _run_asr(
                audio_path,
                device="cpu",
                compute_type=cpu_compute_type,
                **kwargs,
            )
    else:
        transcript = _run_asr(
            audio_path, device="cpu", compute_type=cpu_compute_type, **kwargs
        )

    try:
        cache_path.write_text(transcript.model_dump_json())
        logger.debug("Transcript cached", path=str(cache_path))
    except OSError as e:
        logger.warning("Could not write transcript cache", error=str(e))

    return transcript


def slice_audio(
    audio_path: Path, out_path: Path, start_s: float, duration_s: float
) -> Path:
    """Cut ``[start_s, start_s + duration_s)`` out of a WAV file.

    Sample-accurate: the input is PCM, so ffmpeg's ``-ss`` seeks exactly
    rather than to the nearest compressed frame.  Callers must add
    *start_s* back onto every timestamp the slice produces.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-v", "error",
        "-ss", f"{max(0.0, start_s):.6f}",
        "-t", f"{duration_s:.6f}",
        "-i", str(audio_path),
        "-c", "copy",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise TranscriptionError(f"Audio slicing failed: {e.stderr}") from e
    return out_path


# ---------------------------------------------------------------------------
# Sidecar subtitles
# ---------------------------------------------------------------------------

_SRT_TIME = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)
_TAG = re.compile(r"<[^>]+>")


def _srt_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(srt_path: Path) -> Transcript:
    """Parse an ``.srt`` sidecar into a :class:`Transcript`.

    Subtitle files carry cue-level timing, not word-level timing, so word
    timestamps are interpolated across each cue proportionally to word
    length.  That is accurate enough to *locate* a line; a fine ASR pass
    is what pins down its exact onset.
    """
    try:
        raw = srt_path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as e:
        raise TranscriptionError(f"Could not read subtitles {srt_path}: {e}") from e

    words: list[Word] = []
    seg_id = 0

    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue

        time_match = None
        text_start = 0
        for i, line in enumerate(lines):
            time_match = _SRT_TIME.search(line)
            if time_match:
                text_start = i + 1
                break
        if not time_match:
            continue

        g = time_match.groups()
        start_s = _srt_seconds(*g[0:4])
        end_s = _srt_seconds(*g[4:8])

        text = " ".join(_TAG.sub("", ln) for ln in lines[text_start:]).strip()
        tokens = text.split()
        if not tokens:
            continue

        # Distribute the cue's duration across its words by character
        # length, so long words get proportionally more of the window.
        total_chars = sum(len(t) for t in tokens)
        span = max(end_s - start_s, 1e-6)
        cursor = start_s
        for token in tokens:
            share = (len(token) / total_chars) * span if total_chars else 0.0
            words.append(
                Word(
                    text=token,
                    start_s=cursor,
                    end_s=cursor + share,
                    probability=None,
                    segment_id=seg_id,
                )
            )
            cursor += share
        seg_id += 1

    if not words:
        raise TranscriptionError(f"No cues parsed from {srt_path}")

    logger.info("Parsed sidecar subtitles", path=str(srt_path), words=len(words))
    return Transcript(
        words=words,
        language="unknown",
        model_name=f"srt:{srt_path.name}",
        source="sidecar_subs",
    )


def find_sidecar_subs(video_path: Path) -> Path | None:
    """Return an ``.srt`` sitting next to *video_path*, if one exists."""
    candidates = sorted(video_path.parent.glob(f"{video_path.stem}*.srt"))
    return candidates[0] if candidates else None
