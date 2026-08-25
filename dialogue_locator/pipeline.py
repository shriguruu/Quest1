"""Pipeline orchestration: a URL and a query in, a :class:`LocateResult` out.

    acquire -> probe -> extract_audio -> transcribe
                                             |
                          build_search_index -> find_candidates -> classify
                                             |
                                        AlignmentResult
                                             |
                          audio_time_to_frame -> FrameRef -> PNG

This is the complete working system, built on Whisper's own word
timestamps.  Those timestamps are a byproduct of the model's attention
weights rather than a measurement, and they are routinely off by a couple
of hundred milliseconds -- which at 24fps is several frames.  Rather than
hide that, the alignment produced here is explicitly labelled
``whisper_fallback`` with a stated uncertainty, so every consumer knows it
is holding the imprecise answer.  Forced alignment replaces this one step
later without touching any other stage.

Every stage is timed, and every failure below the top level is converted
into a structured ERROR result rather than a traceback: this is a CLI, and
a stack trace is not an answer.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import structlog

from dialogue_locator.config import Settings
from dialogue_locator.errors import LocatorError
from dialogue_locator.models import (
    Candidate,
    LocateResult,
    MediaInfo,
    ResultStatus,
    Transcript,
)
from dialogue_locator.stages.acquire import acquire, extract_audio, probe
from dialogue_locator.stages.align import align
from dialogue_locator.stages.framemap import (
    FrameMapper,
    build_pts_index,
    format_timestamp,
)
from dialogue_locator.stages.match import (
    SearchIndex,
    build_search_index,
    classify,
    find_candidates,
)
from dialogue_locator.stages.transcribe import transcribe

logger = structlog.get_logger()

# Whisper's word boundaries are inferred, not measured.  This is the
# honest error bar on them until forced alignment lands.
WHISPER_UNCERTAINTY_S = 0.3
# Floor used only to surface near-misses on a NOT_FOUND result.
_NEAR_MISS_FLOOR = 35.0


class StageTimer:
    """Accumulate per-stage wall-clock timings in milliseconds."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - started) * 1000.0
            self.timings[name] = round(self.timings.get(name, 0.0) + elapsed, 2)


def _error_result(
    message: str,
    timer: StageTimer,
    media: MediaInfo | None = None,
    warnings: list[str] | None = None,
) -> LocateResult:
    """Build a structured ERROR result instead of raising."""
    return LocateResult(
        status=ResultStatus.ERROR,
        timestamp=None,
        frame=None,
        text=None,
        confidence=0.0,
        modality=None,
        alternates=[],
        warnings=[*(warnings or []), message],
        media=media,
        timings_ms=timer.timings,
    )


def _not_found_result(
    index: SearchIndex,
    query: str,
    media: MediaInfo,
    timer: StageTimer,
    warnings: list[str],
) -> LocateResult:
    """Return NOT_FOUND, but show what the search *did* turn up.

    An empty answer is much harder to act on than a wrong one.  Re-running
    the search with a deliberately low floor costs milliseconds and lets
    the user see whether the line is genuinely absent, or present in a form
    the thresholds rejected -- which is usually the difference between "not
    in this video" and "your query is worded differently".
    """
    try:
        near_misses = find_candidates(
            index, query, top_k=3, min_score=_NEAR_MISS_FLOOR
        )
    except ValueError:
        near_misses = []

    if near_misses:
        warnings.append(
            f"Closest near-miss scored {near_misses[0].score:.1f} at "
            f"{format_timestamp(near_misses[0].coarse_start_s)}."
        )

    return LocateResult(
        status=ResultStatus.NOT_FOUND,
        timestamp=None,
        frame=None,
        text=None,
        confidence=round(near_misses[0].score, 2) if near_misses else 0.0,
        matched_tier=near_misses[0].tier if near_misses else None,
        modality="audio",
        alternates=near_misses,
        warnings=warnings,
        media=media,
        timings_ms=timer.timings,
    )


def locate(
    url: str,
    query: str,
    settings: Settings,
    *,
    video_path: Path | None = None,
    output_dir: Path | None = None,
    force: bool = False,
) -> LocateResult:
    """Find the frame where *query* is first spoken.

    Parameters
    ----------
    url:
        Media URL to download.  Ignored when *video_path* is supplied, in
        which case it is kept only as provenance on the result.
    video_path:
        A local file to use instead of downloading.
    force:
        Bypass every cache: re-download, re-transcribe.

    Never raises :class:`LocatorError` -- failures come back as a result
    with status ``ERROR``.
    """
    timer = StageTimer()
    warnings: list[str] = []
    cache_dir = Path(settings.cache_dir)
    out_dir = output_dir or Path(settings.output_dir)
    media: MediaInfo | None = None
    mapper: FrameMapper | None = None

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Acquire + probe ---------------------------------------
        if video_path is None:
            with timer.stage("acquire"):
                video_path = acquire(url, cache_dir, force=force)
        else:
            logger.info("Using local file", path=str(video_path))

        with timer.stage("probe"):
            media = probe(
                video_path, url=url or f"file://{video_path.absolute()}"
            )

        logger.info(
            "Probed media",
            fps=str(media.fps),
            vfr=media.is_vfr,
            frames=media.total_frames,
            duration=format_timestamp(media.duration_s),
        )

        # The frame number we report is only meaningful against real frame
        # timings, so the measured PTS index is built up front rather than
        # trusting the container's nominal frame rate.
        with timer.stage("pts_index"):
            pts_index = build_pts_index(
                video_path, cache_dir=cache_dir, video_sha256=media.video_sha256
            )
        if media.is_vfr:
            warnings.append(
                "Frame timing is not uniform; frame numbers come from "
                "measured presentation timestamps, not the nominal frame rate."
            )

        frame_duration_s = float(Fraction(1) / media.fps)
        mapper = FrameMapper(media, pts_index=pts_index, cache_dir=cache_dir)

        # --- 2. Extract audio -----------------------------------------
        audio_path = cache_dir / f"{media.video_sha256}.wav"
        if force or not audio_path.is_file():
            with timer.stage("extract_audio"):
                extract_audio(video_path, audio_path)
        else:
            logger.info("Reusing extracted audio", path=str(audio_path))

        # --- 3. Transcribe --------------------------------------------
        with timer.stage("transcribe"):
            transcript: Transcript = transcribe(
                audio_path, settings, cache_dir, force=force
            )

        # --- 4. Match + classify --------------------------------------
        with timer.stage("match"):
            index = build_search_index(transcript.words)
            candidates = find_candidates(
                index, query, top_k=settings.max_alternates
            )
            status, match_warnings = classify(candidates, settings)
        warnings.extend(match_warnings)

        # --- 5. Bail out early, but say what we saw --------------------
        if status is ResultStatus.NOT_FOUND or not candidates:
            return _not_found_result(index, query, media, timer, warnings)

        # --- 6. Alignment (forced, falling back to Whisper) ----------
        best: Candidate = candidates[0]
        with timer.stage("align"):
            alignment = align(
                audio_path, media, best, transcript, settings
            )
        warnings.extend(alignment.warnings)

        if alignment.uncertainty_s > frame_duration_s:
            span = alignment.uncertainty_s / frame_duration_s
            hint = (
                ""
                if alignment.method == "forced"
                else " Forced alignment would narrow this."
            )
            warnings.append(
                f"Onset precision is +/-{alignment.uncertainty_s * 1000:.0f}ms "
                f"(~{span:.1f} frames at {media.fps} fps); the true first "
                f"frame may differ.{hint}"
            )

        # --- 7. Map onto a frame --------------------------------------
        with timer.stage("frame_map"):
            frame_ref = mapper.audio_time_to_frame(alignment.onset_s)

        # --- 8. Save the frame ----------------------------------------
        with timer.stage("extract_frame"):
            image_path = out_dir / f"frame_{frame_ref.frame_index}.png"
            try:
                mapper.extract(frame_ref, image_path)
                frame_ref = frame_ref.model_copy(update={"image_path": image_path})
            except LocatorError as e:
                warnings.append(f"Could not save the frame image: {e}")

        # --- 9. Assemble ----------------------------------------------
        return LocateResult(
            status=status,
            timestamp=format_timestamp(frame_ref.pts_s),
            frame=frame_ref,
            text=best.matched_text,
            confidence=best.score,
            matched_tier=best.tier,
            alignment=alignment,
            modality="audio",
            alternates=candidates[1:],
            warnings=warnings,
            media=media,
            timings_ms=timer.timings,
        )

    except LocatorError as e:
        logger.error("Pipeline failed", error=str(e), kind=type(e).__name__)
        return _error_result(f"{type(e).__name__}: {e}", timer, media, warnings)
    except ValueError as e:
        logger.error("Invalid input", error=str(e))
        return _error_result(f"Invalid input: {e}", timer, media, warnings)
    finally:
        if mapper is not None:
            mapper.close()
