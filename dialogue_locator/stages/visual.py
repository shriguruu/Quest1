"""Visual stage: find burned-in on-screen text via OCR.

Used when the dialogue is *shown* rather than (or as well as) spoken --
hardcoded subtitles, title cards, chyrons -- and as a cross-check on the
audio answer when the two modalities are both available.

Running OCR on every frame of a feature-length video is not affordable, so
this stage narrows in three steps:

1. **Sample.**  One ffmpeg pass decodes the file and emits a few frames per
   second, cropped to the lower band where subtitles live.  A 54-minute
   video becomes a few thousand small JPEGs in about a minute.

2. **Prefilter.**  Most sampled frames contain no text at all.  A cheap
   contrast/edge-density heuristic rejects those without paying for OCR,
   which typically removes the large majority of samples.

3. **Refine.**  OCR the survivors and fuzzy-match each against the query.
   The best-scoring sample tells us the text is on screen at that moment,
   but not when it *arrived* -- sampling at 2fps leaves up to half a
   second of ambiguity.  So we binary-search backwards, frame by frame,
   for the first frame that still shows the text.  That boundary, not the
   sample, is the answer.

Step 3 is the part that matters for this problem: "the frame in which the
dialogue first appears" is a transition, and transitions have to be
searched for, not sampled.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Protocol

import structlog
from rapidfuzz import fuzz

from dialogue_locator.config import Settings
from dialogue_locator.errors import LocatorError
from dialogue_locator.models import (
    Candidate,
    MatchTier,
    MediaInfo,
    Transcript,
    Word,
)
from dialogue_locator.stages.framemap import FrameMapper
from dialogue_locator.stages.match import normalize_tokens

logger = structlog.get_logger()


class OcrEngine(Protocol):
    """Minimal OCR interface: an image path in, recognised text out."""

    def __call__(self, image_path: Path) -> str: ...


def _load_ocr_engine() -> OcrEngine:
    """Return an OCR callable, preferring RapidOCR then Tesseract.

    RapidOCR is preferred because it ships ONNX weights via pip and needs
    no system-level install, which matters for a tool meant to be cloned
    and run.  Tesseract is accepted as a fallback for environments that
    already have it.
    """
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore

        engine = RapidOCR()

        def _rapid(image_path: Path) -> str:
            result, _ = engine(str(image_path))
            if not result:
                return ""
            return " ".join(line[1] for line in result if len(line) > 1)

        logger.debug("OCR engine: rapidocr-onnxruntime")
        return _rapid
    except ImportError:
        pass

    try:
        import pytesseract  # type: ignore
        from PIL import Image

        if shutil.which("tesseract") is None:
            raise ImportError("tesseract binary not on PATH")

        def _tess(image_path: Path) -> str:
            return pytesseract.image_to_string(Image.open(image_path))

        logger.debug("OCR engine: pytesseract")
        return _tess
    except ImportError:
        pass

    raise ImportError(
        "No OCR engine available. Install one with:\n"
        "  uv add rapidocr-onnxruntime"
    )


def _sample_frames(
    media: MediaInfo, out_dir: Path, sample_fps: float, band_top: float
) -> list[tuple[float, Path]]:
    """Emit cropped sample frames; return ``(approx_time_s, path)`` pairs.

    Cropping to the lower band before writing cuts both disk traffic and
    OCR work, and removes most of the scene content that would otherwise
    produce spurious detections.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("s_*.jpg"))
    if existing:
        logger.info("Reusing cached OCR samples", count=len(existing))
        return [
            ((i) / sample_fps, path) for i, path in enumerate(existing)
        ]

    crop_h = f"ih*{1.0 - band_top:.4f}"
    crop_y = f"ih*{band_top:.4f}"
    vf = f"fps={sample_fps},crop=iw:{crop_h}:0:{crop_y}"

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(media.video_path),
        "-vf", vf,
        "-q:v", "3",
        str(out_dir / "s_%06d.jpg"),
    ]
    logger.info("Sampling frames for OCR", fps=sample_fps, band_top=band_top)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise LocatorError(f"Frame sampling failed: {e.stderr}") from e

    paths = sorted(out_dir.glob("s_*.jpg"))
    if not paths:
        raise LocatorError("Frame sampling produced no images")

    logger.info("Sampled frames", count=len(paths))
    # ffmpeg's fps filter emits frame i at approximately i / sample_fps.
    return [(i / sample_fps, path) for i, path in enumerate(paths)]


def _looks_like_text(image_path: Path, min_score: float = 0.0015) -> bool:
    """Cheap reject for frames that plainly contain no subtitle text.

    Subtitles are high-contrast glyphs against video: a small population
    of near-white pixels with strong local gradients.  Counting bright
    pixels is far cheaper than OCR and safely over-accepts.
    """
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - pillow is declared
        return True

    try:
        with Image.open(image_path) as img:
            small = img.convert("L").resize((160, 90))
            pixels = list(small.getdata())
    except OSError:
        return True

    bright = sum(1 for p in pixels if p > 200)
    return (bright / len(pixels)) >= min_score


def _score_text(text: str, query_tokens: list[str], query_norm: str) -> float:
    """Fuzzy-match OCR output against the query.

    ``partial_ratio`` is the right comparator here: OCR returns whatever
    else is on screen alongside the line we want, so the query should be
    scored as a substring rather than against the whole detection.
    """
    tokens = normalize_tokens(text)
    if not tokens:
        return 0.0
    candidate_norm = " ".join(tokens)
    if query_norm in candidate_norm:
        return 100.0
    return float(fuzz.partial_ratio(query_norm, candidate_norm))


def _ocr_frame_at(
    mapper: FrameMapper,
    frame_index: int,
    tmp_dir: Path,
    ocr: OcrEngine,
    media: MediaInfo,
    band_top: float,
) -> str:
    """Extract one exact frame, crop to the text band, and OCR it."""
    from PIL import Image

    pts_s = mapper.frame_to_time(frame_index)
    raw = tmp_dir / f"probe_{frame_index:08d}.png"
    mapper.extract(
        type(mapper.time_to_frame(pts_s))(
            frame_index=frame_index, pts_s=pts_s, image_path=None
        ),
        raw,
    )
    with Image.open(raw) as img:
        width, height = img.size
        cropped = img.crop((0, int(height * band_top), width, height))
        band = tmp_dir / f"band_{frame_index:08d}.png"
        cropped.save(band)
    return ocr(band)


def _first_frame_showing(
    mapper: FrameMapper,
    media: MediaInfo,
    hit_frame: int,
    query_tokens: list[str],
    query_norm: str,
    threshold: float,
    ocr: OcrEngine,
    tmp_dir: Path,
    band_top: float,
    max_back_frames: int,
) -> int:
    """Binary-search backwards for the first frame still showing the text.

    Invariant: *lo* does not show the text, *hi* does.  We shrink the gap
    until they are adjacent, and *hi* is the onset frame.
    """
    lo = max(0, hit_frame - max_back_frames)
    hi = hit_frame

    # Establish the invariant. If even the earliest bracket frame shows the
    # text, the subtitle started before our search window - report the
    # bracket edge rather than pretending to know better.
    lo_text = _ocr_frame_at(mapper, lo, tmp_dir, ocr, media, band_top)
    if _score_text(lo_text, query_tokens, query_norm) >= threshold:
        logger.warning(
            "Text already present at the start of the refine bracket",
            frame=lo,
        )
        return lo

    probes = 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        text = _ocr_frame_at(mapper, mid, tmp_dir, ocr, media, band_top)
        if _score_text(text, query_tokens, query_norm) >= threshold:
            hi = mid
        else:
            lo = mid
        probes += 1

    logger.info("Refined to onset frame", frame=hi, probes=probes)
    return hi


def locate_visual(
    media: MediaInfo,
    mapper: FrameMapper,
    query: str,
    *,
    settings: Settings,
    cache_dir: Path,
) -> dict[str, Any] | None:
    """Locate *query* as on-screen text. Returns ``None`` if not found."""
    ocr = _load_ocr_engine()

    query_tokens = normalize_tokens(query)
    if not query_tokens:
        raise ValueError("Query is empty after normalisation")
    query_norm = " ".join(query_tokens)

    sample_dir = cache_dir / f"{media.video_sha256}.ocr_samples"
    samples = _sample_frames(
        media, sample_dir, settings.ocr_sample_fps, settings.ocr_band_top
    )

    # --- Prefilter + OCR ---
    best_score = 0.0
    best_time = 0.0
    best_text = ""
    scanned = 0

    for approx_s, path in samples:
        if not _looks_like_text(path):
            continue
        scanned += 1
        text = ocr(path)
        if not text.strip():
            continue
        score = _score_text(text, query_tokens, query_norm)
        if score > best_score:
            best_score, best_time, best_text = score, approx_s, text
            if score >= 100.0:
                break

    logger.info(
        "OCR scan complete",
        sampled=len(samples),
        ocr_run=scanned,
        best_score=round(best_score, 1),
        best_at=round(best_time, 2),
    )

    if best_score < settings.uncertain_threshold:
        return None

    # --- Refine to the exact onset frame ---
    tmp_dir = cache_dir / f"{media.video_sha256}.ocr_refine"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    hit_ref = mapper.time_to_frame(min(best_time, media.duration_s - 1e-3))
    # Search back over slightly more than one sampling interval, which is
    # the whole window in which the text could have appeared.
    fps = float(media.fps)
    max_back = int(fps / settings.ocr_sample_fps) + int(fps * 0.5)

    warnings: list[str] = []
    try:
        onset_frame = _first_frame_showing(
            mapper,
            media,
            hit_ref.frame_index,
            query_tokens,
            query_norm,
            threshold=max(settings.uncertain_threshold, best_score * 0.85),
            ocr=ocr,
            tmp_dir=tmp_dir,
            band_top=settings.ocr_band_top,
            max_back_frames=max_back,
        )
    except LocatorError as e:
        warnings.append(f"Onset refinement failed ({e}); using sample time.")
        onset_frame = hit_ref.frame_index

    frame_ref = mapper.time_to_frame(mapper.frame_to_time(onset_frame))

    matched_text = " ".join(best_text.split())
    transcript = Transcript(
        words=[
            Word(
                text=matched_text,
                start_s=frame_ref.pts_s,
                end_s=frame_ref.pts_s,
                probability=None,
                segment_id=0,
            )
        ],
        language="unknown",
        model_name="ocr",
        source="ocr",
    )
    candidate = Candidate(
        word_index_start=0,
        word_index_end=0,
        matched_text=matched_text,
        score=round(best_score, 2),
        tier=MatchTier.EXACT if best_score >= 100 else MatchTier.FUZZY,
        coarse_start_s=frame_ref.pts_s,
    )

    return {
        "score": best_score,
        "frame_ref": frame_ref,
        "candidate": candidate,
        "transcript": transcript,
        "warnings": warnings,
    }
