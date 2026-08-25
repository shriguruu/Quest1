"""Frame-mapping stage: timestamp -> frame index -> extracted PNG.

This module is the single source of truth for the relationship between
wall-clock time and discrete frame indices.  It is also the *only* place
where the A/V offset correction is applied.

There are two ways to answer "which frame is on screen at time t":

1. **Analytic CFR** -- ``frame_index = floor(t * fps)`` using exact
   :class:`~fractions.Fraction` arithmetic.  Correct only if the file's
   frames really do sit on a clean ``i/fps`` grid starting at zero.

2. **Measured PTS index** -- read every video packet's presentation
   timestamp out of the container and binary-search it.  Exact for CFR,
   for VFR, and for the very common real-world case of a nominally-CFR
   file whose frames are offset from the ideal grid (an HLS remux, a
   stream that starts on a partial GOP, a concatenation).

The measured index is authoritative whenever we have one.  The analytic
path is the fallback for when we don't, and a useful cross-check.
"""

from __future__ import annotations

import bisect
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

import av
import structlog

from dialogue_locator.errors import FrameExtractionError
from dialogue_locator.models import FrameRef, MediaInfo

logger = structlog.get_logger()


def format_timestamp(seconds: float) -> str:
    """Format *seconds* as ``HH:MM:SS.mmm``."""
    if seconds < 0:
        raise ValueError(f"Negative timestamp: {seconds}")
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


# ---------------------------------------------------------------------------
# PTS index
# ---------------------------------------------------------------------------


def _ffprobe_pts(video_path: Path, entity: str) -> list[float]:
    """Return every ``pts_time`` for *entity* (``packet`` or ``frame``)."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", f"{entity}=pts_time",
        "-of", "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    out: list[float] = []
    for line in result.stdout.splitlines():
        token = line.strip().rstrip(",")
        if not token or token == "N/A":
            continue
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out


def build_pts_index(
    video_path: Path,
    cache_dir: Path | None = None,
    video_sha256: str | None = None,
) -> list[float]:
    """Return the sorted presentation timestamps of every video frame.

    The returned list is indexed by **display-order frame number**: entry
    ``i`` is the PTS in seconds of frame ``i``.

    Packet timestamps are read first because ffprobe can list them without
    decoding, which is far faster than a full decode.  For a container
    with B-frames the *set* of packet PTS equals the set of frame PTS, so
    sorting recovers display order.  If the container reports no usable
    packet timestamps we fall back to frame-level probing, which decodes
    but always works.

    **Caveat: stream-copied fragments.**  One packet yields one frame for
    any well-formed file, but a fragment cut mid-GOP with ``ffmpeg -c copy``
    carries leading packets that reference data outside the fragment.  A
    decoder discards those, so it emits fewer frames than there are
    packets, and this index will number frames higher than the decoder
    would.  Timestamps stay correct either way -- :meth:`FrameMapper.extract`
    verifies the decoded PTS against the requested PTS, so the image is
    always the frame at the reported timestamp.  Only the *index* shifts,
    and only for such fragments.  Re-encode rather than stream-copy if you
    need frame numbers from a clip to line up with the source.

    Results are cached under *video_sha256* when a *cache_dir* is given,
    since the index is a pure function of the file.
    """
    cache_path: Path | None = None
    if cache_dir is not None and video_sha256:
        cache_path = cache_dir / f"{video_sha256}.pts.json"
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text())
                if isinstance(cached, list) and cached:
                    logger.debug(
                        "PTS index cache hit",
                        path=str(cache_path),
                        frames=len(cached),
                    )
                    return [float(x) for x in cached]
            except (json.JSONDecodeError, ValueError, OSError):
                logger.warning("Discarding corrupt PTS cache", path=str(cache_path))

    logger.info("Building PTS index", video=str(video_path))
    try:
        pts = _ffprobe_pts(video_path, "packet")
        source = "packet"
        if not pts:
            pts = _ffprobe_pts(video_path, "frame")
            source = "frame"
    except subprocess.CalledProcessError as e:
        raise FrameExtractionError(f"ffprobe PTS scan failed: {e.stderr}") from e

    if not pts:
        raise FrameExtractionError(f"No video timestamps found in {video_path}")

    pts.sort()
    logger.info("PTS index built", frames=len(pts), source=source)

    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(pts))
        except OSError as e:
            logger.warning("Could not write PTS cache", error=str(e))

    return pts


class FrameMapper:
    """Map between wall-clock time and discrete frame indices.

    Parameters
    ----------
    media:
        A probed :class:`MediaInfo` describing the video.
    pts_index:
        Measured frame timestamps from :func:`build_pts_index`.  When
        supplied this is **authoritative** and the analytic CFR formula is
        not used at all.  When omitted, VFR media builds one on demand and
        CFR media uses exact ``Fraction`` arithmetic.
    cache_dir:
        Where to cache a lazily-built PTS index.
    """

    def __init__(
        self,
        media: MediaInfo,
        pts_index: list[float] | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self._media = media
        self._pts = pts_index
        self._cache_dir = cache_dir
        self._container: av.container.InputContainer | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def time_to_frame(self, video_time_s: float) -> FrameRef:
        """Return the frame on screen at *video_time_s*.

        The on-screen frame is the **last** frame whose PTS is <=
        *video_time_s* -- floor, never round.  A viewer who pauses at time
        ``t`` sees the frame that was most recently presented, not the one
        that is about to be.
        """
        self._guard_range(video_time_s)

        pts = self._ensure_index()
        if pts is not None:
            return self._indexed_time_to_frame(video_time_s, pts)
        return self._cfr_time_to_frame(video_time_s)

    def audio_time_to_frame(self, audio_time_s: float) -> FrameRef:
        """Map an audio-domain timestamp to a video frame.

        This is the **only** place where :pyattr:`MediaInfo.av_offset_s` is
        applied.  The correction converts from audio time to video time::

            video_time = audio_time + av_offset_s

        Callers must **not** apply the offset themselves.
        """
        video_time_s = audio_time_s + self._media.av_offset_s
        return self.time_to_frame(video_time_s)

    def frame_to_time(self, frame_index: int) -> float:
        """Return the PTS in seconds of *frame_index*."""
        if frame_index < 0:
            raise ValueError(f"frame_index must be >= 0, got {frame_index}")

        pts = self._ensure_index()
        if pts is not None:
            if frame_index >= len(pts):
                raise ValueError(
                    f"frame_index {frame_index} exceeds frame count {len(pts)}"
                )
            return pts[frame_index]
        return float(Fraction(frame_index) / self._media.fps)

    def extract(self, frame_ref: FrameRef, out_path: Path) -> Path:
        """Decode and save the frame described by *frame_ref* as a PNG.

        Uses PyAV seek-then-decode (NOT ``ffmpeg -ss``) so that we land on
        the exact frame rather than the nearest keyframe.

        Raises :class:`FrameExtractionError` if no decoded frame's PTS
        matches *frame_ref.pts_s* within half a frame duration.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        half_frame_s = float(Fraction(1, 2) / self._media.fps)

        # Two attempts: a tight seek, then a more generous rewind in case
        # the first landed after the target.
        for rewind_s in (0.0, 5.0):
            found = self._decode_frame_at(frame_ref.pts_s, half_frame_s, rewind_s)
            if found is not None:
                found.to_image().save(str(out_path))
                logger.debug(
                    "Extracted frame",
                    index=frame_ref.frame_index,
                    pts_s=frame_ref.pts_s,
                    path=str(out_path),
                )
                return out_path

        raise FrameExtractionError(
            f"Could not decode frame at PTS {frame_ref.pts_s:.6f}s "
            f"(index {frame_ref.frame_index})"
        )

    def close(self) -> None:
        """Release the underlying PyAV container, if open."""
        if self._container is not None:
            self._container.close()
            self._container = None

    def __enter__(self) -> FrameMapper:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_index(self) -> list[float] | None:
        """Return the PTS index, building it lazily for VFR media."""
        if self._pts is not None:
            return self._pts
        if not self._media.is_vfr:
            return None

        self._pts = build_pts_index(
            self._media.video_path,
            cache_dir=self._cache_dir,
            video_sha256=self._media.video_sha256,
        )
        return self._pts

    @staticmethod
    def _indexed_time_to_frame(video_time_s: float, pts: list[float]) -> FrameRef:
        """Binary-search the measured index for the last PTS <= target."""
        idx = bisect.bisect_right(pts, video_time_s) - 1
        if idx < 0:
            # Target precedes the first frame; the first frame is what a
            # player would show.
            idx = 0
        return FrameRef(frame_index=idx, pts_s=pts[idx], image_path=None)

    def _cfr_time_to_frame(self, video_time_s: float) -> FrameRef:
        """Analytic path: exact Fraction arithmetic, int() only at the end.

        ``29.97 * 600`` in floating point is not ``30000/1001 * 600``.  At
        certain timestamps the difference straddles a frame boundary and
        the naive computation returns the wrong index, so every step here
        stays rational until the final floor.
        """
        t = Fraction(video_time_s).limit_denominator(10**9)
        fps = self._media.fps  # already a Fraction

        frame_index = int(math.floor(t * fps))

        if self._media.total_frames is not None:
            frame_index = min(frame_index, self._media.total_frames - 1)
        frame_index = max(frame_index, 0)

        pts_s = float(Fraction(frame_index) / fps)
        return FrameRef(frame_index=frame_index, pts_s=pts_s, image_path=None)

    def _decode_frame_at(
        self, target_pts_s: float, half_frame_s: float, rewind_s: float
    ) -> av.VideoFrame | None:
        """Seek before *target_pts_s* and decode forward to reach it."""
        container = self._open()
        stream = container.streams.video[0]

        seek_pts_s = max(0.0, target_pts_s - rewind_s)
        seek_target = int(seek_pts_s / stream.time_base)
        container.seek(max(0, seek_target - 1), stream=stream, backward=True)

        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            frame_pts_s = float(frame.pts * stream.time_base)
            if abs(frame_pts_s - target_pts_s) <= half_frame_s:
                return frame
            if frame_pts_s > target_pts_s + half_frame_s:
                return None  # overshot without a match
        return None

    def _open(self) -> av.container.InputContainer:
        if self._container is None:
            self._container = av.open(str(self._media.video_path))
        return self._container

    def _guard_range(self, video_time_s: float) -> None:
        if video_time_s < 0:
            raise ValueError(f"video_time_s must be >= 0, got {video_time_s}")
        if video_time_s > self._media.duration_s:
            raise ValueError(
                f"video_time_s {video_time_s} exceeds duration "
                f"{self._media.duration_s}"
            )
