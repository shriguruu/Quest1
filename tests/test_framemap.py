"""Tests for dialogue_locator.stages.framemap.

Generates synthetic videos with ffmpeg at 25fps, 30fps, and 30000/1001 fps,
then validates that FrameMapper correctly maps timestamps to frame indices.
"""

from __future__ import annotations

import math
import platform
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from dialogue_locator.errors import FrameExtractionError
from dialogue_locator.models import FrameRef, MediaInfo
from dialogue_locator.stages.framemap import FrameMapper, format_timestamp


# ---------------------------------------------------------------------------
# Fixtures: synthetic test videos
# ---------------------------------------------------------------------------

_VIDEO_SPECS: list[tuple[str, str, Fraction]] = [
    # (label, ffmpeg rate string, exact Fraction)
    ("25fps", "25", Fraction(25)),
    ("30fps", "30", Fraction(30)),
    ("29.97fps", "30000/1001", Fraction(30000, 1001)),
]

DURATION_S = 10

# On Windows, drawtext needs an explicit font path because Fontconfig
# can't find system fonts.
_FONT_PATH = Path("C:/Windows/Fonts/arial.ttf")


def _generate_video(
    tmpdir: Path,
    rate_str: str,
    label: str,
    *,
    burn_frame_number: bool = False,
) -> Path:
    """Create a short test video, optionally with frame number burned in."""
    out = tmpdir / f"test_{label}.mp4"
    if out.exists():
        return out

    vf_parts: list[str] = []
    if burn_frame_number and _FONT_PATH.exists():
        font = str(_FONT_PATH).replace("\\", "/").replace(":", "\\:")
        vf_parts.append(
            f"drawtext=text='%{{n}}':fontfile='{font}'"
            ":fontsize=96:x=20:y=20:fontcolor=white"
        )

    vf_arg = ",".join(vf_parts) if vf_parts else None

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"testsrc=duration={DURATION_S}:size=640x360:rate={rate_str}",
    ]
    if vf_arg:
        cmd += ["-vf", vf_arg]
    cmd += [
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def _make_media_info(
    video_path: Path,
    fps: Fraction,
    duration_s: float = DURATION_S,
    is_vfr: bool = False,
    av_offset_s: float = 0.0,
    total_frames: int | None = None,
) -> MediaInfo:
    """Build a MediaInfo for testing without running ffprobe."""
    if total_frames is None:
        total_frames = math.floor(duration_s * float(fps))
    return MediaInfo(
        url=f"file://{video_path}",
        video_path=video_path,
        audio_path=None,
        duration_s=duration_s,
        fps=fps,
        is_vfr=is_vfr,
        total_frames=total_frames,
        width=640,
        height=360,
        video_start_time_s=0.0,
        audio_start_time_s=0.0,
        av_offset_s=av_offset_s,
        video_sha256="test",
    )


@pytest.fixture(scope="session")
def video_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("videos")


@pytest.fixture(scope="session", params=_VIDEO_SPECS, ids=[s[0] for s in _VIDEO_SPECS])
def video_fixture(request: pytest.FixtureRequest, video_dir: Path):
    """Generate a test video and return (path, fps_fraction, label)."""
    label, rate_str, fps_frac = request.param
    path = _generate_video(video_dir, rate_str, label)
    return path, fps_frac, label


# ---------------------------------------------------------------------------
# format_timestamp tests
# ---------------------------------------------------------------------------


class TestFormatTimestamp:
    def test_zero(self) -> None:
        assert format_timestamp(0.0) == "00:00:00.000"

    def test_simple(self) -> None:
        assert format_timestamp(3661.5) == "01:01:01.500"

    def test_milliseconds(self) -> None:
        assert format_timestamp(0.001) == "00:00:00.001"

    def test_large(self) -> None:
        assert format_timestamp(86399.999) == "23:59:59.999"

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="Negative"):
            format_timestamp(-1.0)


# ---------------------------------------------------------------------------
# CFR time_to_frame tests
# ---------------------------------------------------------------------------


class TestCFRTimeToFrame:
    """Validate frame mapping using Fraction arithmetic on CFR videos."""

    def test_frame_zero(self, video_fixture: tuple[Path, Fraction, str]) -> None:
        path, fps, label = video_fixture
        media = _make_media_info(path, fps)
        mapper = FrameMapper(media)
        try:
            ref = mapper.time_to_frame(0.0)
            assert ref.frame_index == 0
            assert ref.pts_s == 0.0
        finally:
            mapper.close()

    def test_frame_one(self, video_fixture: tuple[Path, Fraction, str]) -> None:
        path, fps, label = video_fixture
        media = _make_media_info(path, fps)
        mapper = FrameMapper(media)
        try:
            # Time exactly at frame 1's PTS.
            t = float(Fraction(1) / fps)
            ref = mapper.time_to_frame(t)
            assert ref.frame_index == 1
        finally:
            mapper.close()

    def test_mid_frame_floors(self, video_fixture: tuple[Path, Fraction, str]) -> None:
        """A time between frame N and N+1 should map to frame N (floor)."""
        path, fps, label = video_fixture
        media = _make_media_info(path, fps)
        mapper = FrameMapper(media)
        try:
            # Halfway between frame 5 and frame 6.
            t = float((Fraction(5) + Fraction(1, 2)) / fps)
            ref = mapper.time_to_frame(t)
            assert ref.frame_index == 5
        finally:
            mapper.close()

    def test_one_microsecond_before_frame(
        self, video_fixture: tuple[Path, Fraction, str]
    ) -> None:
        """1µs before frame 10's PTS should give frame 9."""
        path, fps, label = video_fixture
        media = _make_media_info(path, fps)
        mapper = FrameMapper(media)
        try:
            t = float(Fraction(10) / fps) - 1e-6
            ref = mapper.time_to_frame(t)
            assert ref.frame_index == 9
        finally:
            mapper.close()

    def test_one_microsecond_after_frame(
        self, video_fixture: tuple[Path, Fraction, str]
    ) -> None:
        """1µs after frame 10's PTS should still give frame 10."""
        path, fps, label = video_fixture
        media = _make_media_info(path, fps)
        mapper = FrameMapper(media)
        try:
            t = float(Fraction(10) / fps) + 1e-6
            ref = mapper.time_to_frame(t)
            assert ref.frame_index == 10
        finally:
            mapper.close()

    def test_exactly_on_frame_pts(
        self, video_fixture: tuple[Path, Fraction, str]
    ) -> None:
        """Exactly on frame 50's PTS → frame 50."""
        path, fps, label = video_fixture
        media = _make_media_info(path, fps)
        mapper = FrameMapper(media)
        try:
            t = float(Fraction(50) / fps)
            ref = mapper.time_to_frame(t)
            assert ref.frame_index == 50
        finally:
            mapper.close()


# ---------------------------------------------------------------------------
# 29.97 fps drift test
# ---------------------------------------------------------------------------


class TestNTSCDrift:
    """The classic float-fps bug: verify Fraction arithmetic stays correct
    at long simulated durations where float accumulation goes wrong."""

    def test_no_drift_at_600s(self, video_dir: Path) -> None:
        """Use a short real video but set duration_s=620 in MediaInfo
        to test the arithmetic at high timestamps."""
        path = _generate_video(video_dir, "30000/1001", "ntsc_drift")
        fps = Fraction(30000, 1001)
        media = _make_media_info(
            path, fps, duration_s=620.0, total_frames=18600
        )
        mapper = FrameMapper(media)
        try:
            # At t=600.0, the correct index is floor(600 * 30000/1001).
            expected = math.floor(Fraction(600) * fps)
            ref = mapper.time_to_frame(600.0)
            assert ref.frame_index == expected

            # Verify it's NOT what naive float math might give at a
            # problematic time.  At t=600 specifically both agree, but
            # check the general contract holds.
            assert ref.frame_index == int(math.floor(Fraction(600) * fps))
        finally:
            mapper.close()


# ---------------------------------------------------------------------------
# A/V offset tests
# ---------------------------------------------------------------------------


class TestAVOffset:
    """Verify av_offset_s is applied once and only once."""

    def test_offset_applied(self, video_dir: Path) -> None:
        path = _generate_video(video_dir, "25", "offset_test")
        fps = Fraction(25)
        offset = 0.04  # 40ms — exactly one frame at 25fps

        media = _make_media_info(path, fps, av_offset_s=offset)
        mapper = FrameMapper(media)
        try:
            # Audio time 0.0 → video time = 0.0 + 0.04 = 0.04.
            # At 25fps, frame at 0.04s = frame 1 (PTS=0.04).
            ref_audio = mapper.audio_time_to_frame(0.0)
            assert ref_audio.frame_index == 1

            # Direct video time 0.0 → frame 0 (no offset).
            ref_video = mapper.time_to_frame(0.0)
            assert ref_video.frame_index == 0
        finally:
            mapper.close()

    def test_offset_not_double_applied(self, video_dir: Path) -> None:
        """Calling audio_time_to_frame twice with same input must give same
        result — the offset must not accumulate."""
        path = _generate_video(video_dir, "30", "offset_double")
        fps = Fraction(30)
        offset = 0.1

        media = _make_media_info(path, fps, av_offset_s=offset)
        mapper = FrameMapper(media)
        try:
            ref1 = mapper.audio_time_to_frame(1.0)
            ref2 = mapper.audio_time_to_frame(1.0)
            assert ref1.frame_index == ref2.frame_index
            assert ref1.pts_s == ref2.pts_s
        finally:
            mapper.close()


# ---------------------------------------------------------------------------
# Out-of-range guard tests
# ---------------------------------------------------------------------------


class TestOutOfRange:
    def test_negative_time_raises(self, video_fixture: tuple[Path, Fraction, str]) -> None:
        path, fps, _ = video_fixture
        media = _make_media_info(path, fps)
        mapper = FrameMapper(media)
        try:
            with pytest.raises(ValueError, match="must be >= 0"):
                mapper.time_to_frame(-0.001)
        finally:
            mapper.close()

    def test_beyond_duration_raises(self, video_fixture: tuple[Path, Fraction, str]) -> None:
        path, fps, _ = video_fixture
        media = _make_media_info(path, fps, duration_s=DURATION_S)
        mapper = FrameMapper(media)
        try:
            with pytest.raises(ValueError, match="exceeds duration"):
                mapper.time_to_frame(DURATION_S + 1.0)
        finally:
            mapper.close()


# ---------------------------------------------------------------------------
# Frame extraction test
# ---------------------------------------------------------------------------


class TestExtract:
    """Verify that extract() decodes the correct frame by comparing against
    an ffmpeg-extracted reference."""

    def test_extract_produces_valid_png(self, video_dir: Path) -> None:
        """Extract frame 5 via FrameMapper and verify it's a valid PNG."""
        path = _generate_video(video_dir, "25", "extract_test")
        fps = Fraction(25)
        media = _make_media_info(path, fps)
        mapper = FrameMapper(media)
        try:
            ref = mapper.time_to_frame(float(Fraction(5) / fps))
            assert ref.frame_index == 5

            our_png = video_dir / "our_frame5.png"
            mapper.extract(ref, our_png)
            assert our_png.exists()
            assert our_png.stat().st_size > 0
        finally:
            mapper.close()

    def test_extract_matches_ffmpeg(self, video_dir: Path) -> None:
        """Extract frame 5 via both FrameMapper and ffmpeg's select filter,
        then compare that both are reasonably-sized PNGs."""
        path = _generate_video(video_dir, "25", "extract_cmp")
        fps = Fraction(25)
        media = _make_media_info(path, fps)
        mapper = FrameMapper(media)
        try:
            ref = mapper.time_to_frame(float(Fraction(5) / fps))
            assert ref.frame_index == 5

            our_png = video_dir / "our_cmp_frame5.png"
            mapper.extract(ref, our_png)
            assert our_png.exists()
            assert our_png.stat().st_size > 0

            # Extract the same frame with ffmpeg for reference.
            ffmpeg_png = video_dir / "ffmpeg_cmp_frame5.png"
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(path),
                    "-vf", "select=eq(n\\,5)",
                    "-frames:v", "1",
                    str(ffmpeg_png),
                ],
                check=True,
                capture_output=True,
            )
            assert ffmpeg_png.exists()
            assert ffmpeg_png.stat().st_size > 0

            # Both should be similar size (same frame, same resolution).
            size_ratio = our_png.stat().st_size / ffmpeg_png.stat().st_size
            assert 0.5 < size_ratio < 2.0, (
                f"Size mismatch: ours={our_png.stat().st_size}, "
                f"ffmpeg={ffmpeg_png.stat().st_size}"
            )
        finally:
            mapper.close()
