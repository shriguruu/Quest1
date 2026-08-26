import hashlib
import json
import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import structlog
import yt_dlp
from yt_dlp.networking.impersonate import ImpersonateTarget

from dialogue_locator.errors import AcquisitionError, ProbeError
from dialogue_locator.models import MediaInfo

logger = structlog.get_logger()

_HINT = (
    "\n\nHint: If the site blocks automated downloads, you can download the "
    "video manually (e.g. with a browser or VPN) and pass it with:\n"
    "  dialogue-locator locate --file /path/to/video.mp4 --query \"...\"\n"
    "For YouTube specifically, passing your browser's cookies often helps:\n"
    "  set LOCATOR_COOKIES_FROM_BROWSER=chrome"
)

# YouTube gates its default ("web") player client behind bot detection that
# routinely rejects datacentre and some residential IPs, reporting the
# thoroughly misleading "This video is not available" for videos that are
# public and fine.  The mobile and embedded clients use a different endpoint
# that is not gated the same way, so we hand yt-dlp an ordered list and let
# it fall through until one answers.  This is YouTube-specific and ignored
# by every other extractor.
_YOUTUBE_PLAYER_CLIENTS = ["default", "android", "ios", "tv_embedded", "web_safari"]


def _compute_sha256(path: Path) -> str:
    """Hash first 1MB + last 1MB + file size for fast identification."""
    size = path.stat().st_size
    chunk_size = 1024 * 1024
    hasher = hashlib.sha256()

    with path.open("rb") as f:
        hasher.update(f.read(chunk_size))

        if size > chunk_size:
            f.seek(max(chunk_size, size - chunk_size))
            hasher.update(f.read(chunk_size))

    hasher.update(str(size).encode("utf-8"))
    return hasher.hexdigest()


def _try_impersonate() -> ImpersonateTarget | None:
    """Return an ImpersonateTarget if curl_cffi is available, else None."""
    try:
        import curl_cffi  # noqa: F401
        return ImpersonateTarget("chrome")
    except ImportError:
        return None


def _url_index_path(cache_dir: Path, url: str) -> Path:
    """Where the URL -> downloaded-file pointer for *url* lives."""
    url_key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{url_key}.source.json"


def _read_url_index(cache_dir: Path, url: str) -> Path | None:
    """Return the cached file for *url*, if the pointer and file both exist."""
    index_path = _url_index_path(cache_dir, url)
    if not index_path.is_file():
        return None
    try:
        record = json.loads(index_path.read_text(encoding="utf-8"))
        cached = Path(record["path"])
    except (json.JSONDecodeError, OSError, KeyError, TypeError):
        logger.warning("Discarding unreadable cache pointer", path=str(index_path))
        return None
    return cached if cached.is_file() else None


def _write_url_index(cache_dir: Path, url: str, format_id: str, path: Path) -> None:
    """Record where *url* was downloaded to, keyed by URL alone."""
    try:
        _url_index_path(cache_dir, url).write_text(
            json.dumps(
                {"url": url, "format_id": format_id, "path": str(path)}, indent=2
            ),
            encoding="utf-8",
        )
    except OSError as e:
        logger.warning("Could not write cache pointer", error=str(e))


def acquire(url: str, cache_dir: Path, force: bool = False) -> Path:
    """Download a video via yt-dlp.  Returns path to the downloaded file.

    Raises AcquisitionError on failure, with a hint to use --file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # The primary cache key includes format_id, which can only be learned by
    # asking the site -- so a naive cache check still needs the network, and
    # an already-downloaded video becomes unusable the moment the site is
    # unreachable.  A pointer keyed by URL alone removes that dependency:
    # once a video is on disk, later runs never touch the network at all.
    if not force:
        cached = _read_url_index(cache_dir, url)
        if cached is not None:
            logger.info("Cache hit (no network needed)", path=str(cached))
            return cached

    impersonate = _try_impersonate()

    ydl_opts: dict = {
        "format": "best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "quiet": True,
        "no_warnings": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitlesformat": "srt",
        "socket_timeout": 60,
        "retries": 3,
        "extractor_retries": 3,
        "extractor_args": {
            "youtube": {"player_client": _YOUTUBE_PLAYER_CLIENTS}
        },
    }

    if impersonate:
        ydl_opts["impersonate"] = impersonate
        logger.debug("Using curl_cffi impersonation", target="chrome")

    # Some videos (age-gated, members-only, or simply an IP YouTube dislikes)
    # need real browser cookies. Opt-in, since reading a browser profile is
    # not something to do silently.
    cookies_browser = os.environ.get("LOCATOR_COOKIES_FROM_BROWSER")
    if cookies_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_browser,)
        logger.info("Using browser cookies", browser=cookies_browser)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                raise AcquisitionError(
                    f"yt-dlp could not extract info for: {url}{_HINT}"
                )

            format_id = info.get("format_id", "unknown")
            logger.info("Selected format", format_id=format_id, url=url)

            key_str = f"{url}{format_id}".encode("utf-8")
            cache_key = hashlib.sha256(key_str).hexdigest()

            # Since yt-dlp might change extension upon merge, look for any
            # matching prefix (but skip subtitle files).
            existing_files = list(cache_dir.glob(f"{cache_key}.*"))
            video_files = [
                f for f in existing_files if f.suffix not in (".srt", ".vtt")
            ]

            if video_files and not force:
                out_path = video_files[0]
                logger.info("Cache hit", path=str(out_path))
                _write_url_index(cache_dir, url, format_id, out_path)
                return out_path

            logger.info("Downloading media")

            download_opts = dict(ydl_opts)
            download_opts["outtmpl"] = str(cache_dir / f"{cache_key}.%(ext)s")

            with yt_dlp.YoutubeDL(download_opts) as dl_ydl:
                dl_info = dl_ydl.extract_info(url, download=True)

            req_dl = dl_info.get("requested_downloads")
            if req_dl and len(req_dl) > 0:
                final_path = Path(req_dl[0].get("filepath"))
            else:
                final_path = Path(dl_ydl.prepare_filename(dl_info))

            _write_url_index(cache_dir, url, format_id, final_path)

            subs = list(cache_dir.glob(f"{cache_key}*.srt"))
            if not subs:
                logger.info("No subtitles found for this media")
            else:
                logger.info("Subtitles found", count=len(subs))

            return final_path

    except yt_dlp.utils.DownloadError as e:
        stale = _read_url_index(cache_dir, url)
        if stale is not None:
            logger.warning(
                "Site unreachable; using the previously downloaded copy",
                path=str(stale),
                error=str(e).splitlines()[0][:160],
            )
            return stale
        raise AcquisitionError(f"yt-dlp failed: {e}{_HINT}") from e
    except Exception as e:
        if isinstance(e, AcquisitionError):
            raise
        stale = _read_url_index(cache_dir, url)
        if stale is not None:
            logger.warning(
                "Acquisition failed; using the previously downloaded copy",
                path=str(stale),
                error=str(e)[:160],
            )
            return stale
        raise AcquisitionError(f"Acquisition failed: {e}{_HINT}") from e


def cache_video(src: Path, dest: Path) -> Path:
    """Copy a downloaded video to a stable location for later --file reuse."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    logger.info("Cached video", src=str(src), dest=str(dest))
    return dest


def probe(video_path: Path, url: str = "") -> MediaInfo:
    """Run ffprobe on a video file and return structured MediaInfo."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        raise ProbeError(f"ffprobe failed: {e.stderr}") from e

    video_streams = [
        s for s in data.get("streams", []) if s.get("codec_type") == "video"
    ]
    if not video_streams:
        raise ProbeError("No video stream found")
    v_stream = video_streams[0]

    audio_streams = [
        s for s in data.get("streams", []) if s.get("codec_type") == "audio"
    ]
    a_stream = audio_streams[0] if audio_streams else None

    # --- fps parsing (always exact Fraction, never float) ---
    r_fps_str = v_stream.get("r_frame_rate", "0/0")
    if "/" in r_fps_str:
        num, den = r_fps_str.split("/")
        fps = Fraction(int(num), int(den)) if int(den) != 0 else Fraction(0)
    else:
        fps = Fraction(int(r_fps_str), 1)

    avg_fps_str = v_stream.get("avg_frame_rate", "0/0")
    if "/" in avg_fps_str:
        num, den = avg_fps_str.split("/")
        avg_fps = Fraction(int(num), int(den)) if int(den) != 0 else Fraction(0)
    else:
        avg_fps = Fraction(int(avg_fps_str), 1)

    if fps == 0:
        raise ProbeError("Video stream has 0/0 fps")

    # --- VFR detection ---
    #
    # ffprobe's own avg_frame_rate is not a reliable VFR signal: it is
    # derived from the *container* duration, so a file whose audio track
    # simply runs longer than its video track reports an avg_frame_rate
    # well below r_frame_rate and looks variable when it isn't.  Compare
    # against the video stream's own nb_frames / duration instead, and
    # only fall back to avg_frame_rate when the stream doesn't report
    # both.
    is_vfr = False
    v_nb_frames = v_stream.get("nb_frames")
    v_duration = v_stream.get("duration")

    if v_nb_frames is not None and v_duration is not None and float(v_duration) > 0:
        measured_fps = Fraction(int(v_nb_frames)) / Fraction(v_duration)
        vfr_source = "stream nb_frames/duration"
    elif avg_fps > 0:
        measured_fps = avg_fps
        vfr_source = "avg_frame_rate"
    else:
        measured_fps = None
        vfr_source = "unavailable"

    if measured_fps is not None:
        diff = abs(float(fps) - float(measured_fps)) / float(fps)
        if diff > 0.001:
            is_vfr = True
            logger.warning(
                "VFR detected",
                r_frame_rate=str(fps),
                measured_frame_rate=str(measured_fps),
                source=vfr_source,
                diff=diff,
            )

    # --- Start times and A/V offset ---
    v_start = float(v_stream.get("start_time", 0.0))
    a_start = float(a_stream.get("start_time", 0.0)) if a_stream else 0.0
    av_offset_s = a_start - v_start

    nb_frames_str = v_stream.get("nb_frames")
    total_frames = int(nb_frames_str) if nb_frames_str is not None else None

    width = int(v_stream.get("width", 0))
    height = int(v_stream.get("height", 0))

    duration_str = data.get("format", {}).get("duration")
    if duration_str is None:
        duration_str = v_stream.get("duration", "0.0")
    duration_s = float(duration_str)

    sha256 = _compute_sha256(video_path)

    # Fallback for url if none provided
    if not url:
        url = f"file://{video_path.absolute()}"

    return MediaInfo(
        url=url,
        video_path=video_path,
        audio_path=None,
        duration_s=duration_s,
        fps=fps,
        is_vfr=is_vfr,
        total_frames=total_frames,
        width=width,
        height=height,
        video_start_time_s=v_start,
        audio_start_time_s=a_start,
        av_offset_s=av_offset_s,
        video_sha256=sha256,
    )


def extract_audio(video_path: Path, out_path: Path) -> Path:
    """Extract audio from a video file to a 16kHz mono PCM s16le WAV file.

    Note:
        The resulting WAV file starts at t=0 of the video. Thus, the WAV's t=0
        corresponds to the video's start_time (audio_start_time_s in MediaInfo).
        Callers must add av_offset_s when converting an audio timestamp to a
        video timestamp.
    """
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-af", "aresample=async=1:first_pts=0",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise AcquisitionError(f"Audio extraction failed: {e.stderr}") from e

    return out_path
