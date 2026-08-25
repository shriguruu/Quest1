import json
import os
import sys
from pathlib import Path
from typing import Annotated, Optional

import structlog
import typer

from dialogue_locator.config import settings
from dialogue_locator.logging_setup import setup_logging
from dialogue_locator.models import Candidate, LocateResult, ResultStatus
from dialogue_locator.stages.framemap import format_timestamp

app = typer.Typer(add_completion=False)
logger = structlog.get_logger()


@app.callback()
def callback() -> None:
    """dialogue-locator: find the video frame where a line is spoken."""
    pass


class ExitCode:
    CONFIDENT = 0
    ERROR = 1
    UNCERTAIN = 2
    NOT_FOUND = 3


def _resolve_video(
    url: str | None,
    file: Path | None,
    cache_dir: Path,
    no_cache: bool,
    cache_video: Path | None,
) -> tuple[Path, str]:
    """Return (video_path, url_for_metadata) from either --url or --file.

    Exactly one of url / file must be provided (enforced by caller).
    """
    from dialogue_locator.stages.acquire import (
        acquire,
        cache_video as _cache_video,
    )

    if file is not None:
        if not file.is_file():
            logger.error("File not found", path=str(file))
            raise typer.Exit(code=ExitCode.ERROR)
        logger.info("Using local file", path=str(file))
        return file, f"file://{file.absolute()}"

    assert url is not None
    video_path = acquire(url, cache_dir, force=no_cache)

    if cache_video is not None:
        _cache_video(video_path, cache_video)

    return video_path, url


@app.command()
def locate(
    url: Annotated[
        Optional[str], typer.Option("--url", help="Video URL to download")
    ] = None,
    file: Annotated[
        Optional[Path], typer.Option("--file", help="Path to a local video file")
    ] = None,
    query: Annotated[str, typer.Option(help="Target line of dialogue")] = ...,
    output_dir: Annotated[
        Optional[Path], typer.Option("--output-dir", help="Output directory")
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Output JSON instead of text")
    ] = False,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Disable caching / force re-run")
    ] = False,
) -> None:
    """Find the exact video frame where a line of dialogue is spoken."""
    if json_output:
        os.environ["LOG_JSON"] = "1"

    setup_logging()

    if url is None and file is None:
        logger.error("Provide either --url or --file")
        raise typer.Exit(code=ExitCode.ERROR)
    if url is not None and file is not None:
        logger.error("--url and --file are mutually exclusive")
        raise typer.Exit(code=ExitCode.ERROR)
    if file is not None and not file.is_file():
        logger.error("File not found", path=str(file))
        raise typer.Exit(code=ExitCode.ERROR)

    from dialogue_locator.pipeline import locate as run_locate

    out_dir = output_dir or Path(settings.output_dir)

    result = run_locate(
        url or "",
        query,
        settings,
        video_path=file,
        output_dir=out_dir,
        force=no_cache,
    )

    # The JSON artefact is written whether or not it was asked for, so a
    # run is always reproducible after the fact.
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    try:
        result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    except OSError as e:
        logger.warning("Could not write result.json", error=str(e))

    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        _print_report(result, result_path)

    raise typer.Exit(code=_exit_code(result.status))


def _exit_code(status: ResultStatus) -> int:
    return {
        ResultStatus.CONFIDENT: ExitCode.CONFIDENT,
        ResultStatus.UNCERTAIN: ExitCode.UNCERTAIN,
        ResultStatus.NOT_FOUND: ExitCode.NOT_FOUND,
        ResultStatus.ERROR: ExitCode.ERROR,
    }[status]


def _print_report(result: LocateResult, result_path: Path) -> None:
    """Human-readable summary on stdout."""
    if result.status is ResultStatus.ERROR:
        typer.echo("Status    : ERROR")
        for warning in result.warnings:
            typer.echo(f"  ! {warning}")
        typer.echo(f"Result    : {result_path}")
        return

    if result.status is ResultStatus.NOT_FOUND:
        typer.echo("Status    : NOT_FOUND")
        typer.echo("Text      : (query not found in this media)")
        if result.alternates:
            typer.echo("")
            typer.echo("Closest near-misses:")
            _print_alternates(result.alternates)
        for warning in result.warnings:
            typer.echo(f"  ! {warning}")
        typer.echo(f"Result    : {result_path}")
        return

    frame = result.frame
    media = result.media
    assert frame is not None and media is not None

    fps_note = f"{media.fps} fps, {'VFR' if media.is_vfr else 'CFR'}"

    typer.echo(f"Timestamp : {result.timestamp}")
    typer.echo(f"Frame     : {frame.frame_index}  ({fps_note})")
    typer.echo(f'Text      : "{result.text}"')
    typer.echo(f"Modality  : {result.modality}")
    typer.echo(f"Precision : {_precision_of(result)}")
    typer.echo(
        f"Status    : {result.status} "
        f"(score {result.confidence:.1f}, tier={_tier_of(result)})"
    )
    if frame.image_path is not None:
        typer.echo(f"Image     : {frame.image_path}")
    typer.echo(f"Result    : {result_path}")

    if result.status is ResultStatus.UNCERTAIN and result.alternates:
        typer.echo("")
        typer.echo("This result is UNCERTAIN. Other candidates:")
        _print_alternates(result.alternates)

    for warning in result.warnings:
        typer.echo(f"  ! {warning}")


def _precision_of(result: LocateResult) -> str:
    """Describe the onset precision and which method produced it."""
    alignment = result.alignment
    if alignment is None:
        return "unknown"
    method = (
        "forced alignment"
        if alignment.method == "forced"
        else "whisper timestamps"
    )
    return f"+/-{alignment.uncertainty_s * 1000:.0f}ms ({method})"


def _tier_of(result: LocateResult) -> str:
    """Winning tier, lowercased for display."""
    return str(result.matched_tier).lower() if result.matched_tier else "unknown"


def _print_alternates(alternates: list[Candidate]) -> None:
    for i, candidate in enumerate(alternates[:3], start=1):
        typer.echo(
            f"  {i}. {format_timestamp(candidate.coarse_start_s)}  "
            f"score {candidate.score:>6.2f}  "
            f"tier={str(candidate.tier).lower():9} "
            f'"{candidate.matched_text}"'
        )



@app.command(name="transcribe")
def transcribe_cmd(
    url: Annotated[
        Optional[str], typer.Option("--url", help="Video URL to download")
    ] = None,
    file: Annotated[
        Optional[Path], typer.Option("--file", help="Path to a local video file")
    ] = None,
    out: Annotated[
        Optional[Path],
        typer.Option("--out", help="Write JSONL here instead of stdout"),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit logs as JSON")
    ] = False,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Force re-download and re-transcription"),
    ] = False,
    cache_video: Annotated[
        Optional[Path],
        typer.Option("--cache-video", help="Save downloaded video to this path"),
    ] = None,
) -> None:
    """Debug: dump the flat word stream as JSONL, one word per line."""
    if json_output:
        os.environ["LOG_JSON"] = "1"

    setup_logging()

    if url is None and file is None:
        logger.error("Provide either --url or --file")
        raise typer.Exit(code=ExitCode.ERROR)
    if url is not None and file is not None:
        logger.error("--url and --file are mutually exclusive")
        raise typer.Exit(code=ExitCode.ERROR)

    from dialogue_locator.stages.acquire import extract_audio, probe as probe_media
    from dialogue_locator.stages.transcribe import transcribe as run_transcribe

    cache_dir = Path(settings.cache_dir)

    try:
        video_path, url_meta = _resolve_video(
            url, file, cache_dir, no_cache, cache_video,
        )
        media = probe_media(video_path, url=url_meta)

        audio_path = cache_dir / f"{media.video_sha256}.wav"
        if not audio_path.is_file() or no_cache:
            logger.info("Extracting audio", path=str(audio_path))
            extract_audio(video_path, audio_path)

        transcript = run_transcribe(
            audio_path, settings, cache_dir, force=no_cache
        )

        lines = [
            json.dumps(
                {
                    "i": i,
                    "text": w.text,
                    "start_s": round(w.start_s, 3),
                    "end_s": round(w.end_s, 3),
                    "probability": (
                        round(w.probability, 4) if w.probability is not None else None
                    ),
                    "segment_id": w.segment_id,
                }
            )
            for i, w in enumerate(transcript.words)
        ]

        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("\n".join(lines) + "\n", encoding="utf-8")
            logger.info("Wrote JSONL", path=str(out), words=len(lines))
        else:
            for line in lines:
                typer.echo(line)

        raise typer.Exit(code=ExitCode.CONFIDENT)
    except typer.Exit:
        raise
    except Exception as e:
        logger.error("Transcribe command failed", error=str(e))
        raise typer.Exit(code=ExitCode.ERROR)


@app.command()
def probe(
    url: Annotated[
        Optional[str], typer.Option("--url", help="Video URL to download")
    ] = None,
    file: Annotated[
        Optional[Path], typer.Option("--file", help="Path to a local video file")
    ] = None,
    output_dir: Annotated[
        Optional[Path], typer.Option("--output-dir", help="Output directory")
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Output JSON instead of text")
    ] = False,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Disable caching / force re-download")
    ] = False,
    cache_video: Annotated[
        Optional[Path],
        typer.Option("--cache-video", help="Save downloaded video to this path for later --file reuse"),
    ] = None,
) -> None:
    """Download (if needed) and probe a video, printing MediaInfo as JSON."""
    if json_output:
        os.environ["LOG_JSON"] = "1"

    setup_logging()

    # --- Mutual-exclusivity check ---
    if url is None and file is None:
        logger.error("Provide either --url or --file")
        raise typer.Exit(code=ExitCode.ERROR)
    if url is not None and file is not None:
        logger.error("--url and --file are mutually exclusive")
        raise typer.Exit(code=ExitCode.ERROR)

    if output_dir:
        settings.output_dir = str(output_dir)

    from dialogue_locator.stages.acquire import probe as probe_media

    try:
        video_path, url_meta = _resolve_video(
            url, file, Path(settings.cache_dir), no_cache, cache_video,
        )
        media_info = probe_media(video_path, url=url_meta)
        typer.echo(media_info.model_dump_json(indent=2))
        raise typer.Exit(code=ExitCode.CONFIDENT)
    except typer.Exit:
        raise
    except Exception as e:
        logger.error("Probe command failed", error=str(e))
        raise typer.Exit(code=ExitCode.ERROR)


if __name__ == "__main__":
    app()
