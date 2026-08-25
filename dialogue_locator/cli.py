import os
import sys
from pathlib import Path
from typing import Annotated, Optional

import structlog
import typer

from dialogue_locator.config import settings
from dialogue_locator.logging_setup import setup_logging

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
        bool, typer.Option("--no-cache", help="Disable caching / force re-download")
    ] = False,
    modality: Annotated[
        str, typer.Option("--modality", help="Modality: auto, audio, or visual")
    ] = "auto",
    keep_video: Annotated[
        bool, typer.Option("--keep-video", help="Keep downloaded video after run")
    ] = False,
    cache_video: Annotated[
        Optional[Path],
        typer.Option("--cache-video", help="Save downloaded video to this path for later --file reuse"),
    ] = None,
) -> None:
    """Find the exact video frame where a line of dialogue is spoken."""
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

    try:
        video_path, url_meta = _resolve_video(
            url, file, Path(settings.cache_dir), no_cache, cache_video,
        )
    except typer.Exit:
        raise
    except Exception as e:
        logger.error("Acquisition failed", error=str(e))
        raise typer.Exit(code=ExitCode.ERROR)

    logger.info(
        "Starting locate",
        video=str(video_path),
        query=query,
        config=settings.model_dump(),
    )

    # Pipeline stages go here in future prompts.
    typer.echo(f"Config: {settings.model_dump()}")

    raise typer.Exit(code=ExitCode.CONFIDENT)


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
