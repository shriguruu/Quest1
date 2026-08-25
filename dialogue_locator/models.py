from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class MediaInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    url: str
    video_path: Path
    audio_path: Path | None
    duration_s: float
    fps: Fraction
    is_vfr: bool
    total_frames: int | None
    width: int
    height: int
    video_start_time_s: float
    audio_start_time_s: float
    av_offset_s: float
    video_sha256: str


class Word(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    start_s: float
    end_s: float
    probability: float | None
    segment_id: int


class Transcript(BaseModel):
    model_config = ConfigDict(frozen=True)

    words: list[Word]
    language: str
    model_name: str
    source: Literal["asr", "sidecar_subs", "ocr"]


class MatchTier(StrEnum):
    EXACT = "EXACT"
    FUZZY = "FUZZY"
    PHONETIC = "PHONETIC"
    SEMANTIC = "SEMANTIC"


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    word_index_start: int
    word_index_end: int
    matched_text: str
    score: float
    tier: MatchTier
    coarse_start_s: float


class AlignmentResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    onset_s: float
    offset_s: float
    method: Literal["forced", "whisper_fallback"]
    uncertainty_s: float
    vad_confirmed: bool
    per_word: list[Word]
    # Why forced alignment was rejected, when it was. Surfaced on the
    # result so a fallback is never silent.
    warnings: list[str] = []


class FrameRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    frame_index: int
    pts_s: float
    image_path: Path | None


class ResultStatus(StrEnum):
    CONFIDENT = "CONFIDENT"
    UNCERTAIN = "UNCERTAIN"
    NOT_FOUND = "NOT_FOUND"
    ERROR = "ERROR"


class LocateResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: ResultStatus
    timestamp: str | None
    frame: FrameRef | None
    text: str | None
    confidence: float
    # Which tier produced the winning match, for display and for auditing
    # a result after the fact.
    matched_tier: MatchTier | None = None
    # How the onset was determined, and how much to trust it.
    alignment: AlignmentResult | None = None
    modality: Literal["audio", "visual"] | None
    alternates: list[Candidate]
    warnings: list[str]
    # Optional because an ERROR result may be produced before (or because)
    # probing succeeded -- a failed download has no MediaInfo to report.
    media: MediaInfo | None = None
    timings_ms: dict[str, float] = {}
