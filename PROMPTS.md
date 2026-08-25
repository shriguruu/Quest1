# Prompts

this is the problem statement that is given to me which i need to solve in 2days time, i need to solve this using the criteria given in the doc. how do i go about planning for this project. what do i need to consider? what tech stack do i use? what sort of architecture should i use so that i can explain clearly to the interviewers what i have done... so i have some ideas in mind:
1) we can parse the audio, convert to text and then find the given dialogue in the text to find out which part of the video the dialogue occurs in
2) if transcript is available then we can use that to find which part of the video the dialogue occurs in
3) if subtitles are part of the video, then we have to do an ocr kind of thing to find out the transcript and then find out which part of the video its part of

and there might be more ways that i might have missed, so comprehensively evaluate all the different methods that can be used to solve the problem and tell whats the best way. lets go step by step, once we have figured out the way then we will move ahead to the tech stack

#### next prompt:
Set up a Python 3.11 project called `dialogue-locator`. It's a CLI tool that,
given a video URL and a target line of dialogue, finds the exact video frame
where that line is first spoken.

Do NOT implement any pipeline logic yet. This prompt is only project structure,
data models, config, and a CLI skeleton.

Structure:

dialogue_locator/
  __init__.py
  models.py        # pydantic v2 models — the contracts every stage uses
  config.py        # pydantic-settings, env-overridable
  errors.py        # exception hierarchy
  logging_setup.py # structlog, human-readable to stderr, JSON when LOG_JSON=1
  cli.py           # typer entrypoint
  stages/
    __init__.py    # empty placeholder modules for: acquire, framemap,
                   # transcribe, match, align, visual
tests/
pyproject.toml     # managed by uv
README.md
PROMPTS.md         # empty, I'll fill it in

models.py — define exactly these, all frozen where sensible:

  class MediaInfo:
      url: str
      video_path: Path
      audio_path: Path | None
      duration_s: float
      fps: Fraction          # exact rational, e.g. 30000/1001 — NOT a float
      is_vfr: bool
      total_frames: int | None
      width: int
      height: int
      video_start_time_s: float
      audio_start_time_s: float
      av_offset_s: float     # audio_start - video_start
      video_sha256: str

  class Word:
      text: str
      start_s: float
      end_s: float
      probability: float | None
      segment_id: int

  class Transcript:
      words: list[Word]
      language: str
      model_name: str
      source: Literal["asr", "sidecar_subs", "ocr"]

  class MatchTier(StrEnum):  EXACT | FUZZY | PHONETIC | SEMANTIC

  class Candidate:
      word_index_start: int
      word_index_end: int
      matched_text: str
      score: float           # 0..100
      tier: MatchTier
      coarse_start_s: float

  class AlignmentResult:
      onset_s: float
      offset_s: float
      method: Literal["forced", "whisper_fallback"]
      uncertainty_s: float
      vad_confirmed: bool
      per_word: list[Word]

  class FrameRef:
      frame_index: int
      pts_s: float
      image_path: Path | None

  class ResultStatus(StrEnum): CONFIDENT | UNCERTAIN | NOT_FOUND | ERROR

  class LocateResult:
      status: ResultStatus
      timestamp: str | None       # HH:MM:SS.mmm
      frame: FrameRef | None
      text: str | None
      confidence: float
      modality: Literal["audio", "visual"] | None
      alternates: list[Candidate]
      warnings: list[str]
      media: MediaInfo
      timings_ms: dict[str, float]

config.py — Settings with: whisper_model="large-v3",
whisper_compute_type="int8_float16", device="cuda", cache_dir=".cache",
output_dir="output", confident_threshold=85.0, uncertain_threshold=65.0,
align_window_s=3.0, ocr_sample_fps=2.0.

errors.py — LocatorError base; then AcquisitionError, ProbeError,
TranscriptionError, AlignmentError, FrameExtractionError.

cli.py — typer app with one command:
  locate --url TEXT --query TEXT [--output-dir PATH] [--json] [--no-cache]
     [--modality auto|audio|visual] [--keep-video]
Right now it should just parse args, set up logging, and print the parsed
config. Exit codes to reserve: 0=CONFIDENT, 2=UNCERTAIN, 3=NOT_FOUND, 1=ERROR.

Constraints:
- pydantic v2 syntax, full type hints everywhere.
- fps must be a Fraction, never a float — 29.97 is a lie and it accumulates.
- No business logic. Stage modules stay empty.

Verify: `uv run dialogue-locator locate --url x --query y` runs and prints config.

#### next prompt:
Implement `dialogue_locator/stages/acquire.py`.

Two public functions:

  def acquire(url: str, cache_dir: Path, force: bool = False) -> Path
  def probe(video_path: Path) -> MediaInfo

acquire():
- Download with yt-dlp's Python API (not subprocess).
- Format selection: prefer a progressive mp4 with audio; fall back to bestvideo
  +bestaudio merged. Log the exact selected format_id — the frame number we
  report is a property of this rendition and must be reproducible.
- Cache key = sha256(url + format_id). Skip download if present and not force.
- Also attempt `writesubtitles`/`writeautomaticsub` to .srt if available.
  Don't fail if absent — log and continue. We'll use them later as an
  optional accelerator only.
- Wrap yt-dlp failures in AcquisitionError with the original message.

probe():
- Use `ffprobe -v quiet -print_format json -show_streams -show_format`
  via subprocess, parse JSON.
- fps: parse `r_frame_rate` as "num/den" into a Fraction. Never float().
- VFR detection: compare `r_frame_rate` to `avg_frame_rate`. If they differ by
  more than 0.1%, set is_vfr=True and add a warning. Everything downstream
  branches on this.
- Read `start_time` from the VIDEO stream and the AUDIO stream separately and
  compute av_offset_s = audio_start - video_start. This is usually small but
  nonzero, and ignoring it puts every answer off by a frame or two.
- total_frames from `nb_frames` if present, else None (don't guess).
- video_sha256: hash the first and last 1MB plus file size, not the whole file.

Third function:

  def extract_audio(video_path: Path, out_path: Path) -> Path

- ffmpeg → 16kHz mono PCM s16le WAV. Explicitly `-vn`.
- Use `-af aresample=async=1:first_pts=0` so the WAV starts at t=0 with no
  padding or drift.
- Document in a docstring that WAV t=0 corresponds to video time
  (audio_start_time_s), and that callers must add av_offset_s when converting
  an audio timestamp to a video timestamp.

Add a debug CLI command `probe --url TEXT` that runs acquire+probe and pretty-
prints MediaInfo, so I can inspect any video before running the full pipeline.

Constraints: no ASR, no frame extraction. Just acquisition and metadata.

#### next prompt:
Implement `dialogue_locator/stages/framemap.py`. This converts a timestamp
into an exact frame index and extracts that frame as a PNG.

Public API:

  class FrameMapper:
      def __init__(self, media: MediaInfo)
      def time_to_frame(self, video_time_s: float) -> FrameRef
      def audio_time_to_frame(self, audio_time_s: float) -> FrameRef
      def extract(self, frame_ref: FrameRef, out_path: Path) -> Path
      def close(self) -> None

Semantics — get these exactly right, they are the whole point of this module:

1. The frame "on screen" at time t is the LAST frame whose PTS <= t.
   Use floor, never round. Rounding to nearest is wrong roughly half the time.

2. audio_time_to_frame() must add media.av_offset_s before mapping. Make this
   the only place that correction is applied, and assert it isn't double-applied.

3. CFR path: frame_index = floor(video_time_s * fps) using Fraction arithmetic
   throughout, converting to int only at the end. No float multiplication.

4. VFR path: open with PyAV, seek to just before the target, decode forward
   accumulating real frame.pts * time_base, return the last frame whose pts <= t.
   Cache decoded PTS values so repeated lookups nearby are cheap.

5. extract(): use PyAV, NOT `ffmpeg -ss`. ffmpeg's fast seek lands on the
   nearest keyframe and can be several frames off. Seek to the keyframe before
   the target, decode forward, and pick the frame whose PTS matches. Assert the
   decoded frame's PTS matches frame_ref.pts_s within half a frame duration;
   raise FrameExtractionError if not.

6. Guard against out-of-range times (negative, or beyond duration).

Also add `format_timestamp(seconds: float) -> str` producing HH:MM:SS.mmm.

Now write tests/test_framemap.py:

- Generate synthetic fixture videos with ffmpeg at 25fps, 30fps, and 30000/1001
  fps, 10 seconds each, using drawtext to burn the frame number `%{n}` into
  every frame:
    ffmpeg -f lavfi -i testsrc=duration=10:size=640x360:rate=25 \
      -vf "drawtext=text='%{n}':fontsize=96:x=20:y=20:fontcolor=white" ...
- For each: assert time_to_frame() returns the frame index that ffmpeg burned
  in. Verify by extracting the PNG and reading the number (crop + OCR, or
  simpler: compare against a frame extracted by ffmpeg's own select filter).
- Boundary tests: exactly on a frame PTS, one microsecond before, one after.
- Test that 29.97fps doesn't drift at t=600s — the classic float-fps bug.
- Test av_offset is applied once and only once.

Constraint: this module knows nothing about audio, ASR, or matching.

#### next prompt:
Implement `dialogue_locator/stages/transcribe.py`.

  def transcribe(audio_path: Path, settings: Settings,
                 cache_dir: Path, force: bool = False) -> Transcript

Use faster-whisper (CTranslate2):
- model_size = settings.whisper_model ("large-v3")
- compute_type = "int8_float16", device="cuda"
- word_timestamps=True   (non-negotiable, the whole pipeline depends on it)
- vad_filter=True with Silero VAD, min_silence_duration_ms=500
- language="en", task="transcribe"
- condition_on_previous_text=False — reduces hallucination loops on long media
- beam_size=5

Flatten segments into ONE continuous list[Word] across the whole video, with
segment_id recorded on each word. This matters: Whisper's segment boundaries
are arbitrary, and a target phrase frequently straddles two of them. All
downstream matching happens on the flat stream, never on segments.

Caching (critical for my iteration speed — ASR on a 90-min video is slow):
- Cache key = sha256(audio file) + model name + relevant params.
- Serialize Transcript to JSON in cache_dir. On cache hit, skip the model
  entirely. `--no-cache` forces re-run.

VRAM (I have 8GB):
- Load the model lazily inside the function.
- Explicitly `del model; gc.collect(); torch.cuda.empty_cache()` before
  returning. Later stages load their own models and must not OOM.
- If CUDA init fails, log a warning and fall back to CPU with compute_type
  "int8" rather than crashing.

Progress: log every ~30s of processed audio with the wall-clock rate.

Also add a debug CLI command `transcribe --url TEXT` that dumps the flat word
list as JSONL (one word per line with times) so I can eyeball it.

Constraints: no matching logic here. Output is a Transcript and nothing else.

#### next prompt:
Implement `dialogue_locator/stages/match.py`. This finds where a target phrase
occurs in a Transcript. Pure functions, no I/O, no models.

  def normalize(text: str) -> str
  def build_search_index(words: list[Word]) -> SearchIndex
  def find_candidates(index: SearchIndex, query: str,
                      top_k: int = 5) -> list[Candidate]

normalize(): casefold, strip all punctuation, expand common contractions
(don't→do not, it's→it is), collapse whitespace, normalize unicode to NFKC,
convert digits to words ("3" → "three") since Whisper is inconsistent there.

SearchIndex: one normalized string of the entire transcript, PLUS a mapping
from every character offset back to its source word index. Build the mapping
as you concatenate — do not try to reconstruct it afterward.

find_candidates() runs a tiered cascade, stopping at the first tier that
produces a hit above threshold:

  TIER 1 EXACT — normalized substring search. Score 100.

  TIER 2 FUZZY — rapidfuzz.fuzz.partial_ratio_alignment() of the normalized
    query against the normalized transcript. It returns src_start/src_end
    character offsets — use those with the char→word map to recover the exact
    first word. Accept score >= 65.
    Also run a sliding window of length ~len(query_words) ± 3 words and keep
    local maxima, so we find ALL occurrences, not just the global best.

  TIER 3 PHONETIC — encode query and transcript token streams with Double
    Metaphone (jellyfish), fuzzy-match the phoneme streams. Catches Whisper
    homophone errors like "rebels at" → "rebel that". Cap score at 85 since
    phonetic matches are inherently weaker evidence.

Every Candidate records which tier produced it. Return top_k sorted by score
descending, then by time ascending. Deduplicate candidates whose word spans
overlap by more than 50%, keeping the higher score.

Add:
  def classify(candidates, settings) -> tuple[ResultStatus, list[str]]
returning CONFIDENT (top >= 85), UNCERTAIN (65..85, or top two scores within
5 points of each other = genuinely ambiguous), or NOT_FOUND, plus warning
strings explaining the classification.

tests/test_match.py — build synthetic Transcripts by hand (no video needed):
- exact match
- Whisper dropped a word / added a filler word
- phrase straddles two segments  ← the important one
- phrase occurs three times → all three returned in time order
- homophone substitution caught only by phonetic tier
- query absent entirely → NOT_FOUND
- two near-identical passages → UNCERTAIN, not a confident guess

Constraint: no semantic/embedding tier yet. English only for now.

#### next prompt:
Implement `dialogue_locator/pipeline.py` and wire up the real `locate` CLI
command. This produces a complete working system using Whisper's word
timestamps directly — forced alignment comes next and will slot in later.

  def locate(url: str, query: str, settings: Settings) -> LocateResult

Flow:
  1. acquire(url) → probe() → MediaInfo
  2. extract_audio()
  3. transcribe() → Transcript
  4. build_search_index() → find_candidates() → classify()
  5. If NOT_FOUND: return early with alternates populated (best near-misses,
     so the user sees what it *did* find) and status NOT_FOUND.
  6. Take top candidate. coarse_start_s = its first word's start_s.
     Wrap it in an AlignmentResult with method="whisper_fallback" and
     uncertainty_s=0.3 — be explicit that this is the imprecise path.
  7. FrameMapper.audio_time_to_frame(onset) → FrameRef
  8. Extract PNG to output_dir/frame_<index>.png
  9. Build LocateResult with timings for each stage.

Output:
- Human-readable to stdout:
      Timestamp : 01:10:31.147
      Frame     : 105779  (25/1 fps, CFR)
      Text      : "My mind rebels at stagnation"
      Modality  : audio
      Precision : ±300ms (whisper timestamps)
      Status    : CONFIDENT (score 94.2, tier=fuzzy)
      Image     : output/frame_105779.png
- `--json` emits LocateResult.model_dump_json(indent=2).
- Always write result.json to output_dir regardless of the flag.
- If status is UNCERTAIN, print the top 3 alternates with their timestamps
  and scores. Never silently pick one.
- Exit codes: 0 CONFIDENT, 2 UNCERTAIN, 3 NOT_FOUND, 1 ERROR.

Wrap each stage in a timer and populate timings_ms. Catch LocatorError
subclasses at the top level and emit a structured ERROR result rather than
a traceback.

Verify end-to-end on a short clip before I move on.

#### next prompt:
Implement `dialogue_locator/stages/align.py` and slot it into the pipeline,
replacing the whisper_fallback path.

  def align(audio_path: Path, media: MediaInfo, candidate: Candidate,
            transcript: Transcript, settings: Settings) -> AlignmentResult

Why this stage exists: Whisper's word timestamps come from cross-attention
DTW and are ±100-300ms, which is 3-8 frames. Forced alignment solves a much
easier problem — the text is already known, so the model only has to locate
known words in known audio — and resolves to ~20ms, under one frame.

Implementation:
- torchaudio's wav2vec2 CTC forced alignment API (or whisperx.align if
  simpler — your call, but justify it in a module docstring).
- Model: WAV2VEC2_ASR_BASE_960H (English).
- Slice audio from (candidate.coarse_start_s - settings.align_window_s) to
  (candidate_end + align_window_s). Clamp to media bounds. Aligning the whole
  file is unnecessary and slow.
- Align the *transcript's* text for that span (not my raw query) — the query
  may differ slightly from what was actually said, and aligning text that
  isn't in the audio produces garbage.
- Return the onset of the FIRST word, plus per-word timings.
- uncertainty_s = 0.02 for successful forced alignment.

Sanity checks — if any fail, fall back to whisper timestamps with a warning
in the result rather than returning a confidently wrong answer:
- Run Silero VAD on the window. If the onset lands inside detected silence,
  alignment drifted → set vad_confirmed=False, widen the window by 2x,
  retry once, then fall back.
- If aligned onset is more than 2.0s from Whisper's estimate, distrust it.
- If the CTC path has very low likelihood, distrust it.

VRAM: load wav2vec2 only here, free it before returning. Whisper must already
be unloaded by this point — assert that if you can.

Update pipeline.py: use align() when it succeeds, whisper_fallback when it
doesn't, and surface which one was used in both the human output ("Precision:
±20ms (forced alignment)") and the JSON.

tests/test_align.py: synthesize a WAV with a known word starting at exactly
2.500s (TTS or a spliced clip), assert onset is within 30ms.