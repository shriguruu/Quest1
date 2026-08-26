# dialogue-locator

Given a video URL and a line of dialogue, find the **exact frame** where that
line begins.

```
Timestamp : 00:05:25.264
Frame     : 7798  (24000/1001 fps, VFR)
Text      : "My mind rebels at stagnation."
Modality  : audio
Precision : +/-20ms (forced alignment)
Status    : CONFIDENT (score 100.0, tier=exact)
Image     : output\frame_7798.png
```

No one has to watch the video first. See [DESIGN.md](DESIGN.md) for how it
works and why, and [PROMPTS.md](PROMPTS.md) for the prompts used to build it.

---

## Requirements

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** — the package manager this project uses
- **ffmpeg and ffprobe** on your `PATH`

Check ffmpeg first, since everything depends on it:

```bash
ffmpeg -version
ffprobe -version
```

<details>
<summary>Installing ffmpeg</summary>

| Platform | Command |
|---|---|
| Windows | `winget install Gyan.FFmpeg` |
| macOS | `brew install ffmpeg` |
| Debian/Ubuntu | `sudo apt install ffmpeg` |

</details>

---

## Setup

```bash
git clone <your-repo-url>
cd Quest1
uv sync
```

That installs everything, including a **CPU-only** build of PyTorch. This is
deliberate — `pyproject.toml` pins torch to PyTorch's CPU index so `uv sync`
cannot silently pull the 2–3GB CUDA build. Total download is roughly 400MB.

Two models download automatically on first use and are then cached in
`~/.cache`:

| Model | Size | Used for |
|---|---|---|
| Whisper (`faster-whisper`) | 150MB – 3GB, depending on size | Transcription |
| `WAV2VEC2_ASR_BASE_960H` | 360MB | Forced alignment |

Verify the install:

```bash
uv run pytest -q          # 99 tests, no network needed
uv run dialogue-locator --help
```

---

## Choosing a Whisper model

**This is the one setting that matters.** It is the difference between a
70-second run and a 54-minute one.

The default is `large-v3`, which assumes a working CUDA GPU. If CUDA is
unavailable the tool falls back to CPU automatically — correct, but *slow*, at
roughly realtime. On CPU, use a smaller model:

```bash
# Windows PowerShell
$env:LOCATOR_WHISPER_MODEL = "base.en"

# macOS / Linux
export LOCATOR_WHISPER_MODEL=base.en
```

Measured on a 54-minute video, CPU:

| Model | Transcription time | Notes |
|---|---|---|
| `base.en` | **~70s** (46x realtime) | Found the target line exactly. Recommended on CPU |
| `small.en` | ~4 min | More accurate on hard audio |
| `large-v3` | ~54 min | Only sensible with a working GPU |

Use the multilingual names (`base`, `small`, `large-v3`) for non-English media;
the `.en` variants are English-only but more accurate at these sizes.

---

## Usage

```bash
uv run dialogue-locator locate --url "<video-url>" --query "<line of dialogue>"
uv run dialogue-locator locate --file path/to/video.mp4 --query "<line>"
```

### Options

| Flag | Meaning |
|---|---|
| `--url TEXT` | Video URL to download (mutually exclusive with `--file`) |
| `--file PATH` | Local video file |
| `--query TEXT` | The line to find **(required)** |
| `--output-dir PATH` | Where the PNG and `result.json` go (default `output/`) |
| `--json` | Print the full `LocateResult` as JSON |
| `--no-cache` | Force re-download and re-transcription |

### Other commands

```bash
uv run dialogue-locator probe --url "<url>"          # container metadata as JSON
uv run dialogue-locator transcribe --file v.mp4 --out words.jsonl
```

`transcribe` dumps the flat word stream, one JSON object per line — useful for
seeing what the ASR actually heard when a query doesn't match.

---

## Output

Every run writes `result.json` to the output directory whether or not you passed
`--json`, alongside the extracted frame as `frame_<index>.png`. The JSON records
the full `MediaInfo`, the alignment method and its per-word timings, any
alternates, warnings, and per-stage timings.

### Exit codes

| Code | Status | Meaning |
|---|---|---|
| `0` | `CONFIDENT` | Match ≥85 and timing verified |
| `2` | `UNCERTAIN` | Match 65–85, or two candidates too close to separate |
| `3` | `NOT_FOUND` | Nothing above threshold — near-misses are printed |
| `1` | `ERROR` | Download, decode, or input failure |

`UNCERTAIN` prints the top three alternates with timestamps and scores rather
than silently picking one.

---

## Environment variables

All settings are overridable with a `LOCATOR_` prefix, or via a `.env` file.

| Variable | Default | Purpose |
|---|---|---|
| `LOCATOR_WHISPER_MODEL` | `large-v3` | Whisper model size |
| `LOCATOR_DEVICE` | `cuda` | `cuda` or `cpu` (falls back automatically) |
| `LOCATOR_LANGUAGE` | `en` | ASR language |
| `LOCATOR_CONFIDENT_THRESHOLD` | `85.0` | Score for `CONFIDENT` |
| `LOCATOR_UNCERTAIN_THRESHOLD` | `65.0` | Floor for a usable result |
| `LOCATOR_ALIGN_WINDOW_S` | `3.0` | Padding around the candidate for alignment |
| `LOCATOR_COOKIES_FROM_BROWSER` | unset | e.g. `chrome` — use browser cookies for gated videos |
| `LOCATOR_CACHE_DIR` | `.cache` | Downloads, audio, transcripts, PTS indexes |
| `LOCATOR_OUTPUT_DIR` | `output` | Frames and `result.json` |
| `LOG_JSON` | unset | Set to `1` for JSON logs on stderr |

---

## Caching

Everything expensive is cached in `.cache/` and keyed by content:

- **Video** by `sha256(url + format_id)`
- **Audio, transcripts, PTS indexes** by file/audio hash plus model parameters

A second run against the same media is near-instant. Delete `.cache/` or pass
`--no-cache` to start clean.

> `.cache/` holds the downloaded video and a WAV of its audio — hundreds of MB
> to a few GB. Keep it out of git.

---

## Troubleshooting

**`Library cublas64_12.dll is not found`** — CUDA runtime missing. Harmless: the
tool logs a warning and continues on CPU. Set `LOCATOR_WHISPER_MODEL=base.en` so
it isn't slow.

**`This video is not available` on a YouTube video that plainly is** — YouTube
gates its default player client behind bot detection that rejects many IPs, and
reports this misleading message for perfectly public videos. The tool already
falls through to the `android`, `ios`, and `tv_embedded` clients automatically,
which resolves it in most cases. If it still fails, supply browser cookies:

```powershell
$env:LOCATOR_COOKIES_FROM_BROWSER = "chrome"
```

**Download fails on another site** — some sites block automated downloads.
Download the video in a browser and use `--file` instead; the error message says
so and repeats the command for you.

**`NOT_FOUND` on a line you know is there** — read the near-misses it printed,
then run `transcribe` to see what the ASR actually heard. A larger model often
fixes it.

**Alignment fell back to `+/-300ms`** — the reason is printed. It means one of
the three trust checks failed; the answer is still usable, just less precise.
