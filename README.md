# Dialogue Locator

Given a video URL and a line of dialogue, find the **exact frame** where that
line begins — without anyone watching the video first.

```
Timestamp : 00:05:25.264
Frame     : 7798  (24000/1001 fps, VFR)
Text      : "My mind rebels at stagnation."
Modality  : audio
Precision : +/-20ms (forced alignment)
Status    : CONFIDENT (score 100.0, tier=exact)
Image     : output/frame_7798.png
```

---

## How it works

```
download → transcribe → search the text → pin the exact instant → map to a frame
```

1. **Transcribe once.** Whisper turns the whole audio track into a word-level
   transcript. Searching text is cheap; searching video is not.
2. **Find the line.** A tiered matcher (exact → fuzzy → phonetic) locates the
   phrase even when the ASR misheard it — which it usually does.
3. **Pin the instant.** Whisper's timestamps are only accurate to ±100–300ms
   (3–8 frames), so a second model does forced alignment on a few seconds of
   audio and narrows that to **±20ms**.
4. **Map to a frame.** Frame numbers come from the video's *measured*
   presentation timestamps, not `floor(t × fps)` — which on the test video is
   wrong by up to 221 frames.

Every answer carries a confidence and a stated precision. When the tool isn't
sure, it says so and shows its alternatives rather than guessing.

[DESIGN.md](DESIGN.md) explains the reasoning in full.

---

## Setup

No GPU required — everything runs on CPU by default.

### 1. Install ffmpeg

The tool shells out to `ffmpeg` and `ffprobe` for decoding, audio extraction,
and reading frame timestamps. Both must be on your `PATH`.

| Platform | Command |
|---|---|
| Windows | `winget install Gyan.FFmpeg` |
| macOS | `brew install ffmpeg` |
| Debian / Ubuntu | `sudo apt install ffmpeg` |

Verify before going further — this is the most common setup failure:

```bash
ffmpeg -version
ffprobe -version
```

If Windows still reports "not recognised" after installing, open a **new**
terminal so it picks up the updated `PATH`.

### 2. Install uv

[uv](https://docs.astral.sh/uv/) manages the virtualenv, the dependencies, and
even the Python version.

```powershell
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

You do **not** need to install Python yourself. The project requires 3.11+, and
if your system doesn't have it, uv downloads a suitable interpreter during the
next step.

### 3. Clone and install

```bash
git clone https://github.com/shriguruu/Quest1.git
cd Quest1
uv sync
```

`uv sync` creates `.venv/`, resolves everything from `uv.lock` (so you get the
exact versions this was tested against), and installs the package itself.

It pulls roughly **400MB**, including a **CPU-only** PyTorch build.
`pyproject.toml` deliberately pins torch to PyTorch's CPU index, so this can't
silently grab the 2–3GB CUDA wheel you don't need.

### 4. Verify

```bash
uv run pytest -q                    # 99 tests, no network or media needed
uv run dialogue-locator --help      # should list: locate, probe, transcribe
```

All 99 tests passing means the frame arithmetic, matching, and alignment logic
are sound on your machine.

### 5. First run

Two models download automatically the first time they're needed, then are
cached in `~/.cache` and never fetched again:

| Model | Size | Purpose |
|---|---|---|
| `faster-whisper-base.en` | 141MB | Transcription |
| `WAV2VEC2_ASR_BASE_960H` | 378MB | Forced alignment |

**Disk space:** budget ~1.5GB for the virtualenv and models, plus room in
`.cache/` for downloaded videos and their extracted audio — a 54-minute video
takes about 1.1GB. Both `.cache/` and `output/` are gitignored.

<details>
<summary>Setup problems</summary>

<br>

**`uv: command not found`** — restart your terminal after installing uv, or add
`~/.local/bin` (macOS/Linux) or `%USERPROFILE%\.local\bin` (Windows) to `PATH`.

**`ffprobe failed` or `No video stream found`** — ffmpeg isn't installed or
isn't on `PATH`. Re-check step 1 in a fresh terminal.

**`uv sync` fails resolving torch** — you're likely behind a proxy blocking
`download.pytorch.org`, which the CPU-only pin depends on.

**First run seems to hang** — it's downloading the 141MB and 378MB model files.
Progress goes to stderr; subsequent runs skip this entirely.

</details>

---

## Usage

```bash
uv run dialogue-locator locate --url "<video-url>" --query "<line of dialogue>"
uv run dialogue-locator locate --file video.mp4    --query "<line of dialogue>"
```

Working example:

```bash
uv run dialogue-locator locate \
  --url "https://ok.ru/video/248244667877" \
  --query "My mind rebels at stagnation"
```

First run on a 54-minute video takes ~2 minutes on CPU (mostly transcription);
repeat runs are seconds, since everything expensive is cached in `.cache/`.

| Flag | Meaning |
|---|---|
| `--url` / `--file` | Video source (one is required) |
| `--query TEXT` | The line to find **(required)** |
| `--output-dir PATH` | Where the PNG and `result.json` go (default `output/`) |
| `--json` | Print the full result as JSON |
| `--no-cache` | Force re-download and re-transcription |

Also: `probe` prints container metadata, and `transcribe --out words.jsonl`
dumps the word stream — handy for seeing what the ASR actually heard.

---

## Output

Each run writes `frame_<index>.png` and `result.json` to the output directory.
The JSON carries the media info, alignment method, per-word timings, alternates,
warnings, and per-stage timings.

| Exit | Status | Meaning |
|---|---|---|
| `0` | `CONFIDENT` | Strong match, timing verified |
| `2` | `UNCERTAIN` | Weak match, or several candidates too close to separate — top 3 are printed |
| `3` | `NOT_FOUND` | Nothing matched; closest near-misses are printed |
| `1` | `ERROR` | Download, decode, or input failure |

---

<details>
<summary><b>Configuration</b> — all settings override via <code>LOCATOR_</code> env vars or a <code>.env</code> file</summary>

<br>

| Variable | Default | Purpose |
|---|---|---|
| `LOCATOR_WHISPER_MODEL` | `base.en` | Whisper model size |
| `LOCATOR_DEVICE` | `cuda` | Falls back to CPU automatically |
| `LOCATOR_LANGUAGE` | `en` | ASR language |
| `LOCATOR_CONFIDENT_THRESHOLD` | `85.0` | Score required for `CONFIDENT` |
| `LOCATOR_UNCERTAIN_THRESHOLD` | `65.0` | Floor for a usable result |
| `LOCATOR_ALIGN_WINDOW_S` | `3.0` | Audio padding around the candidate |
| `LOCATOR_COOKIES_FROM_BROWSER` | unset | e.g. `chrome`, for gated videos |
| `LOCATOR_CACHE_DIR` | `.cache` | Downloads, audio, transcripts, PTS indexes |
| `LOCATOR_OUTPUT_DIR` | `output` | Frames and `result.json` |
| `LOG_JSON` | unset | `1` for JSON logs on stderr |

**Choosing a model.** `base.en` is the default because it runs at ~46x realtime
on CPU (70s for a 54-minute video) and the matcher is built to tolerate its
mistakes — it heard *"My mind rebels **its** stagnation"* and the fuzzy tier
still scored 94.7 and landed on the right frame. Step up to `small.en` (~4 min)
for accented or noisy audio. Drop the `.en` suffix for non-English media and set
`LOCATOR_LANGUAGE`.

</details>

<details>
<summary><b>Troubleshooting</b></summary>

<br>

**`Library cublas64_12.dll is not found`** — CUDA runtime missing. Harmless; it
logs a warning and continues on CPU, which is the default path anyway.

**`This video is not available` on a YouTube video that plainly is** — YouTube's
bot detection rejecting your IP, not a real availability problem. The tool
already falls through to the `android`, `ios`, and `tv_embedded` clients, which
fixes most cases. If not, set `LOCATOR_COOKIES_FROM_BROWSER=chrome`.

**Connection reset / download fails** — some networks throttle or block certain
hosts. If the video was fetched successfully before, the cache is used
automatically with no network access. Otherwise download it in a browser and use
`--file`.

**`NOT_FOUND` on a line you know is there** — read the near-misses it printed,
then run `transcribe` to see what the ASR heard. If the wording is badly
mangled, `LOCATOR_WHISPER_MODEL=small.en` usually fixes it.

**Precision says `±300ms (whisper timestamps)`** — forced alignment was rejected
by one of its trust checks and the reason is printed. The answer is still usable,
just less precise. Common on short phrases that repeat.

</details>

---

Built with AI assistance; see [PROMPTS.md](PROMPTS.md) for the prompts used.
