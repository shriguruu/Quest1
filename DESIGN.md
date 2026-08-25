# Design & Approach

**dialogue-locator** takes a video URL and a line of dialogue, and returns the
exact frame where that line begins — timestamp, frame number, the text as
actually spoken, and the frame as a PNG.

---

## The shape of the problem

Three things have to be true for an answer to be correct, and they fail
independently:

1. **We must find the right moment** in a video that may be an hour long,
   without anyone watching it first.
2. **We must know precisely when the line starts** — precisely enough that
   rounding to a frame is meaningful.
3. **We must convert that instant into the right frame number** for this
   specific file.

Most of the engineering below exists because the naive answer to (2) is off
by 3–8 frames, and the naive answer to (3) is off by up to 221 frames.

---

## Pipeline

```
URL ──► acquire ──► probe ──► PTS index
                                 │
                                 ├──► extract_audio ──► transcribe (Whisper)
                                 │                            │
                                 │                     build_search_index
                                 │                            │
                                 │                     find_candidates  ──► classify
                                 │                            │
                                 │                     align (wav2vec2 CTC)
                                 │                            │
                                 └──────────────► audio_time_to_frame ──► PNG + JSON
```

Each stage is a module under `dialogue_locator/stages/`, communicating only
through frozen Pydantic models in `models.py`. Nothing shares mutable state,
so any stage can be tested, replaced, or reasoned about alone.

---

## How we decide where to look

We do not guess. Whisper transcribes the **whole** audio track once, producing
a word-level transcript with a timestamp on every word. The search then happens
over text, which is cheap, rather than over video, which is not.

This is the step that makes the tool usable without a human skimming the video
first. The transcript is cached against a hash of the audio content plus every
parameter that affects the output, so the expensive part is paid once.

**Whisper's segments are deliberately discarded.** Whisper emits "segments"
whose boundaries are an artefact of its decoding window, not of the speech, and
a target phrase straddles one often enough to matter — in this very video, the
target sits across a boundary:

```
in[s34] time.[s34] My[s35] mind[s35] rebels[s35] at[s35] stagnation.[s35] Give[s36]
```

So segments are flattened immediately into one continuous `list[Word]`, with
`segment_id` kept only for provenance. All matching runs on the flat stream.

---

## How we find the line in the transcript

ASR output is never a character-perfect copy of what you typed. It mishears
words, drops articles, invents filler, and punctuates to taste. A literal
string search fails on most real queries.

Matching is a **tiered cascade** that stops at the first tier producing a hit
above threshold:

| Tier | Method | Catches |
|---|---|---|
| `EXACT` | Normalised substring | Literal matches. Score 100 |
| `FUZZY` | rapidfuzz character alignment | Dropped/added words, mangled inflections |
| `PHONETIC` | Double Metaphone streams | Homophones spelled differently |

Both sides are normalised first: NFKC → casefold → expand contractions →
**spell out digits** (Whisper is inconsistent about `3` vs `three`) → strip
punctuation.

**Why stop early rather than run all three?** The cascade is also a precision
order. Turned loose on a transcript that already contains a literal match, a
phonetic tier will happily surface worse answers that merely rhyme.

**Where the tiers actually divide.** Measured on real transcripts, the fuzzy
tier absorbs nearly everything, including most homophones — English spelling is
regular enough that a mishearing usually stays spelled similarly. `rebels at`
heard as `rebel that` scores 94.7 on characters alone. The phonetic tier only
earns its place when spelling diverges hard while pronunciation does not:
`eight pheasants` → `ate fezzants` scores 51.9 on characters but both encode to
`AT FSNTS`. This is documented in `tests/test_match.py::TestTierBoundaries`
rather than assumed.

### Recovering the exact word

`build_search_index()` concatenates the transcript into one normalised string
**and builds a character→word map as it goes**. A match at character offset *i*
belongs to word `char_to_word[i]`, with no reconstruction needed.

Building the map during concatenation rather than re-deriving it afterwards is
deliberate: normalisation is not length-preserving. One `Word` can become
several tokens (`don't` → `do not`) or none at all (pure punctuation), and
re-tokenising to recover the correspondence is where off-by-one errors breed.

---

## How we pin down the exact instant

Whisper does **not** measure when a word begins. Its word timestamps come from
cross-attention DTW — a byproduct of where the decoder was attending — and land
within roughly 100–300ms of the truth. At 24fps that is 3–8 frames of slop,
which is fatal for a tool whose entire output is one frame number.

So a second model re-does only this one job. **Forced alignment** solves a
strictly easier problem: recognition asks *"what words are in this audio?"*,
alignment already knows the words and asks only *"where does each one sit?"*
Constraining the search to a known token sequence collapses the ambiguity, and
a CTC model resolves to one emission frame — measured at **20.03ms** here,
comfortably under a single video frame.

On the target line, this moved the answer **228ms later — about 5 frames**.

**Two decisions worth defending:**

*torchaudio, not whisperx.* `whisperx.align` wraps roughly the same wav2vec2 CTC
procedure but arrives with a large transitive dependency tree and pins
`faster-whisper` versions, risking a transcription stage that already works.
`torchaudio.functional.forced_align` is the bare Viterbi-over-CTC primitive, so
the stage is ~30 lines of real logic with no version coupling. When a
dependency's only advantage is convenience and the thing it wraps is this
small, wrapping it yourself is cheaper than owning the dependency.

*Align the transcript, not the query.* CTC forced alignment **cannot refuse**.
Hand it a token sequence that is not in the audio and it will still produce a
path, smearing those tokens across whatever sound is present and reporting
confident nonsense. The query is only *approximately* what was said — that is
the whole premise of the fuzzy matcher. So the query decides *which span*, and
the transcript decides *what text* gets aligned inside it.

---

## How we convert an instant into a frame

This is where the largest error was hiding.

The obvious formula is `frame = floor(t × fps)`. Applied to the target video it
is **wrong by 221 frames — about 9 seconds — near the end of the file.**

Two separate causes:

**1. `fps` is nominal, not real.** The container advertises `r_frame_rate =
24000/1001`. But this is an HLS stream reassembled from segments, and its actual
frames are not on that grid — 77984 frames span 3261.74s, an average of 23.909
fps, with irregular gaps at segment boundaries.

**2. ffprobe's own VFR signal is misleading.** `avg_frame_rate` is derived from
the *container* duration. Here the audio track runs ~9s longer than the video
track, so `avg_frame_rate` reads low and a naive comparison flags VFR for
entirely the wrong reason. We compare against the video stream's own
`nb_frames / duration` instead.

**The fix: measure, don't compute.** `build_pts_index()` reads every video
packet's presentation timestamp via ffprobe — packet-level, so no decoding, a
few seconds for 78k frames — sorts it into display order, and caches it by file
hash. Frame lookup is then a binary search for the last PTS ≤ *t*.

Two properties follow. `floor`, never `round`: a viewer who pauses at *t* sees
the frame most recently presented, not the one about to be. And the A/V offset
(video and audio streams often have different `start_time`) is applied in
exactly one place, `audio_time_to_frame()`, so it can never be double-applied
or forgotten.

Extraction uses **PyAV seek-then-decode**, not `ffmpeg -ss`, which lands on the
nearest keyframe. We seek to the keyframe before the target, decode forward, and
**assert the decoded PTS matches the request** within half a frame.

That assertion is what makes the output trustworthy: verified against ffmpeg
independently, our PNG is **pixel-identical** to `select=lt(abs(t-25.067),0.001)`.

> **Known limit.** One packet yields one frame for well-formed files, but a
> fragment cut mid-GOP with `ffmpeg -c copy` carries leading packets a decoder
> discards — we measured 1510 packets vs 1438 decoded frames on such a clip.
> The full video shows a smaller version of this at its tail (77984 packets,
> 77980 decoded). Timestamps and images stay correct regardless, because
> extraction validates PTS; only the integer index can shift. Re-encode rather
> than stream-copy if frame numbers from a clip must line up with the source.

---

## How uncertainty is handled

The governing rule: **a known-imprecise answer is more useful than a
precise-looking wrong one.** Nothing degrades silently.

**Match confidence** → `CONFIDENT` (≥85) / `UNCERTAIN` (65–85) / `NOT_FOUND`
(<65). Additionally, if the top two candidates land within 5 points of each
other, the result is forced to `UNCERTAIN` — genuine ambiguity is a different
failure from a weak match, and usually means the phrase occurs more than once.
The CLI then prints the top three alternates with timestamps and never silently
picks one.

**Alignment confidence** → three independent guards, any of which demotes the
answer back to Whisper's timestamp *with a stated reason*:

- **Silero VAD** — if the onset lands inside detected silence, alignment slid
  off the utterance. Retry once with a 2× window, then fall back.
- **Drift** — an onset more than 2.0s from Whisper's estimate means the two
  models disagree about *which utterance* this is, not merely about its edge.
- **CTC likelihood** — a low-probability path means the text does not fit the
  audio.

The method actually used is surfaced in both outputs, so precision is never
overstated:

```
Precision : +/-20ms (forced alignment)      ← succeeded
Precision : +/-300ms (whisper timestamps)   ← fell back, with the reason printed
```

**`NOT_FOUND` still shows its work.** An empty answer is hard to act on, so the
search re-runs at a deliberately low floor and reports the near-misses — usually
the difference between "not in this video" and "your query is worded differently".

**Exit codes** make this scriptable: `0` CONFIDENT · `2` UNCERTAIN · `3`
NOT_FOUND · `1` ERROR. Failures return a structured `ERROR` result, never a
traceback.

---

## Other decisions

**`Fraction` for frame rates, never `float`.** `29.97` is a lie; the value is
`30000/1001`. Arithmetic stays rational until a single `int()` at the end.

**Cache everything expensive, keyed by content.** Downloads by `sha256(url +
format_id)` — the frame number is a property of a specific rendition, so the
format is part of the identity. Transcripts by `sha256(audio) + model +
parameters`, so a changed parameter can never silently reuse a stale result.
PTS indexes by file hash.

**CPU-only by default, and pinned that way.** `pyproject.toml` scopes torch to
PyTorch's CPU index, so a future `uv sync` cannot silently pull the 2–3GB CUDA
build. `device` defaults to `cuda` with an automatic CPU fallback — and that
fallback wraps the *whole transcription*, not the model load, because
CTranslate2 constructs a CUDA model happily with no cuBLAS present and only
fails on the first `encode()` call, lazily, mid-generator.

**Models are freed explicitly.** Whisper is released before alignment loads
wav2vec2, so the two never coexist.

---

## Testing

99 tests, no network and no media fixtures required.

| Suite | Tests | Covers |
|---|---|---|
| `test_framemap.py` | 34 | CFR/VFR mapping at 25 / 30 / 30000-1001 fps, NTSC drift, A/V offset applied exactly once, boundary conditions at ±1µs, extraction verified against ffmpeg |
| `test_match.py` | 47 | Normalisation, char→word mapping, all three tiers, **phrase straddling segments**, repeated occurrences returned in time order, classification |
| `test_align.py` | 18 | Onset accuracy against ground truth, all three fallback guards, token preparation |

`test_align.py` builds its own ground truth: a TTS clip is trimmed to its first
audible sample and spliced into digital silence at exactly 2.500s, making the
true onset exact by construction rather than by annotation.

That fixture also exposed something worth recording. The aligner's error is
always *positive* and tracks how abruptly the sound starts:

```
plosive /k/  +7ms      fricative /s/  +66ms
plosive /t/ +23ms      nasal /m/      +68ms
plosive /p/ +24ms
```

A plosive opens with a burst, so an energy threshold and a CTC model agree on
where it begins. A nasal or fricative ramps up, so a 2%-of-peak threshold fires
early on the ramp while the model waits for evidence. That gap is a
disagreement about what "onset" *means* for a gradual sound — not aligner error.
The strict ±30ms assertion therefore uses a plosive, and the continuant
behaviour gets its own test documenting it rather than a loosened tolerance
hiding it.
