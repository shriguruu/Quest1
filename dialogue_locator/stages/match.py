"""Matching stage: find where a target phrase occurs in a transcript.

Pure functions.  No I/O, no models, no video.  Everything here operates on
a :class:`Transcript` that some earlier stage already produced, which is
what makes this the one stage that is cheap to test exhaustively.

ASR output is never a character-perfect copy of the line you are looking
for.  It mishears words, drops filler, invents filler, splits contractions
its own way, and punctuates to taste.  So matching is a **tiered cascade**,
and it stops at the first tier that produces a hit above threshold:

``EXACT``
    The normalised query appears verbatim in the normalised transcript.
    Score 100.  Cheap, and when it fires there is nothing to argue about.

``FUZZY``
    Character-level alignment via rapidfuzz.  Catches ordinary ASR slips --
    a dropped article, an inserted "well", a mangled inflection.

``PHONETIC``
    Both sides reduced to Double Metaphone phoneme streams before
    comparison.  Catches homophone errors that are spelled nothing alike
    but sound identical: "rebels at" heard as "rebel that".  Capped at 85
    because sounding alike is weaker evidence than being alike.

The cascade order is also a cost order, but the real reason for stopping
early is precision: a phonetic tier let loose on a transcript that already
contains a literal match will happily surface worse answers that merely
rhyme.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import structlog
from rapidfuzz import fuzz

from dialogue_locator.config import Settings
from dialogue_locator.models import (
    Candidate,
    MatchTier,
    ResultStatus,
    Word,
)

logger = structlog.get_logger()

# Tier acceptance thresholds.
_FUZZY_MIN = 65.0
_PHONETIC_CAP = 85.0
# Two candidates this close together mean we cannot honestly pick one.
_AMBIGUITY_MARGIN = 5.0
# Spans overlapping by more than this fraction are the same hit.
_OVERLAP_LIMIT = 0.5


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

# Expanded on *both* sides so that "don't" and "do not" meet in the middle
# rather than costing an edit-distance penalty.
_CONTRACTIONS = {
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "you're": "you are", "you've": "you have", "you'll": "you will",
    "you'd": "you would", "he's": "he is", "he'll": "he will",
    "he'd": "he would", "she's": "she is", "she'll": "she will",
    "she'd": "she would", "it's": "it is", "it'll": "it will",
    "that's": "that is", "that'll": "that will", "there's": "there is",
    "there'll": "there will", "here's": "here is", "what's": "what is",
    "who's": "who is", "where's": "where is", "when's": "when is",
    "how's": "how is", "let's": "let us", "we're": "we are",
    "we've": "we have", "we'll": "we will", "we'd": "we would",
    "they're": "they are", "they've": "they have", "they'll": "they will",
    "they'd": "they would", "can't": "cannot", "won't": "will not",
    "shan't": "shall not", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "weren't": "were not", "haven't": "have not",
    "hasn't": "has not", "hadn't": "had not", "wouldn't": "would not",
    "shouldn't": "should not", "couldn't": "could not",
    "mustn't": "must not", "needn't": "need not", "ain't": "is not",
}

_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_CONTRACTIONS, key=len, reverse=True)) + r")\b"
)

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
]
_SCALES = [(1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand")]

_DIGITS_RE = re.compile(r"\d[\d,]*")
_APOSTROPHE_RE = re.compile(r"['’ʼ]")
_NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _int_to_words(n: int) -> str:
    """Spell out a non-negative integer in English."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        rest = n % 10
        return _TENS[n // 10] + (f" {_ONES[rest]}" if rest else "")
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + (f" {_int_to_words(rest)}" if rest else "")
    for scale, name in _SCALES:
        if n >= scale:
            rest = n % scale
            return (
                f"{_int_to_words(n // scale)} {name}"
                + (f" {_int_to_words(rest)}" if rest else "")
            )
    return str(n)  # pragma: no cover - unreachable for int inputs


def _spell_digits(match: re.Match[str]) -> str:
    raw = match.group(0).replace(",", "")
    try:
        return _int_to_words(int(raw))
    except (ValueError, OverflowError):
        return raw


def normalize(text: str) -> str:
    """Reduce *text* to a canonical comparable form.

    NFKC first, so that typographic variants (ligatures, full-width forms,
    curly quotes) collapse onto their plain equivalents before anything
    else looks at the string.  Digits become words because Whisper is
    genuinely inconsistent about which it emits -- the same spoken "three"
    can come back as ``3`` in one segment and ``three`` in the next, and a
    query should match either.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _APOSTROPHE_RE.sub("'", text)
    text = _CONTRACTION_RE.sub(lambda m: _CONTRACTIONS[m.group(0)], text)
    text = _DIGITS_RE.sub(_spell_digits, text)
    # Apostrophes are deleted rather than spaced, so any contraction that
    # survived expansion ("o'clock") stays one token instead of splitting.
    text = _APOSTROPHE_RE.sub("", text)
    text = _NON_WORD_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def normalize_tokens(text: str) -> list[str]:
    """Convenience wrapper: :func:`normalize` split into tokens."""
    normalized = normalize(text)
    return normalized.split() if normalized else []


# ---------------------------------------------------------------------------
# Search index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchIndex:
    """A transcript flattened into one searchable string.

    The point of this structure is ``char_to_word``: a match found in
    ``text`` at character offset *i* belongs to word ``char_to_word[i]``,
    with no reconstruction needed.  The mapping is built during
    concatenation, when the correspondence is known exactly -- trying to
    recover it afterwards by re-tokenising is where off-by-one errors
    breed, because normalisation is not length-preserving.
    """

    text: str
    char_to_word: list[int]
    words: list[Word] = field(repr=False)
    word_spans: list[tuple[int, int] | None] = field(repr=False)

    def word_at(self, offset: int) -> int:
        """Return the word index owning character *offset* (clamped)."""
        if not self.char_to_word:
            return 0
        offset = max(0, min(offset, len(self.char_to_word) - 1))
        return self.char_to_word[offset]

    def __len__(self) -> int:
        return len(self.text)


def build_search_index(words: list[Word]) -> SearchIndex:
    """Concatenate *words* into one normalised string with a char->word map.

    A single :class:`Word` may normalise to several tokens (an expanded
    contraction) or to nothing at all (pure punctuation).  Both cases are
    handled here rather than being pushed onto callers.
    """
    parts: list[str] = []
    char_to_word: list[int] = []
    word_spans: list[tuple[int, int] | None] = []
    cursor = 0

    for word_index, word in enumerate(words):
        normalized = normalize(word.text)
        if not normalized:
            # Punctuation-only token: it occupies no characters, so it can
            # never be the start of a match.
            word_spans.append(None)
            continue

        if parts:
            # The separating space is attributed to the *upcoming* word, so
            # an alignment that begins on the space still resolves to the
            # right first word.
            parts.append(" ")
            char_to_word.append(word_index)
            cursor += 1

        start = cursor
        parts.append(normalized)
        char_to_word.extend([word_index] * len(normalized))
        cursor += len(normalized)
        word_spans.append((start, cursor))

    return SearchIndex(
        text="".join(parts),
        char_to_word=char_to_word,
        words=words,
        word_spans=word_spans,
    )


# ---------------------------------------------------------------------------
# Phonetics
# ---------------------------------------------------------------------------


def _phoneme_stream(tokens: list[str]) -> str:
    """Encode tokens as a Double Metaphone stream.

    The primary code is used, falling back to the alternate (and then to
    the token itself) when a word has no primary encoding.
    """
    try:
        from metaphone import doublemetaphone
    except ImportError:  # pragma: no cover - dependency is declared
        try:
            import jellyfish

            return " ".join(jellyfish.metaphone(t) or t for t in tokens)
        except ImportError:
            return " ".join(tokens)

    codes: list[str] = []
    for token in tokens:
        primary, alternate = doublemetaphone(token)
        codes.append(primary or alternate or token)
    return " ".join(codes)


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------


def _make_candidate(
    index: SearchIndex,
    char_start: int,
    char_end: int,
    score: float,
    tier: MatchTier,
) -> Candidate | None:
    """Build a Candidate from a character span in the index."""
    if not index.words or char_start >= char_end:
        return None

    word_start = index.word_at(char_start)
    word_end = index.word_at(char_end - 1)
    if word_end < word_start:
        word_start, word_end = word_end, word_start

    matched = index.words[word_start : word_end + 1]
    if not matched:
        return None

    return Candidate(
        word_index_start=word_start,
        word_index_end=word_end,
        matched_text=" ".join(w.text for w in matched),
        score=round(float(score), 2),
        tier=tier,
        coarse_start_s=matched[0].start_s,
    )


def _word_windows(
    index: SearchIndex, query_word_count: int, slack: int = 3
) -> list[tuple[int, int]]:
    """Character spans for sliding word-windows of plausible width."""
    spans = [s for s in index.word_spans if s is not None]
    if not spans:
        return []

    widths = sorted(
        {
            w
            for w in range(query_word_count - slack, query_word_count + slack + 1)
            if 1 <= w <= len(spans)
        }
    )

    out: list[tuple[int, int]] = []
    for width in widths:
        for start in range(0, len(spans) - width + 1):
            out.append((spans[start][0], spans[start + width - 1][1]))
    return out


def _local_maxima(
    scored: list[tuple[float, int, int]], min_score: float
) -> list[tuple[float, int, int]]:
    """Keep windows that are at least as good as their neighbours.

    A single true occurrence produces a run of overlapping windows with
    similar scores.  Taking only the global best would report one hit for
    a phrase that occurs three times, so we keep every local peak and let
    deduplication collapse each run to its best representative.
    """
    peaks = [item for item in scored if item[0] >= min_score]
    peaks.sort(key=lambda item: (-item[0], item[1]))
    return peaks


# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------


def _tier_exact(index: SearchIndex, query_norm: str) -> list[Candidate]:
    """Every verbatim occurrence of the normalised query."""
    out: list[Candidate] = []
    if not query_norm:
        return out

    start = index.text.find(query_norm)
    while start != -1:
        candidate = _make_candidate(
            index, start, start + len(query_norm), 100.0, MatchTier.EXACT
        )
        if candidate is not None:
            out.append(candidate)
        start = index.text.find(query_norm, start + 1)
    return out


def _tier_fuzzy(
    index: SearchIndex, query_norm: str, query_words: int, floor: float = _FUZZY_MIN
) -> list[Candidate]:
    """Global alignment plus sliding windows, so repeats are all found."""
    candidates: list[Candidate] = []

    # Global best alignment. Called as (transcript, query) so that
    # src_start/src_end are offsets into the transcript.
    alignment = fuzz.partial_ratio_alignment(index.text, query_norm)
    if alignment is not None and alignment.score >= floor:
        candidate = _make_candidate(
            index,
            alignment.src_start,
            alignment.src_end,
            alignment.score,
            MatchTier.FUZZY,
        )
        if candidate is not None:
            candidates.append(candidate)

    # Sliding windows recover the occurrences the global best hides.
    scored = [
        (fuzz.ratio(query_norm, index.text[start:end]), start, end)
        for start, end in _word_windows(index, query_words)
    ]
    for score, start, end in _local_maxima(scored, floor):
        candidate = _make_candidate(index, start, end, score, MatchTier.FUZZY)
        if candidate is not None:
            candidates.append(candidate)

    return candidates


def _tier_phonetic(
    index: SearchIndex,
    query_tokens: list[str],
    query_words: int,
    floor: float = _FUZZY_MIN,
) -> list[Candidate]:
    """Compare Double Metaphone streams instead of letters."""
    query_phon = _phoneme_stream(query_tokens)
    if not query_phon:
        return []

    scored: list[tuple[float, int, int]] = []
    for start, end in _word_windows(index, query_words):
        window_tokens = index.text[start:end].split()
        if not window_tokens:
            continue
        window_phon = _phoneme_stream(window_tokens)
        # Phonetic agreement is weaker evidence than character agreement,
        # so the tier is capped rather than allowed to reach 100.
        score = min(fuzz.ratio(query_phon, window_phon), _PHONETIC_CAP)
        scored.append((score, start, end))

    out: list[Candidate] = []
    for score, start, end in _local_maxima(scored, floor):
        candidate = _make_candidate(index, start, end, score, MatchTier.PHONETIC)
        if candidate is not None:
            out.append(candidate)
    return out


# ---------------------------------------------------------------------------
# Deduplication and ranking
# ---------------------------------------------------------------------------


def _overlap_fraction(a: Candidate, b: Candidate) -> float:
    """Fraction of the shorter word-span that the two candidates share."""
    lo = max(a.word_index_start, b.word_index_start)
    hi = min(a.word_index_end, b.word_index_end)
    if hi < lo:
        return 0.0
    shared = hi - lo + 1
    shortest = min(
        a.word_index_end - a.word_index_start + 1,
        b.word_index_end - b.word_index_start + 1,
    )
    return shared / shortest if shortest else 0.0


def _dedupe(candidates: list[Candidate]) -> list[Candidate]:
    """Collapse candidates that overlap by more than half, keeping the best."""
    ordered = sorted(candidates, key=lambda c: (-c.score, c.coarse_start_s))
    kept: list[Candidate] = []
    for candidate in ordered:
        if any(_overlap_fraction(candidate, k) > _OVERLAP_LIMIT for k in kept):
            continue
        kept.append(candidate)
    return kept


def find_candidates(
    index: SearchIndex,
    query: str,
    top_k: int = 5,
    min_score: float | None = None,
) -> list[Candidate]:
    """Return the best matches for *query*, best score first.

    Ties in score are broken by time ascending, so that a phrase repeated
    verbatim comes back in the order it occurs in the media.

    *min_score* overrides the tier acceptance floor.  Its purpose is to let
    a caller ask "what was the closest thing you saw?" after a normal
    search found nothing, so a NOT_FOUND result can show near-misses rather
    than an unhelpful empty list.  Leave it as ``None`` for real searches.
    """
    query_norm = normalize(query)
    if not query_norm:
        raise ValueError("Query is empty after normalisation")
    if not index.words or not index.text:
        return []

    query_tokens = query_norm.split()
    query_words = len(query_tokens)
    floor = _FUZZY_MIN if min_score is None else min_score

    for tier_name, produce in (
        ("exact", lambda: _tier_exact(index, query_norm)),
        ("fuzzy", lambda: _tier_fuzzy(index, query_norm, query_words, floor)),
        (
            "phonetic",
            lambda: _tier_phonetic(index, query_tokens, query_words, floor),
        ),
    ):
        found = produce()
        if not found:
            continue

        results = _dedupe(found)
        results.sort(key=lambda c: (-c.score, c.coarse_start_s))
        results = results[:top_k]

        logger.info(
            "Match complete",
            tier=tier_name,
            candidates=len(results),
            best_score=results[0].score if results else None,
        )
        return results

    logger.info("No candidates found in any tier", query=query)
    return []


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify(
    candidates: list[Candidate], settings: Settings
) -> tuple[ResultStatus, list[str]]:
    """Turn a ranked candidate list into a status plus human explanation."""
    if not candidates:
        return ResultStatus.NOT_FOUND, [
            "No part of the transcript resembled the query."
        ]

    best = candidates[0]
    warnings: list[str] = []

    if best.score < settings.uncertain_threshold:
        return ResultStatus.NOT_FOUND, [
            f"Best match scored {best.score:.1f}, below the "
            f"{settings.uncertain_threshold:.0f} floor for a usable result."
        ]

    status = (
        ResultStatus.CONFIDENT
        if best.score >= settings.confident_threshold
        else ResultStatus.UNCERTAIN
    )

    if status is ResultStatus.UNCERTAIN:
        warnings.append(
            f"Match scored {best.score:.1f}, between the "
            f"{settings.uncertain_threshold:.0f} and "
            f"{settings.confident_threshold:.0f} thresholds: the located text "
            f"resembles the query but is not a literal match."
        )

    # Two contenders too close to separate is a different failure from a
    # weak single match, and it is worth naming: the phrase may genuinely
    # occur more than once.
    if len(candidates) > 1:
        runner_up = candidates[1]
        gap = best.score - runner_up.score
        if gap <= _AMBIGUITY_MARGIN:
            status = ResultStatus.UNCERTAIN
            warnings.append(
                f"Ambiguous: the top two matches score {best.score:.1f} and "
                f"{runner_up.score:.1f} ({gap:.1f} apart) at "
                f"{best.coarse_start_s:.2f}s and {runner_up.coarse_start_s:.2f}s. "
                f"The phrase may occur more than once."
            )

    if best.tier is MatchTier.PHONETIC:
        warnings.append(
            "Matched on pronunciation rather than spelling; the transcript "
            "wording differs from the query."
        )

    return status, warnings
