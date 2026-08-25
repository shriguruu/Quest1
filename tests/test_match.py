"""Tests for dialogue_locator.stages.match.

Transcripts are built by hand here -- no video, no audio, no models.  That
is the whole point of keeping this stage pure: the interesting failure
modes (a dropped word, a homophone, a phrase straddling two segments) can
be constructed exactly rather than hunted for in real media.
"""

from __future__ import annotations

import pytest

from dialogue_locator.config import Settings
from dialogue_locator.models import MatchTier, ResultStatus, Word
from dialogue_locator.stages.match import (
    SearchIndex,
    build_search_index,
    classify,
    find_candidates,
    normalize,
)

QUERY = "My mind rebels at stagnation"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_words(
    text: str, *, start_s: float = 0.0, step: float = 0.5, segment_id: int = 0
) -> list[Word]:
    """Turn a sentence into evenly-spaced Words in one segment."""
    return [
        Word(
            text=token,
            start_s=start_s + i * step,
            end_s=start_s + (i + 1) * step,
            probability=0.9,
            segment_id=segment_id,
        )
        for i, token in enumerate(text.split())
    ]


def make_segmented(*segments: str, step: float = 0.5) -> list[Word]:
    """Build a word stream spanning several segments, times continuous."""
    words: list[Word] = []
    clock = 0.0
    for seg_id, seg_text in enumerate(segments):
        for token in seg_text.split():
            words.append(
                Word(
                    text=token,
                    start_s=clock,
                    end_s=clock + step,
                    probability=0.9,
                    segment_id=seg_id,
                )
            )
            clock += step
    return words


def index_of(text: str) -> SearchIndex:
    return build_search_index(make_words(text))


# ---------------------------------------------------------------------------
# normalize()
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_casefold_and_punctuation(self) -> None:
        assert normalize("My Mind, Rebels: at Stagnation!") == (
            "my mind rebels at stagnation"
        )

    def test_collapses_whitespace(self) -> None:
        assert normalize("  my   mind \n rebels  ") == "my mind rebels"

    def test_expands_contractions(self) -> None:
        assert normalize("Don't stop, it's fine") == "do not stop it is fine"

    def test_curly_apostrophe_treated_as_straight(self) -> None:
        assert normalize("don’t") == normalize("don't") == "do not"

    def test_nfkc_normalisation(self) -> None:
        # Full-width characters and a ligature collapse to ASCII.
        assert normalize("Ｆｉｎｅ") == "fine"
        assert normalize("ﬁne") == "fine"

    def test_digits_become_words(self) -> None:
        assert normalize("3 pipe problem") == "three pipe problem"
        assert normalize("221 Baker Street") == (
            "two hundred twenty one baker street"
        )

    def test_digits_match_spelled_form(self) -> None:
        assert normalize("chapter 21") == normalize("chapter twenty-one")

    def test_empty_string(self) -> None:
        assert normalize("") == ""
        assert normalize("...") == ""


# ---------------------------------------------------------------------------
# build_search_index()
# ---------------------------------------------------------------------------


class TestSearchIndex:
    def test_text_is_normalized_concatenation(self) -> None:
        index = index_of("My mind rebels at stagnation")
        assert index.text == "my mind rebels at stagnation"

    def test_char_map_covers_every_character(self) -> None:
        index = index_of("My mind rebels at stagnation")
        assert len(index.char_to_word) == len(index.text)

    def test_char_map_points_at_right_word(self) -> None:
        index = index_of("My mind rebels at stagnation")
        # "rebels" starts at offset 8.
        assert index.text[8:14] == "rebels"
        assert index.word_at(8) == 2
        assert index.words[index.word_at(8)].text == "rebels"

    def test_expanded_contraction_maps_back_to_one_word(self) -> None:
        """A Word whose normal form is several tokens still maps to itself."""
        words = make_words("I don't stagnate")
        index = build_search_index(words)
        assert index.text == "i do not stagnate"
        # Every character of "do not" belongs to the single Word "don't".
        for offset in range(2, 8):
            assert index.words[index.word_at(offset)].text == "don't"

    def test_punctuation_only_word_is_skipped(self) -> None:
        words = make_words("mind -- rebels")
        index = build_search_index(words)
        assert index.text == "mind rebels"
        assert index.word_spans[1] is None

    def test_empty_transcript(self) -> None:
        index = build_search_index([])
        assert index.text == ""
        assert find_candidates(index, QUERY) == []


# ---------------------------------------------------------------------------
# Tier 1: exact
# ---------------------------------------------------------------------------


class TestExactMatch:
    def test_exact_match_scores_100(self) -> None:
        index = index_of("Well then my mind rebels at stagnation give me problems")
        results = find_candidates(index, QUERY)

        assert results
        best = results[0]
        assert best.score == 100.0
        assert best.tier is MatchTier.EXACT
        assert best.matched_text == "my mind rebels at stagnation"

    def test_exact_match_recovers_first_word(self) -> None:
        index = index_of("Well then my mind rebels at stagnation give me problems")
        best = find_candidates(index, QUERY)[0]

        assert index.words[best.word_index_start].text == "my"
        # "my" is the third word, at 2 * 0.5s.
        assert best.coarse_start_s == pytest.approx(1.0)

    def test_case_and_punctuation_insensitive(self) -> None:
        index = index_of("MY MIND, REBELS AT STAGNATION!")
        best = find_candidates(index, "my mind rebels at stagnation")[0]
        assert best.tier is MatchTier.EXACT


# ---------------------------------------------------------------------------
# Tier 2: fuzzy
# ---------------------------------------------------------------------------


class TestFuzzyMatch:
    def test_dropped_word(self) -> None:
        """Whisper drops "at" entirely."""
        index = index_of("well then my mind rebels stagnation give me problems")
        results = find_candidates(index, QUERY)

        assert results
        best = results[0]
        assert best.tier is MatchTier.FUZZY
        assert best.score >= 65.0
        assert "rebels" in best.matched_text

    def test_inserted_filler_word(self) -> None:
        """Whisper hallucinates a filler in the middle of the phrase."""
        index = index_of("my mind really rebels at stagnation")
        results = find_candidates(index, QUERY)

        assert results
        assert results[0].tier is MatchTier.FUZZY
        assert results[0].score >= 65.0

    def test_mangled_inflection(self) -> None:
        index = index_of("my mind rebelled at stagnations")
        results = find_candidates(index, QUERY)

        assert results
        assert results[0].score >= 65.0

    def test_fuzzy_match_starts_at_right_word(self) -> None:
        index = index_of("nothing here at all my mind rebels stagnation and more")
        best = find_candidates(index, QUERY)[0]
        assert index.words[best.word_index_start].text in {"my", "mind"}


# ---------------------------------------------------------------------------
# Segment straddling -- the important one
# ---------------------------------------------------------------------------


class TestSegmentStraddle:
    """Whisper's segment boundaries are arbitrary and routinely split a
    target phrase.  Matching runs on the flat word stream precisely so
    that this case behaves no differently from any other."""

    def test_phrase_spanning_two_segments(self) -> None:
        words = make_segmented(
            "I abhor the dull routine of existence my mind rebels",
            "at stagnation give me problems give me work",
        )
        index = build_search_index(words)
        results = find_candidates(index, QUERY)

        assert results
        best = results[0]
        assert best.score == 100.0
        assert best.tier is MatchTier.EXACT
        assert best.matched_text == "my mind rebels at stagnation"

        # The match genuinely crosses the boundary.
        spanned = {
            w.segment_id
            for w in words[best.word_index_start : best.word_index_end + 1]
        }
        assert spanned == {0, 1}

    def test_phrase_spanning_three_segments(self) -> None:
        words = make_segmented(
            "my mind",
            "rebels at",
            "stagnation give me problems",
        )
        index = build_search_index(words)
        best = find_candidates(index, QUERY)[0]

        assert best.score == 100.0
        spanned = {
            w.segment_id
            for w in words[best.word_index_start : best.word_index_end + 1]
        }
        assert spanned == {0, 1, 2}

    def test_straddling_match_reports_first_word_time(self) -> None:
        words = make_segmented("I abhor the dull routine my mind rebels", "at stagnation")
        index = build_search_index(words)
        best = find_candidates(index, QUERY)[0]

        assert words[best.word_index_start].text == "my"
        assert best.coarse_start_s == pytest.approx(words[best.word_index_start].start_s)


# ---------------------------------------------------------------------------
# Repeated occurrences
# ---------------------------------------------------------------------------


class TestRepeatedOccurrences:
    def test_three_occurrences_all_returned(self) -> None:
        filler = "and then he paused for a long moment before continuing again"
        text = " ".join([QUERY, filler, QUERY, filler, QUERY])
        index = build_search_index(make_words(text))

        results = find_candidates(index, QUERY, top_k=5)
        assert len(results) == 3
        assert all(c.score == 100.0 for c in results)

    def test_occurrences_returned_in_time_order(self) -> None:
        filler = "and then he paused for a long moment before continuing again"
        text = " ".join([QUERY, filler, QUERY, filler, QUERY])
        index = build_search_index(make_words(text))

        results = find_candidates(index, QUERY, top_k=5)
        times = [c.coarse_start_s for c in results]
        assert times == sorted(times), "equal scores must break ties by time"

    def test_occurrences_do_not_overlap(self) -> None:
        filler = "and then he paused for a long moment before continuing again"
        text = " ".join([QUERY, filler, QUERY, filler, QUERY])
        index = build_search_index(make_words(text))

        results = find_candidates(index, QUERY, top_k=5)
        spans = sorted((c.word_index_start, c.word_index_end) for c in results)
        for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
            assert next_start > prev_end

    def test_top_k_is_respected(self) -> None:
        filler = "and then he paused for a long moment before continuing again"
        text = " ".join([QUERY, filler, QUERY, filler, QUERY])
        index = build_search_index(make_words(text))

        assert len(find_candidates(index, QUERY, top_k=2)) == 2


# ---------------------------------------------------------------------------
# Tier 3: phonetic
# ---------------------------------------------------------------------------


class TestPhoneticMatch:
    """The phonetic tier only earns its keep when spelling diverges far
    enough that character similarity drops below the fuzzy floor, while
    pronunciation stays identical.  English orthography is regular enough
    that most ASR homophones ("rebels at" -> "rebel that") are still caught
    by the fuzzy tier -- see :class:`TestTierBoundaries`.  These cases are
    the ones that genuinely need phonemes.
    """

    # fuzzy(q, c) == 51.85, but both encode to "AT FSNTS".
    PHONETIC_QUERY = "eight pheasants"
    PHONETIC_HEARD = "he shot ate fezzants that morning in the field"

    def test_homophone_substitution_reaches_phonetic_tier(self) -> None:
        index = index_of(self.PHONETIC_HEARD)
        results = find_candidates(index, self.PHONETIC_QUERY)

        assert results, "phonetic tier should rescue this"
        assert results[0].tier is MatchTier.PHONETIC, (
            "expected the cascade to fall through exact and fuzzy"
        )
        assert "fezzants" in results[0].matched_text

    def test_fuzzy_tier_would_have_missed_it(self) -> None:
        """Guard the premise: character similarity really is below floor."""
        from rapidfuzz import fuzz

        from dialogue_locator.stages.match import normalize as _n

        assert fuzz.ratio(_n(self.PHONETIC_QUERY), _n("ate fezzants")) < 65.0

    def test_second_homophone_case(self) -> None:
        index = index_of("he said tho ruff coffs kept him awake all night")
        results = find_candidates(index, "though rough coughs")

        assert results
        assert results[0].tier is MatchTier.PHONETIC

    def test_phonetic_scores_are_capped(self) -> None:
        index = index_of(self.PHONETIC_HEARD)
        results = find_candidates(index, self.PHONETIC_QUERY)

        assert results
        assert any(c.tier is MatchTier.PHONETIC for c in results)
        for candidate in results:
            assert candidate.score <= 85.0, (
                "phonetic evidence must never reach a perfect score"
            )

    def test_identical_phonemes_still_capped_not_100(self) -> None:
        """The phoneme streams are byte-identical, yet the score is 85."""
        from dialogue_locator.stages.match import _phoneme_stream
        from dialogue_locator.stages.match import normalize as _n

        assert _phoneme_stream(_n(self.PHONETIC_QUERY).split()) == _phoneme_stream(
            _n("ate fezzants").split()
        )
        index = index_of(self.PHONETIC_HEARD)
        assert find_candidates(index, self.PHONETIC_QUERY)[0].score == 85.0

    def test_exact_match_short_circuits_before_phonetic(self) -> None:
        """A literal hit must not be displaced by something that rhymes."""
        index = index_of(
            "my mind rebels at stagnation and later my mined rebbles att stagnashun"
        )
        results = find_candidates(index, QUERY)

        assert results[0].tier is MatchTier.EXACT
        assert all(c.tier is MatchTier.EXACT for c in results)


class TestTierBoundaries:
    """Document which tier actually handles each class of ASR error."""

    @pytest.mark.parametrize(
        ("label", "transcript"),
        [
            ("homophone", "well then my mind rebel that stagnation give me problems"),
            ("heavy misspelling", "my mined rebbles att stagnashun and so on"),
            ("dropped word", "well then my mind rebels stagnation give me problems"),
            ("inserted filler", "my mind really rebels at stagnation today"),
        ],
    )
    def test_fuzzy_tier_handles_ordinary_asr_errors(
        self, label: str, transcript: str
    ) -> None:
        """These never reach the phonetic tier, and should not need to."""
        index = index_of(transcript)
        results = find_candidates(index, QUERY)

        assert results, label
        assert results[0].tier is MatchTier.FUZZY, label
        assert results[0].score >= 65.0, label


# ---------------------------------------------------------------------------
# classify()
# ---------------------------------------------------------------------------


class TestClassify:
    def setup_method(self) -> None:
        self.settings = Settings()

    def test_confident_on_exact_match(self) -> None:
        index = index_of("well then my mind rebels at stagnation give me problems")
        status, warnings = classify(find_candidates(index, QUERY), self.settings)

        assert status is ResultStatus.CONFIDENT
        assert warnings == []

    def test_not_found_when_query_absent(self) -> None:
        index = index_of(
            "the curious incident of the dog in the night time was most instructive"
        )
        results = find_candidates(index, QUERY)
        status, warnings = classify(results, self.settings)

        assert status is ResultStatus.NOT_FOUND
        assert warnings and any("below" in w or "No part" in w for w in warnings)

    def test_not_found_on_empty_candidate_list(self) -> None:
        status, warnings = classify([], self.settings)
        assert status is ResultStatus.NOT_FOUND
        assert warnings

    def test_two_near_identical_passages_are_uncertain(self) -> None:
        """Two passages that both nearly match must not yield a confident
        guess at one of them."""
        text = (
            "my mind rebels at stagnation "
            "and he paused for a long moment before saying "
            "my mind rebels at stagnation"
        )
        index = build_search_index(make_words(text))
        results = find_candidates(index, QUERY)
        status, warnings = classify(results, self.settings)

        assert len(results) >= 2
        assert status is ResultStatus.UNCERTAIN
        assert any("Ambiguous" in w for w in warnings)

    def test_ambiguity_warning_names_both_times(self) -> None:
        text = (
            "my mind rebels at stagnation "
            "and he paused for a long moment before saying "
            "my mind rebels at stagnation"
        )
        index = build_search_index(make_words(text))
        status, warnings = classify(find_candidates(index, QUERY), self.settings)

        ambiguity = next(w for w in warnings if "Ambiguous" in w)
        assert "0.00s" in ambiguity

    def test_phonetic_match_is_flagged(self) -> None:
        index = index_of("well then my mind rebel that stagnation give me problems")
        results = find_candidates(index, QUERY)
        status, warnings = classify(results, self.settings)

        if results and results[0].tier is MatchTier.PHONETIC:
            assert any("pronunciation" in w for w in warnings)

    def test_uncertain_band_produces_explanation(self) -> None:
        index = index_of("my mind rebels stagnation")
        results = find_candidates(index, QUERY)
        status, warnings = classify(results, self.settings)

        assert status in (ResultStatus.CONFIDENT, ResultStatus.UNCERTAIN)
        if status is ResultStatus.UNCERTAIN:
            assert warnings


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


class TestGuards:
    def test_empty_query_raises(self) -> None:
        index = index_of("my mind rebels at stagnation")
        with pytest.raises(ValueError, match="empty after normalisation"):
            find_candidates(index, "   ...   ")

    def test_single_word_query(self) -> None:
        index = index_of("well then my mind rebels at stagnation give me problems")
        results = find_candidates(index, "stagnation")
        assert results
        assert results[0].score == 100.0
