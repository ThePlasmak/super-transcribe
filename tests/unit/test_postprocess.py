"""Unit tests for lib/postprocess.py."""
from __future__ import annotations

import pytest
from lib.postprocess import (
    detect_chapters,
    detect_paragraphs,
    filter_hallucinations,
    merge_sentences,
    remove_filler_words,
    search_transcript,
)


def seg(start, end, text, **kwargs):
    return {"start": start, "end": end, "text": text, **kwargs}


# AIDEV-NOTE: detect_paragraphs mutates segments in-place; pass fresh copies per test.


# ── filter_hallucinations ─────────────────────────────────────────────────────


class TestFilterHallucinations:
    @pytest.mark.parametrize(
        "text",
        [
            "[music]",
            "[Music]",
            "[MUSIC]",
            "[applause]",
            "[laughter]",
            "[silence]",
            "[inaudible]",
            "[background noise]",
            "(music)",
            "(upbeat music)",
            "(dramatic music)",
            "Thank you for watching",
            "thank you for listening",
            "subtitles by SomeGroup",
            "transcribed by AI",
            "www.example.com",
            "...",
            "",
            "   ",
        ],
    )
    def test_known_pattern_removed(self, text):
        segs = [seg(0, 1, text)]
        assert filter_hallucinations(segs) == []

    def test_consecutive_duplicates_removed(self):
        segs = [seg(0, 1, "Hello world."), seg(1, 2, "Hello world.")]
        result = filter_hallucinations(segs)
        assert len(result) == 1

    def test_non_consecutive_duplicates_kept(self):
        segs = [
            seg(0, 1, "Hello world."),
            seg(1, 2, "Different text."),
            seg(2, 3, "Hello world."),
        ]
        result = filter_hallucinations(segs)
        assert len(result) == 3

    def test_clean_segments_pass_through(self):
        segs = [seg(0, 1, "This is a normal sentence.")]
        assert filter_hallucinations(segs) == segs

    def test_mixed_cleans_and_hallucinations(self):
        segs = [
            seg(0, 1, "Good content."),
            seg(1, 2, "[music]"),
            seg(2, 3, "More content."),
        ]
        result = filter_hallucinations(segs)
        assert len(result) == 2
        assert result[0]["text"] == "Good content."
        assert result[1]["text"] == "More content."


# ── remove_filler_words ───────────────────────────────────────────────────────


class TestRemoveFillerWords:
    @pytest.mark.parametrize("filler", ["um", "uh", "er", "ah", "hmm"])
    def test_single_filler_removed(self, filler):
        segs = [seg(0, 1, f"{filler} hello")]
        result = remove_filler_words(segs)
        assert filler.lower() not in result[0]["text"].lower()

    def test_you_know_removed(self):
        segs = [seg(0, 1, "you know this is good")]
        result = remove_filler_words(segs)
        assert "you know" not in result[0]["text"].lower()

    def test_i_mean_removed(self):
        segs = [seg(0, 1, "I mean it was great")]
        result = remove_filler_words(segs)
        assert "i mean" not in result[0]["text"].lower()

    def test_you_see_removed(self):
        segs = [seg(0, 1, "you see the point")]
        result = remove_filler_words(segs)
        assert "you see" not in result[0]["text"].lower()

    def test_empty_after_removal_drops_segment(self):
        segs = [seg(0, 1, "um uh")]
        result = remove_filler_words(segs)
        assert result == []

    def test_punctuation_cleaned_after_removal(self):
        segs = [seg(0, 1, "um, hello")]
        result = remove_filler_words(segs)
        # Should not start with comma after um is removed
        assert not result[0]["text"].startswith(",")

    def test_word_list_filtered(self):
        words = [
            {"word": " um", "start": 0.0, "end": 0.2},
            {"word": " hello", "start": 0.2, "end": 0.8},
        ]
        segs = [seg(0, 1, " um hello", words=words)]
        result = remove_filler_words(segs)
        word_texts = [w["word"].strip().lower() for w in result[0]["words"]]
        assert "um" not in word_texts
        assert "hello" in word_texts


# ── detect_paragraphs ─────────────────────────────────────────────────────────


class TestDetectParagraphs:
    def test_first_segment_always_paragraph_start(self):
        segs = [seg(0, 1, "Hello."), seg(2, 3, "World.")]
        result = detect_paragraphs(segs)
        assert result[0].get("paragraph_start") is True

    def test_gap_exceeds_min_gap_triggers_paragraph(self):
        segs = [seg(0, 1, "Hello."), seg(10, 11, "World.")]
        result = detect_paragraphs(segs, min_gap=3.0)
        assert result[1].get("paragraph_start") is True

    def test_small_gap_no_paragraph(self):
        segs = [seg(0, 1, "Hello world"), seg(1.5, 2.5, "how are you")]
        result = detect_paragraphs(segs, min_gap=3.0)
        assert not result[1].get("paragraph_start")

    def test_sentence_end_plus_medium_gap_triggers_paragraph(self):
        # gap=2.0 < min_gap=3.0, but sentence ends with '.' and gap >= sentence_gap=1.5
        segs = [seg(0, 1, "Hello."), seg(3, 4, "World.")]
        result = detect_paragraphs(segs, min_gap=3.0, sentence_gap=1.5)
        assert result[1].get("paragraph_start") is True

    def test_empty_list_handled(self):
        assert detect_paragraphs([]) == []


# ── merge_sentences ───────────────────────────────────────────────────────────


class TestMergeSentences:
    def test_short_segments_merged_within_gap(self):
        segs = [seg(0, 0.5, " Hello"), seg(0.6, 1.0, " world.")]
        result = merge_sentences(segs)
        assert len(result) == 1
        assert "Hello" in result[0]["text"]

    def test_terminal_punctuation_flushes(self):
        segs = [
            seg(0, 0.5, " Hello world."),
            seg(0.6, 1.0, " How are you."),
        ]
        result = merge_sentences(segs)
        assert len(result) == 2

    def test_large_gap_flushes(self):
        segs = [seg(0, 0.5, " Hello"), seg(5, 5.5, " world")]
        # gap=4.5 > MAX_GAP=2.0 -> flush
        result = merge_sentences(segs)
        assert len(result) == 2

    def test_words_merged(self):
        segs = [
            seg(0, 0.5, " Hi", words=[{"word": " Hi", "start": 0.0, "end": 0.4}]),
            seg(0.6, 1.1, " there.", words=[{"word": " there.", "start": 0.6, "end": 1.0}]),
        ]
        result = merge_sentences(segs)
        assert len(result[0]["words"]) == 2

    def test_speaker_majority_voting(self):
        segs = [
            seg(0, 0.5, " A", speaker="Alice"),
            seg(0.6, 1.0, " B", speaker="Alice"),
            seg(1.1, 1.5, " C.", speaker="Bob"),
        ]
        result = merge_sentences(segs)
        # Merged segment should have Alice (2 vs 1)
        assert result[0]["speaker"] == "Alice"


# ── detect_chapters ───────────────────────────────────────────────────────────


class TestDetectChapters:
    def test_empty_segments_returns_empty(self):
        assert detect_chapters([]) == []

    def test_no_gaps_single_chapter(self):
        segs = [seg(0, 1, "A"), seg(2, 3, "B")]
        chapters = detect_chapters(segs, min_gap=8.0)
        assert len(chapters) == 1
        assert chapters[0]["chapter"] == 1
        assert chapters[0]["start"] == 0

    def test_large_gap_creates_second_chapter(self):
        segs = [seg(0, 1, "A"), seg(15, 16, "B")]
        chapters = detect_chapters(segs, min_gap=8.0)
        assert len(chapters) == 2
        assert chapters[1]["chapter"] == 2
        assert chapters[1]["start"] == 15

    def test_chapter_titles_sequential(self):
        segs = [seg(0, 1, "A"), seg(20, 21, "B"), seg(40, 41, "C")]
        chapters = detect_chapters(segs, min_gap=8.0)
        assert chapters[0]["title"] == "Chapter 1"
        assert chapters[1]["title"] == "Chapter 2"
        assert chapters[2]["title"] == "Chapter 3"

    def test_gap_below_threshold_no_chapter(self):
        segs = [seg(0, 1, "A"), seg(5, 6, "B")]
        chapters = detect_chapters(segs, min_gap=8.0)
        assert len(chapters) == 1


# ── search_transcript ─────────────────────────────────────────────────────────


class TestSearchTranscript:
    def test_exact_match(self):
        segs = [seg(0, 1, "Hello world"), seg(2, 3, "foo bar")]
        matches = search_transcript(segs, "Hello")
        assert len(matches) == 1
        assert matches[0]["start"] == 0

    def test_case_insensitive(self):
        segs = [seg(0, 1, "Hello World")]
        assert len(search_transcript(segs, "hello world")) == 1

    def test_no_match_returns_empty(self):
        segs = [seg(0, 1, "Hello world")]
        assert search_transcript(segs, "xyz") == []

    def test_fuzzy_match_typo(self):
        segs = [seg(0, 1, "Hello wrold")]  # typo
        matches = search_transcript(segs, "world", fuzzy=True)
        assert len(matches) == 1

    def test_fuzzy_multi_word(self):
        segs = [seg(0, 1, "the quick brown fox")]
        matches = search_transcript(segs, "quik bown", fuzzy=True)
        assert len(matches) == 1

    def test_match_result_has_required_fields(self):
        segs = [seg(0, 1, "Hello world", speaker="Alice")]
        matches = search_transcript(segs, "Hello")
        m = matches[0]
        assert "start" in m and "end" in m and "text" in m and "speaker" in m

    def test_no_fuzzy_match_without_flag(self):
        segs = [seg(0, 1, "Hello wrold")]
        assert search_transcript(segs, "world", fuzzy=False) == []
