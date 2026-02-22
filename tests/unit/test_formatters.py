"""Unit tests for lib/formatters.py."""
from __future__ import annotations

import json
import math

import pytest

from lib.formatters import (
    format_agent_json,
    format_ts_ass,
    format_ts_srt,
    format_ts_ttml,
    format_ts_vtt,
    split_words_by_chars,
    to_ass,
    to_csv,
    to_html,
    to_json,
    to_lrc,
    to_srt,
    to_text,
    to_tsv,
    to_ttml,
    to_vtt,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def two_segs():
    return [
        {"start": 0.0, "end": 1.5, "text": " Hello world."},
        {"start": 2.0, "end": 3.5, "text": " How are you?"},
    ]


@pytest.fixture
def speaker_segs():
    return [
        {"start": 0.0, "end": 1.0, "text": " Hi.", "speaker": "SPEAKER_1"},
        {"start": 1.5, "end": 2.5, "text": " Hello.", "speaker": "SPEAKER_2"},
    ]


@pytest.fixture
def result_dict(two_segs):
    return {
        "text": "Hello world. How are you?",
        "language": "en",
        "duration": 3.5,
        "segments": two_segs,
        "file": "test.wav",
    }


def _words(*texts):
    """Build minimal word-level dicts for testing."""
    return [
        {"word": t, "start": i * 0.5, "end": i * 0.5 + 0.4}
        for i, t in enumerate(texts)
    ]


# ── Timestamp helpers ─────────────────────────────────────────────────────────

class TestFormatTsSrt:
    def test_zero(self):
        assert format_ts_srt(0) == "00:00:00,000"

    def test_one_hour(self):
        assert format_ts_srt(3600) == "01:00:00,000"

    def test_half_second(self):
        assert format_ts_srt(0.5) == "00:00:00,500"

    def test_complex(self):
        # 3661.25 = 1h 1m 1.25s
        assert format_ts_srt(3661.25) == "01:01:01,250"

    def test_uses_comma_separator(self):
        assert "," in format_ts_srt(1.0)


class TestFormatTsVtt:
    def test_zero(self):
        assert format_ts_vtt(0) == "00:00:00.000"

    def test_uses_dot_separator(self):
        assert "." in format_ts_vtt(1.0)
        assert "," not in format_ts_vtt(1.0)

    def test_milliseconds(self):
        assert format_ts_vtt(1.5) == "00:00:01.500"


class TestFormatTsAss:
    def test_zero(self):
        assert format_ts_ass(0) == "0:00:00.00"

    def test_centiseconds_not_milliseconds(self):
        # 1.25s → 25 centiseconds
        assert format_ts_ass(1.25) == "0:00:01.25"
        # 1.5s → 50 centiseconds
        assert format_ts_ass(1.5) == "0:00:01.50"

    def test_hour_uses_single_digit(self):
        assert format_ts_ass(3661.0).startswith("1:")


class TestFormatTsTtml:
    def test_zero(self):
        assert format_ts_ttml(0) == "00:00:00.000"

    def test_milliseconds(self):
        assert format_ts_ttml(1.5) == "00:00:01.500"

    def test_matches_vtt_format(self):
        # TTML and VTT both use HH:MM:SS.mmm with dot separator
        assert format_ts_ttml(3661.0) == "01:01:01.000"


# ── split_words_by_chars ──────────────────────────────────────────────────────

class TestSplitWordsByChars:
    def test_empty_returns_empty_chunk(self):
        assert split_words_by_chars([], 20) == [[]]

    def test_all_fits_in_one_chunk(self):
        words = _words("Hi", " there")
        assert len(split_words_by_chars(words, 20)) == 1

    def test_splits_on_boundary(self):
        # "Hello" (5) + " world" (6) = 11 chars; limit 8 → splits
        words = _words("Hello", " world", " foo", " bar")
        chunks = split_words_by_chars(words, 8)
        assert len(chunks) >= 2

    def test_single_word_exceeding_limit_stays_as_chunk(self):
        words = _words("Superlongword")
        chunks = split_words_by_chars(words, 5)
        assert len(chunks) == 1  # can't split a single word

    def test_each_chunk_respects_limit(self):
        words = _words("aa", " bb", " cc", " dd", " ee")
        for chunk in split_words_by_chars(words, 6):
            total = sum(len(w["word"]) for w in chunk)
            # first word in chunk may exceed limit; rest are guarded
            assert total <= 6 or len(chunk) == 1


# ── to_srt ────────────────────────────────────────────────────────────────────

class TestToSrt:
    def test_first_cue_number(self, two_segs):
        lines = to_srt(two_segs).split("\n")
        assert lines[0] == "1"

    def test_arrow_in_timing_line(self, two_segs):
        lines = to_srt(two_segs).split("\n")
        assert "-->" in lines[1]

    def test_blank_line_after_each_cue(self, two_segs):
        srt = to_srt(two_segs)
        # Every cue block ends with a blank line
        assert "\n\n" in srt

    def test_two_cues_numbered(self, two_segs):
        srt = to_srt(two_segs)
        assert "\n2\n" in srt

    def test_text_present(self, two_segs):
        srt = to_srt(two_segs)
        assert "Hello world." in srt

    def test_speaker_label(self, speaker_segs):
        srt = to_srt(speaker_segs)
        assert "[SPEAKER_1]" in srt

    def test_empty_segments(self):
        assert to_srt([]) == ""


# ── to_vtt ────────────────────────────────────────────────────────────────────

class TestToVtt:
    def test_webvtt_header(self, two_segs):
        assert to_vtt(two_segs).startswith("WEBVTT")

    def test_dot_separator_in_timestamps(self, two_segs):
        vtt = to_vtt(two_segs)
        assert "00:00:00.000" in vtt

    def test_speaker_label(self, speaker_segs):
        assert "[SPEAKER_1]" in to_vtt(speaker_segs)

    def test_empty_segments(self):
        # to_vtt([]) returns the WEBVTT header only (no cue blocks)
        assert to_vtt([]).startswith("WEBVTT")


# ── to_text ───────────────────────────────────────────────────────────────────

class TestToText:
    def test_segments_joined(self, two_segs):
        txt = to_text(two_segs)
        assert "Hello world." in txt
        assert "How are you?" in txt

    def test_no_speaker_labels_when_absent(self, two_segs):
        txt = to_text(two_segs)
        assert "[" not in txt

    def test_speaker_label_present(self, speaker_segs):
        txt = to_text(speaker_segs)
        assert "SPEAKER_1" in txt
        assert "SPEAKER_2" in txt

    def test_speaker_transition_shown_once(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": " A.", "speaker": "Alice"},
            {"start": 1.5, "end": 2.5, "text": " B.", "speaker": "Alice"},
            {"start": 3.0, "end": 4.0, "text": " C.", "speaker": "Bob"},
        ]
        txt = to_text(segs)
        assert txt.count("[Alice]") == 1  # only on first occurrence
        assert txt.count("[Bob]") == 1

    def test_paragraph_break_inserted(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": "First.", "paragraph_start": True},
            {"start": 5.0, "end": 6.0, "text": "Second.", "paragraph_start": True},
        ]
        txt = to_text(segs)
        assert "\n\n" in txt


# ── to_csv ────────────────────────────────────────────────────────────────────

class TestToCsv:
    def test_header_row(self, two_segs):
        first_line = to_csv(two_segs).split("\n")[0]
        assert first_line == "start_s,end_s,text"

    def test_no_speaker_column_when_absent(self, two_segs):
        assert "speaker" not in to_csv(two_segs).split("\n")[0]

    def test_speaker_column_present(self, speaker_segs):
        assert "speaker" in to_csv(speaker_segs).split("\n")[0]

    def test_three_decimal_timestamps(self, two_segs):
        csv_str = to_csv(two_segs)
        assert "0.000" in csv_str

    def test_row_count(self, two_segs):
        lines = [line for line in to_csv(two_segs).split("\n") if line]
        assert len(lines) == 3  # header + 2 data rows


# ── to_tsv ────────────────────────────────────────────────────────────────────

class TestToTsv:
    def test_three_tab_fields(self, two_segs):
        first_line = to_tsv(two_segs).split("\n")[0]
        assert len(first_line.split("\t")) == 3

    def test_millisecond_start_timestamp(self, two_segs):
        # seg start=0.0 → 0 ms
        assert to_tsv(two_segs).startswith("0\t")

    def test_end_ms_computed(self, two_segs):
        # seg end=1.5 → 1500 ms
        first_line = to_tsv(two_segs).split("\n")[0]
        assert first_line.split("\t")[1] == "1500"


# ── to_lrc ────────────────────────────────────────────────────────────────────

class TestToLrc:
    def test_bracket_timestamp_format(self, two_segs):
        first_line = to_lrc(two_segs).split("\n")[0]
        assert first_line.startswith("[00:00.00]")

    def test_speaker_label_in_lrc(self, speaker_segs):
        assert "[SPEAKER_1]" in to_lrc(speaker_segs)

    def test_two_lines_for_two_segs(self, two_segs):
        assert len(to_lrc(two_segs).split("\n")) == 2


# ── to_ass ────────────────────────────────────────────────────────────────────

class TestToAss:
    def test_script_info_header(self, two_segs):
        ass = to_ass(two_segs)
        assert "[Script Info]" in ass

    def test_events_section_present(self, two_segs):
        ass = to_ass(two_segs)
        assert "[Events]" in ass

    def test_dialogue_lines_present(self, two_segs):
        ass = to_ass(two_segs)
        assert ass.count("Dialogue:") == 2

    def test_ass_timestamp_format(self, two_segs):
        # ASS timestamps use H:MM:SS.cc (centiseconds, single-digit hour)
        ass = to_ass(two_segs)
        assert "0:00:00.00" in ass

    def test_text_content_present(self, two_segs):
        ass = to_ass(two_segs)
        assert "Hello world." in ass

    def test_speaker_label_included(self, speaker_segs):
        ass = to_ass(speaker_segs)
        assert "[SPEAKER_1]" in ass

    def test_empty_segments(self):
        # Only header lines, no Dialogue entries
        ass = to_ass([])
        assert "Dialogue:" not in ass
        assert "[Script Info]" in ass

    def test_max_words_per_line_splits_dialogue(self):
        segs = [{
            "start": 0.0, "end": 2.0, "text": " One two three four.",
            "words": [
                {"word": " One", "start": 0.0, "end": 0.4},
                {"word": " two", "start": 0.4, "end": 0.8},
                {"word": " three", "start": 0.8, "end": 1.2},
                {"word": " four.", "start": 1.2, "end": 2.0},
            ],
        }]
        ass = to_ass(segs, max_words_per_line=2)
        assert ass.count("Dialogue:") == 2


# ── to_ttml ───────────────────────────────────────────────────────────────────

class TestToTtml:
    def test_xml_declaration(self, two_segs):
        ttml = to_ttml(two_segs)
        assert ttml.startswith('<?xml version="1.0"')

    def test_tt_root_element(self, two_segs):
        ttml = to_ttml(two_segs)
        assert "<tt " in ttml

    def test_body_and_div_present(self, two_segs):
        ttml = to_ttml(two_segs)
        assert "<body>" in ttml
        assert "<div " in ttml

    def test_p_elements_for_segments(self, two_segs):
        ttml = to_ttml(two_segs)
        assert ttml.count("<p ") == 2

    def test_ttml_timestamp_format(self, two_segs):
        # TTML timestamps use HH:MM:SS.mmm with dot separator
        ttml = to_ttml(two_segs)
        assert 'begin="00:00:00.000"' in ttml

    def test_text_content_present(self, two_segs):
        ttml = to_ttml(two_segs)
        assert "Hello world." in ttml

    def test_speaker_label_included(self, speaker_segs):
        ttml = to_ttml(speaker_segs)
        assert "[SPEAKER_1]" in ttml

    def test_language_attribute(self, two_segs):
        ttml = to_ttml(two_segs, language="fr")
        assert 'xml:lang="fr"' in ttml

    def test_language_underscore_converted_to_hyphen(self, two_segs):
        ttml = to_ttml(two_segs, language="en_US")
        assert 'xml:lang="en-US"' in ttml

    def test_xml_escaping_in_text(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "<b>Hello & World</b>"}]
        ttml = to_ttml(segs)
        assert "<b>Hello & World</b>" not in ttml
        assert "&lt;b&gt;" in ttml
        assert "&amp;" in ttml

    def test_max_words_per_line_splits_p_elements(self):
        segs = [{
            "start": 0.0, "end": 2.0, "text": " One two three four.",
            "words": [
                {"word": " One", "start": 0.0, "end": 0.4},
                {"word": " two", "start": 0.4, "end": 0.8},
                {"word": " three", "start": 0.8, "end": 1.2},
                {"word": " four.", "start": 1.2, "end": 2.0},
            ],
        }]
        ttml = to_ttml(segs, max_words_per_line=2)
        assert ttml.count("<p ") == 2


# ── to_html ───────────────────────────────────────────────────────────────────

class TestToHtml:
    def test_valid_html_structure(self, result_dict):
        html = to_html(result_dict)
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html

    def test_conf_high_class(self):
        result = {
            "file": "t.wav", "language": "en", "duration": 1.0,
            "segments": [{
                "start": 0.0, "end": 1.0, "text": "hi",
                "words": [{"word": "hi", "start": 0.0, "end": 0.5, "probability": 0.95}],
            }],
        }
        assert "conf-high" in to_html(result)

    def test_conf_med_class(self):
        result = {
            "file": "t.wav", "language": "en", "duration": 1.0,
            "segments": [{
                "start": 0.0, "end": 1.0, "text": "hi",
                "words": [{"word": "hi", "start": 0.0, "end": 0.5, "probability": 0.75}],
            }],
        }
        assert "conf-med" in to_html(result)

    def test_conf_low_class(self):
        result = {
            "file": "t.wav", "language": "en", "duration": 1.0,
            "segments": [{
                "start": 0.0, "end": 1.0, "text": "hi",
                "words": [{"word": "hi", "start": 0.0, "end": 0.5, "probability": 0.5}],
            }],
        }
        assert "conf-low" in to_html(result)

    def test_segment_level_fallback(self, result_dict):
        # Segments without words use plain text, not word spans
        html = to_html(result_dict)
        assert "Hello world." in html

    def test_xss_safe_angle_brackets(self):
        # AIDEV-NOTE: If this test fails, to_html does not escape segment text — XSS bug
        result = {
            "file": "t.wav", "language": "en", "duration": 1.0,
            "segments": [{"start": 0.0, "end": 1.0, "text": "<script>alert(1)</script>"}],
        }
        html = to_html(result)
        # Raw <script> tag must not appear unescaped
        assert "<script>alert(1)</script>" not in html

    def test_xss_safe_word_level(self):
        # AIDEV-NOTE: Verifies to_html escapes word-level text — word-level XSS fix
        result = {
            "file": "t.wav", "language": "en", "duration": 1.0,
            "segments": [{
                "start": 0.0, "end": 1.0, "text": "<script>",
                "words": [{"word": "<script>", "start": 0.0, "end": 0.5, "probability": 0.9}],
            }],
        }
        html = to_html(result)
        assert "<script>" not in html


# ── to_json ───────────────────────────────────────────────────────────────────

class TestToJson:
    def test_valid_json(self, result_dict):
        parsed = json.loads(to_json(result_dict))
        assert parsed["language"] == "en"

    def test_segments_present(self, result_dict):
        parsed = json.loads(to_json(result_dict))
        assert "segments" in parsed
        assert len(parsed["segments"]) == 2

    def test_roundtrippable(self, result_dict):
        parsed = json.loads(to_json(result_dict))
        assert json.loads(json.dumps(parsed)) == parsed


# ── format_agent_json ─────────────────────────────────────────────────────────

class TestFormatAgentJson:
    def test_required_fields(self, result_dict):
        j = json.loads(format_agent_json(result_dict, "parakeet"))
        for field in ("text", "duration", "language", "backend", "segments", "word_count"):
            assert field in j, f"missing field: {field}"
        assert isinstance(j["segments"], int), "segments should be an integer count"

    def test_backend_echoed(self, result_dict):
        j = json.loads(format_agent_json(result_dict, "faster-whisper"))
        assert j["backend"] == "faster-whisper"

    def test_single_line_json(self, result_dict):
        output = format_agent_json(result_dict, "parakeet")
        assert "\n" not in output

    def test_no_summary_hint_for_short_transcript(self, result_dict):
        # Under 400 chars → no summary_hint
        j = json.loads(format_agent_json(result_dict, "parakeet"))
        assert "summary_hint" not in j

    def test_summary_hint_for_long_transcript(self):
        long_text = "This is a fairly long sentence with many words. " * 10
        segs = [
            {"start": float(i), "end": float(i) + 0.9, "text": f" {long_text}"}
            for i in range(10)
        ]
        result = {
            "text": long_text * 10, "language": "en",
            "duration": 10.0, "segments": segs,
        }
        j = json.loads(format_agent_json(result, "parakeet"))
        assert "summary_hint" in j
        assert "first" in j["summary_hint"]
        assert "last" in j["summary_hint"]

    def test_avg_confidence_from_word_probability(self):
        result = {
            "text": "hello", "language": "en", "duration": 1.0,
            "segments": [{
                "start": 0.0, "end": 1.0, "text": "hello",
                "words": [{"word": "hello", "start": 0.0, "end": 1.0, "probability": 0.9}],
            }],
        }
        j = json.loads(format_agent_json(result, "fw"))
        assert "avg_confidence" in j
        assert abs(j["avg_confidence"] - 0.9) < 0.01

    def test_avg_confidence_from_avg_logprob(self):
        logprob = math.log(0.8)
        result = {
            "text": "hello", "language": "en", "duration": 1.0,
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello", "avg_logprob": logprob}],
        }
        j = json.loads(format_agent_json(result, "fw"))
        assert "avg_confidence" in j
        assert abs(j["avg_confidence"] - 0.8) < 0.01

    def test_file_path_echoed(self):
        result = {
            "text": "hi", "language": "en", "duration": 1.0,
            "segments": [], "file_path": "/audio/test.mp3",
        }
        j = json.loads(format_agent_json(result, "fw"))
        assert j["file_path"] == "/audio/test.mp3"

    def test_output_path_echoed(self):
        result = {
            "text": "hi", "language": "en", "duration": 1.0,
            "segments": [], "output_path": "/out/test.txt",
        }
        j = json.loads(format_agent_json(result, "fw"))
        assert j["output_path"] == "/out/test.txt"

    def test_word_count_computed(self, two_segs):
        result = {"text": "", "language": "en", "duration": 3.5, "segments": two_segs}
        j = json.loads(format_agent_json(result, "fw"))
        # "Hello world." = 2 words; "How are you?" = 3 words
        assert j["word_count"] == 5
