# Testing Implementation Plan — super-transcribe

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a comprehensive pytest suite covering all shared lib modules (fast unit tests) plus both ML backends end-to-end (model-gated integration tests).

**Architecture:** Two layers — fast modelless unit tests run by default (`pytest`); model-gated integration tests auto-skip when backends aren't installed (`pytest -m integration`). Shared lib modules are imported directly via `sys.path` injection in `conftest.py`. The bash router is tested via subprocess against stub backends living in a temporary directory. A committed 1-second 440Hz tone WAV is used as the audio fixture across both layers.

**Tech Stack:** pytest, pytest-mock, numpy, stdlib wave/json/subprocess, bash

---

## Prerequisite reading

Before starting, skim these files:
- `scripts/backends/lib/formatters.py` — 10 format functions + agent JSON
- `scripts/backends/lib/postprocess.py` — 6 postprocessing functions
- `scripts/backends/lib/audio.py` — probe, convert, preprocess, channel, resolve
- `scripts/backends/lib/speakers.py` — speaker name mapping + export
- `scripts/backends/lib/rss.py` — RSS feed parser
- `scripts/transcribe` — bash router (routing arrays, `backend_available`, `backend_ready`)

---

### Task 0: Scaffold test infrastructure

**Files:**
- Create: `pytest.ini`
- Create: `requirements-test.txt`
- Create: `tests/conftest.py`
- Create: `tests/unit/` (directory)
- Create: `tests/integration/` (directory)
- Create: `tests/fixtures/` (auto-created by conftest.py)

**Step 1: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
markers =
    unit: Unit tests (fast, modelless)
    integration: End-to-end tests (requires backends installed)
addopts = -m "not integration"
```

**Step 2: Create `requirements-test.txt`**

```
pytest>=8.0
pytest-mock>=3.12
numpy>=1.24
```

**Step 3: Create `tests/conftest.py`**

```python
"""
Root conftest: injects scripts/backends onto sys.path and generates tone.wav.
"""
from __future__ import annotations

import math
import struct
import sys
import wave
from pathlib import Path

# AIDEV-NOTE: All unit tests import lib.* via this sys.path injection
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "backends"))

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TONE_WAV = FIXTURES_DIR / "tone.wav"


def _generate_tone_wav() -> None:
    """Write a 1-second 440 Hz sine wave at 16 kHz mono PCM if absent."""
    FIXTURES_DIR.mkdir(exist_ok=True)
    if TONE_WAV.exists():
        return
    sample_rate = 16_000
    frequency = 440.0
    n_samples = sample_rate  # 1 second
    with wave.open(str(TONE_WAV), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            v = int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
            wf.writeframes(struct.pack("<h", v))


_generate_tone_wav()
```

**Step 4: Install test deps and verify scaffold**

```bash
pip install -r requirements-test.txt
pytest --collect-only
```

Expected: `no tests ran`, 0 errors, `tests/fixtures/tone.wav` now exists on disk.

**Step 5: Commit tone.wav and scaffold**

```bash
git add pytest.ini requirements-test.txt tests/conftest.py tests/fixtures/tone.wav
git commit -m "test: scaffold pytest infra + committed tone.wav fixture"
```

---

### Task 1: test_formatters.py

**Files:**
- Create: `tests/unit/test_formatters.py`

**Step 1: Write the test file**

```python
"""Unit tests for lib/formatters.py."""
from __future__ import annotations

import json

import pytest

from lib.formatters import (
    format_agent_json,
    format_ts_ass,
    format_ts_srt,
    format_ts_ttml,
    format_ts_vtt,
    split_words_by_chars,
    to_csv,
    to_html,
    to_json,
    to_lrc,
    to_srt,
    to_text,
    to_tsv,
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
        lines = [l for l in to_csv(two_segs).split("\n") if l]
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
        import math
        logprob = math.log(0.8)  # avg_logprob → exp → 0.8
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
```

**Step 2: Run tests**

```bash
pytest tests/unit/test_formatters.py -v
```

Expected: mostly PASS. Watch for:
- `test_xss_safe_angle_brackets` — may FAIL if `to_html` doesn't escape plain text segments. If so, see the fix note below.

**Step 2a (if XSS test fails): Fix `to_html` in `scripts/backends/lib/formatters.py`**

Locate the `else` branch in `to_html` (around line 464):
```python
else:
    text_html = seg.get("text", "").strip()
```
Replace with:
```python
else:
    import html as _html
    text_html = _html.escape(seg.get("text", "").strip())
```

**Step 3: Run again to confirm all pass**

```bash
pytest tests/unit/test_formatters.py -v
```

Expected: all PASS.

**Step 4: Commit**

```bash
git add tests/unit/test_formatters.py scripts/backends/lib/formatters.py
git commit -m "test: add test_formatters.py (all 10 formats + helpers + agent JSON)"
```

---

### Task 2: test_postprocess.py

**Files:**
- Create: `tests/unit/test_postprocess.py`

**Step 1: Write the test file**

```python
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


# ── filter_hallucinations ─────────────────────────────────────────────────────

class TestFilterHallucinations:
    @pytest.mark.parametrize("text", [
        "[music]", "[Music]", "[MUSIC]",
        "[applause]", "[laughter]", "[silence]",
        "[inaudible]", "[background noise]",
        "(music)", "(upbeat music)", "(dramatic music)",
        "Thank you for watching",
        "thank you for listening",
        "subtitles by SomeGroup",
        "transcribed by AI",
        "www.example.com",
        "...",
        "",
        "   ",
    ])
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
        # gap=2.0 < min_gap=3.0, but sentence ends + gap >= sentence_gap=1.5
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
        # gap=4.5 > MAX_GAP=2.0 → flush
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
```

**Step 2: Run tests**

```bash
pytest tests/unit/test_postprocess.py -v
```

Expected: all PASS. If any fail, check the specific assertion against the source in `lib/postprocess.py`.

**Step 3: Commit**

```bash
git add tests/unit/test_postprocess.py
git commit -m "test: add test_postprocess.py (filter, filler, paragraphs, merge, chapters, search)"
```

---

### Task 3: test_audio.py

**Files:**
- Create: `tests/unit/test_audio.py`

**Step 1: Write the test file**

```python
"""Unit tests for lib/audio.py — all subprocess calls mocked."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lib.audio import (
    convert_to_wav,
    extract_channel,
    is_url,
    preprocess_audio,
    probe_audio,
    resolve_inputs,
)


# ── is_url ────────────────────────────────────────────────────────────────────

class TestIsUrl:
    def test_http(self):
        assert is_url("http://example.com/audio.mp3")

    def test_https(self):
        assert is_url("https://example.com/audio.mp3")

    def test_www(self):
        assert is_url("www.example.com/audio.mp3")

    def test_absolute_path(self):
        assert not is_url("/path/to/audio.mp3")

    def test_relative_path(self):
        assert not is_url("audio.mp3")

    def test_no_prefix(self):
        assert not is_url("example.com/audio.mp3")


# ── resolve_inputs ────────────────────────────────────────────────────────────

class TestResolveInputs:
    def test_url_passthrough(self):
        result = resolve_inputs(["https://example.com/audio.mp3"])
        assert result == ["https://example.com/audio.mp3"]

    def test_single_audio_file(self, tmp_path):
        f = tmp_path / "audio.mp3"
        f.touch()
        assert str(f) in resolve_inputs([str(f)])

    def test_directory_traversal_audio_only(self, tmp_path):
        (tmp_path / "a.mp3").touch()
        (tmp_path / "b.wav").touch()
        (tmp_path / "readme.txt").touch()
        result = resolve_inputs([str(tmp_path)])
        assert len(result) == 2
        assert all(p.endswith((".mp3", ".wav")) for p in result)

    def test_non_audio_file_warning(self, tmp_path, capsys):
        f = tmp_path / "readme.txt"
        f.touch()
        resolve_inputs([str(f)])
        captured = capsys.readouterr()
        assert "skipping" in captured.err.lower() or "warning" in captured.err.lower()

    def test_missing_file_warning(self, capsys):
        resolve_inputs(["/no/such/file.mp3"])
        captured = capsys.readouterr()
        assert captured.err  # some warning emitted


# ── probe_audio ───────────────────────────────────────────────────────────────

def _ffprobe_json(duration="2.5", channels=1, sample_rate="16000"):
    return json.dumps({
        "format": {
            "duration": duration,
            "bit_rate": "256000",
            "format_name": "wav",
        },
        "streams": [{
            "codec_type": "audio",
            "channels": channels,
            "sample_rate": sample_rate,
        }],
    })


class TestProbeAudio:
    def test_returns_expected_keys(self, tmp_path):
        f = tmp_path / "test.wav"
        f.write_bytes(b"\x00" * 44)

        with patch("lib.audio.shutil.which", return_value="/usr/bin/ffprobe"), \
             patch("lib.audio.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=_ffprobe_json(), returncode=0)
            result = probe_audio(str(f))

        assert result["duration"] == 2.5
        assert result["channels"] == 1
        assert result["sample_rate"] == 16000
        assert result["file"] == "test.wav"

    def test_duration_human_seconds(self, tmp_path):
        f = tmp_path / "test.wav"
        f.write_bytes(b"\x00" * 44)

        with patch("lib.audio.shutil.which", return_value="/usr/bin/ffprobe"), \
             patch("lib.audio.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=_ffprobe_json(duration="45.0"))
            result = probe_audio(str(f))

        assert result["duration_human"] == "45s"

    def test_duration_human_minutes(self, tmp_path):
        f = tmp_path / "test.wav"
        f.write_bytes(b"\x00" * 44)

        with patch("lib.audio.shutil.which", return_value="/usr/bin/ffprobe"), \
             patch("lib.audio.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=_ffprobe_json(duration="125.0"))
            result = probe_audio(str(f))

        assert "m" in result["duration_human"]
        assert "s" in result["duration_human"]

    def test_duration_human_hours(self, tmp_path):
        f = tmp_path / "test.wav"
        f.write_bytes(b"\x00" * 44)

        with patch("lib.audio.shutil.which", return_value="/usr/bin/ffprobe"), \
             patch("lib.audio.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=_ffprobe_json(duration="7261.0"))
            result = probe_audio(str(f))

        assert "h" in result["duration_human"]

    def test_fallback_when_ffprobe_absent(self, tmp_path):
        f = tmp_path / "test.wav"
        f.write_bytes(b"\x00" * 44)

        with patch("lib.audio.shutil.which", return_value=None):
            result = probe_audio(str(f))

        # Must return a dict with standard keys, even without ffprobe
        for key in ("file", "duration", "duration_human", "format"):
            assert key in result


# ── convert_to_wav ────────────────────────────────────────────────────────────

class TestConvertToWav:
    def test_wav_is_passthrough(self, tmp_path):
        f = tmp_path / "audio.wav"
        f.touch()
        path, tmp = convert_to_wav(str(f))
        assert path == str(f) and tmp is None

    def test_flac_is_passthrough(self, tmp_path):
        f = tmp_path / "audio.flac"
        f.touch()
        path, tmp = convert_to_wav(str(f))
        assert path == str(f) and tmp is None

    def test_mp3_calls_ffmpeg(self, tmp_path):
        f = tmp_path / "audio.mp3"
        f.touch()

        with patch("lib.audio.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("lib.audio.subprocess.run") as mock_run, \
             patch("lib.audio.os.path.exists", return_value=False):
            mock_run.return_value = MagicMock(returncode=0)
            convert_to_wav(str(f), quiet=True)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-ar" in cmd and "16000" in cmd

    def test_ffmpeg_failure_returns_original(self, tmp_path):
        f = tmp_path / "audio.mp3"
        f.touch()

        with patch("lib.audio.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("lib.audio.subprocess.run") as mock_run, \
             patch("lib.audio.os.path.exists", return_value=False):
            mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg")
            path, tmp = convert_to_wav(str(f), quiet=True)

        assert path == str(f) and tmp is None

    def test_no_ffmpeg_returns_original(self, tmp_path):
        f = tmp_path / "audio.mp3"
        f.touch()

        with patch("lib.audio.shutil.which", return_value=None):
            path, tmp = convert_to_wav(str(f), quiet=True)

        assert path == str(f) and tmp is None


# ── preprocess_audio ──────────────────────────────────────────────────────────

class TestPreprocessAudio:
    def test_no_flags_is_noop(self, tmp_path):
        f = tmp_path / "audio.wav"
        f.touch()
        path, tmp = preprocess_audio(str(f), normalize=False, denoise=False)
        assert path == str(f) and tmp is None

    def test_normalize_includes_loudnorm(self, tmp_path):
        f = tmp_path / "audio.wav"
        f.touch()

        with patch("lib.audio.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("lib.audio.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            preprocess_audio(str(f), normalize=True, quiet=True)

        cmd = " ".join(mock_run.call_args[0][0])
        assert "loudnorm" in cmd

    def test_denoise_includes_afftdn(self, tmp_path):
        f = tmp_path / "audio.wav"
        f.touch()

        with patch("lib.audio.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("lib.audio.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            preprocess_audio(str(f), denoise=True, quiet=True)

        cmd = " ".join(mock_run.call_args[0][0])
        assert "afftdn" in cmd

    def test_both_flags_combined(self, tmp_path):
        f = tmp_path / "audio.wav"
        f.touch()

        with patch("lib.audio.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("lib.audio.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            preprocess_audio(str(f), normalize=True, denoise=True, quiet=True)

        cmd = " ".join(mock_run.call_args[0][0])
        assert "loudnorm" in cmd and "afftdn" in cmd

    def test_ffmpeg_failure_returns_original(self, tmp_path):
        f = tmp_path / "audio.wav"
        f.touch()

        with patch("lib.audio.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("lib.audio.subprocess.run") as mock_run, \
             patch("lib.audio.os.path.exists", return_value=False):
            mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg")
            path, tmp = preprocess_audio(str(f), normalize=True, quiet=True)

        assert path == str(f) and tmp is None


# ── extract_channel ───────────────────────────────────────────────────────────

class TestExtractChannel:
    def test_mix_is_passthrough(self, tmp_path):
        f = tmp_path / "audio.wav"
        f.touch()
        path, tmp = extract_channel(str(f), "mix")
        assert path == str(f) and tmp is None

    def test_left_uses_c0_filter(self, tmp_path):
        f = tmp_path / "audio.wav"
        f.touch()

        with patch("lib.audio.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("lib.audio.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            extract_channel(str(f), "left", quiet=True)

        cmd = " ".join(mock_run.call_args[0][0])
        assert "c0=c0" in cmd

    def test_right_uses_c1_filter(self, tmp_path):
        f = tmp_path / "audio.wav"
        f.touch()

        with patch("lib.audio.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("lib.audio.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            extract_channel(str(f), "right", quiet=True)

        cmd = " ".join(mock_run.call_args[0][0])
        assert "c0=c1" in cmd

    def test_no_ffmpeg_returns_original(self, tmp_path):
        f = tmp_path / "audio.wav"
        f.touch()

        with patch("lib.audio.shutil.which", return_value=None):
            path, tmp = extract_channel(str(f), "left", quiet=True)

        assert path == str(f) and tmp is None
```

**Step 2: Run tests**

```bash
pytest tests/unit/test_audio.py -v
```

Expected: all PASS.

**Step 3: Commit**

```bash
git add tests/unit/test_audio.py
git commit -m "test: add test_audio.py (probe, convert, preprocess, channel, resolve)"
```

---

### Task 4: test_speakers.py

**Files:**
- Create: `tests/unit/test_speakers.py`

**Step 1: Write the test file**

```python
"""Unit tests for lib/speakers.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lib.speakers import apply_speaker_names, export_speakers_audio


def seg(start, end, text, speaker=None, words=None):
    s = {"start": start, "end": end, "text": text}
    if speaker:
        s["speaker"] = speaker
    if words:
        s["words"] = words
    return s


# ── apply_speaker_names ───────────────────────────────────────────────────────

class TestApplySpeakerNames:
    def test_speaker_1_mapped_to_first_name(self):
        segs = [seg(0, 1, "Hi", speaker="SPEAKER_1")]
        result = apply_speaker_names(segs, "Alice,Bob")
        assert result[0]["speaker"] == "Alice"

    def test_speaker_2_mapped_to_second_name(self):
        segs = [seg(0, 1, "Hi", speaker="SPEAKER_2")]
        result = apply_speaker_names(segs, "Alice,Bob")
        assert result[0]["speaker"] == "Bob"

    def test_out_of_range_keeps_raw_label(self):
        segs = [seg(0, 1, "Hi", speaker="SPEAKER_5")]
        result = apply_speaker_names(segs, "Alice")
        assert result[0]["speaker"] == "SPEAKER_5"

    def test_word_level_speaker_updated(self):
        words = [{"word": " hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_1"}]
        segs = [seg(0, 1, " hello", speaker="SPEAKER_1", words=words)]
        result = apply_speaker_names(segs, "Alice,Bob")
        assert result[0]["words"][0]["speaker"] == "Alice"

    def test_word_without_speaker_not_modified(self):
        words = [{"word": " hello", "start": 0.0, "end": 0.5}]
        segs = [seg(0, 1, " hello", speaker="SPEAKER_1", words=words)]
        result = apply_speaker_names(segs, "Alice")
        # word has no "speaker" key → it should not gain one
        assert "speaker" not in result[0]["words"][0]

    def test_no_speaker_segments_unchanged(self):
        segs = [seg(0, 1, "Hi")]
        result = apply_speaker_names(segs, "Alice")
        assert result[0] == segs[0]

    def test_multiple_segments_same_speaker_mapped_once(self):
        segs = [
            seg(0, 1, "A", speaker="SPEAKER_1"),
            seg(1, 2, "B", speaker="SPEAKER_1"),
        ]
        result = apply_speaker_names(segs, "Alice")
        assert result[0]["speaker"] == "Alice"
        assert result[1]["speaker"] == "Alice"

    def test_whitespace_in_names_stripped(self):
        segs = [seg(0, 1, "Hi", speaker="SPEAKER_1")]
        result = apply_speaker_names(segs, " Alice , Bob ")
        assert result[0]["speaker"] == "Alice"


# ── export_speakers_audio ─────────────────────────────────────────────────────

class TestExportSpeakersAudio:
    def test_no_ffmpeg_prints_warning(self, capsys, tmp_path):
        with patch("lib.speakers.shutil.which", return_value=None):
            export_speakers_audio("audio.wav", [], str(tmp_path))
        captured = capsys.readouterr()
        assert "ffmpeg" in captured.err.lower()

    def test_no_speaker_segments_warns(self, capsys, tmp_path):
        segs = [seg(0, 1, "Hello")]  # no speaker field
        with patch("lib.speakers.shutil.which", return_value="/usr/bin/ffmpeg"):
            export_speakers_audio("audio.wav", segs, str(tmp_path))
        captured = capsys.readouterr()
        assert captured.err  # warning printed

    def test_ffmpeg_called_once_per_speaker(self, tmp_path):
        segs = [
            seg(0, 1, "A", speaker="Alice"),
            seg(2, 3, "B", speaker="Bob"),
        ]
        with patch("lib.speakers.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("lib.speakers.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            export_speakers_audio("audio.wav", segs, str(tmp_path), quiet=True)
        assert mock_run.call_count == 2

    def test_aselect_filter_in_ffmpeg_command(self, tmp_path):
        segs = [seg(0, 1, "A", speaker="Alice")]
        with patch("lib.speakers.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("lib.speakers.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            export_speakers_audio("audio.wav", segs, str(tmp_path), quiet=True)
        cmd = " ".join(mock_run.call_args[0][0])
        assert "aselect" in cmd

    def test_output_dir_created(self, tmp_path):
        out_dir = tmp_path / "speakers"
        segs = [seg(0, 1, "A", speaker="Alice")]
        with patch("lib.speakers.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("lib.speakers.subprocess.run", return_value=MagicMock(returncode=0)):
            export_speakers_audio("audio.wav", segs, str(out_dir), quiet=True)
        assert out_dir.exists()
```

**Step 2: Run tests**

```bash
pytest tests/unit/test_speakers.py -v
```

Expected: all PASS.

**Step 3: Commit**

```bash
git add tests/unit/test_speakers.py
git commit -m "test: add test_speakers.py (name mapping + speaker audio export)"
```

---

### Task 5: test_rss.py

**Files:**
- Create: `tests/unit/test_rss.py`

**Step 1: Write the test file**

```python
"""Unit tests for lib/rss.py — HTTP calls mocked."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from lib.rss import fetch_rss_episodes


VALID_RSS = b"""<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Test Podcast</title>
    <item>
      <title>Episode 1</title>
      <enclosure url="https://example.com/ep1.mp3" type="audio/mpeg"/>
    </item>
    <item>
      <title>Episode 2</title>
      <enclosure url="https://example.com/ep2.mp3" type="audio/mpeg"/>
    </item>
    <item>
      <title>Episode 3</title>
      <enclosure url="https://example.com/ep3.mp3" type="audio/mpeg"/>
    </item>
  </channel>
</rss>"""

NO_ENCLOSURE_RSS = b"""<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <item><title>Ep 1</title></item>
  </channel>
</rss>"""


def _mock_urlopen(xml_data):
    """Return a MagicMock that acts as a urllib context manager."""
    from unittest.mock import MagicMock
    mock_resp = MagicMock()
    mock_resp.read.return_value = xml_data
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_resp)
    mock_cm.__exit__ = MagicMock(return_value=False)
    return MagicMock(return_value=mock_cm)


# ── Happy path ─────────────────────────────────────────────────────────────────

class TestFetchRssEpisodes:
    def test_valid_feed_returns_tuples(self):
        with patch("urllib.request.urlopen", _mock_urlopen(VALID_RSS)):
            episodes = fetch_rss_episodes("https://x.com/feed.rss", latest=0, quiet=True)
        assert len(episodes) == 3
        assert all(isinstance(e, tuple) and len(e) == 2 for e in episodes)

    def test_latest_slices_correctly(self):
        with patch("urllib.request.urlopen", _mock_urlopen(VALID_RSS)):
            episodes = fetch_rss_episodes("https://x.com/feed.rss", latest=2, quiet=True)
        assert len(episodes) == 2

    def test_latest_zero_returns_all(self):
        with patch("urllib.request.urlopen", _mock_urlopen(VALID_RSS)):
            episodes = fetch_rss_episodes("https://x.com/feed.rss", latest=0, quiet=True)
        assert len(episodes) == 3

    def test_first_episode_url_and_title(self):
        with patch("urllib.request.urlopen", _mock_urlopen(VALID_RSS)):
            episodes = fetch_rss_episodes("https://x.com/feed.rss", latest=1, quiet=True)
        assert episodes[0][0] == "https://example.com/ep1.mp3"
        assert episodes[0][1] == "Episode 1"

    def test_latest_5_default(self):
        # 3 episodes, latest=5 → returns all 3
        with patch("urllib.request.urlopen", _mock_urlopen(VALID_RSS)):
            episodes = fetch_rss_episodes("https://x.com/feed.rss", quiet=True)
        assert len(episodes) == 3


# ── Error paths ────────────────────────────────────────────────────────────────

class TestFetchRssErrors:
    def test_no_enclosures_exits_3(self):
        with patch("urllib.request.urlopen", _mock_urlopen(NO_ENCLOSURE_RSS)):
            with pytest.raises(SystemExit) as exc:
                fetch_rss_episodes("https://x.com/feed.rss", quiet=True)
        assert exc.value.code == 3

    def test_network_error_exits_3(self):
        mock_urlopen = MagicMock(side_effect=Exception("Connection refused"))
        with patch("urllib.request.urlopen", mock_urlopen):
            with pytest.raises(SystemExit) as exc:
                fetch_rss_episodes("https://x.com/feed.rss", quiet=True)
        assert exc.value.code == 3

    def test_malformed_xml_exits_3(self):
        with patch("urllib.request.urlopen", _mock_urlopen(b"<not valid xml???")):
            with pytest.raises(SystemExit) as exc:
                fetch_rss_episodes("https://x.com/feed.rss", quiet=True)
        assert exc.value.code == 3

    def test_no_items_exits_3(self):
        empty_rss = b"""<?xml version="1.0"?>
        <rss version="2.0"><channel></channel></rss>"""
        with patch("urllib.request.urlopen", _mock_urlopen(empty_rss)):
            with pytest.raises(SystemExit) as exc:
                fetch_rss_episodes("https://x.com/feed.rss", quiet=True)
        assert exc.value.code == 3
```

> **Note on patching:** `rss.py` imports `urllib.request` lazily inside the function. Patching `urllib.request.urlopen` works because Python caches the module in `sys.modules`, so the patch modifies the same object the function accesses.

**Step 2: Run tests**

```bash
pytest tests/unit/test_rss.py -v
```

Expected: all PASS.

**Step 3: Commit**

```bash
git add tests/unit/test_rss.py
git commit -m "test: add test_rss.py (RSS parsing + all error exits)"
```

---

### Task 6: test_router.py

**Files:**
- Create: `tests/unit/test_router.py`

**Step 1: Write the test file**

> The router is a bash script (`scripts/transcribe`). Tests copy it to a temp directory alongside stub backends that print `BACKEND=<name>` to stdout and exit 0. Fake venv python files satisfy `backend_ready()` checks without actually installing anything.

```python
"""
Unit tests for scripts/transcribe bash router.

Strategy: copy the router to tmp_path, create stub backend scripts that
echo 'BACKEND=faster-whisper' or 'BACKEND=parakeet', and create fake
venv python files so backend_ready() passes.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

# ── Constants ──────────────────────────────────────────────────────────────────

ROUTER_SRC = Path(__file__).parent.parent.parent / "scripts" / "transcribe"
LIB_SRC = Path(__file__).parent.parent.parent / "scripts" / "backends" / "lib"
TONE_WAV = Path(__file__).parent.parent / "fixtures" / "tone.wav"

_STUB_FW = """\
#!/usr/bin/env bash
echo "BACKEND=faster-whisper"
echo "ARGS=$*"
exit 0
"""

_STUB_PK = """\
#!/usr/bin/env bash
echo "BACKEND=parakeet"
echo "ARGS=$*"
exit 0
"""


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_router_env(tmp_path: Path, fw_ready=True, pk_ready=True) -> Path:
    """Build a router environment with optional stub backend readiness."""
    # Copy router script
    router = tmp_path / "transcribe"
    shutil.copy(ROUTER_SRC, router)
    router.chmod(0o755)

    # Copy shared lib so --probe works
    shutil.copytree(LIB_SRC, tmp_path / "backends" / "lib")

    # faster-whisper backend
    fw_dir = tmp_path / "backends" / "faster-whisper"
    fw_dir.mkdir(parents=True)
    fw_stub = fw_dir / "transcribe"
    fw_stub.write_text(_STUB_FW)
    fw_stub.chmod(0o755)
    if fw_ready:
        venv_bin = fw_dir / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").write_text("#!/usr/bin/env python3\n")
        (venv_bin / "python").chmod(0o755)

    # parakeet backend
    pk_dir = tmp_path / "backends" / "parakeet"
    pk_dir.mkdir(parents=True)
    pk_stub = pk_dir / "transcribe"
    pk_stub.write_text(_STUB_PK)
    pk_stub.chmod(0o755)
    if pk_ready:
        venv_bin = pk_dir / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").write_text("#!/usr/bin/env python3\n")
        (venv_bin / "python").chmod(0o755)

    return tmp_path


@pytest.fixture
def router_env(tmp_path):
    """Router env with both backends installed and ready."""
    return _make_router_env(tmp_path)


@pytest.fixture
def router_env_no_backends(tmp_path):
    """Router env where neither backend has a venv (not ready)."""
    return _make_router_env(tmp_path, fw_ready=False, pk_ready=False)


def run(env: Path, *args, timeout=10) -> subprocess.CompletedProcess:
    """Run the router with the given args; always capture output."""
    return subprocess.run(
        ["bash", str(env / "transcribe"), *args],
        capture_output=True, text=True, timeout=timeout,
    )


# ── --version ─────────────────────────────────────────────────────────────────

class TestVersion:
    def test_prints_version(self, router_env):
        r = run(router_env, "--version")
        assert r.returncode == 0
        assert "super-transcribe" in r.stdout

    def test_version_format(self, router_env):
        r = run(router_env, "--version")
        # Should contain at least one digit (version number)
        assert any(c.isdigit() for c in r.stdout)


# ── --backends ────────────────────────────────────────────────────────────────

class TestBackends:
    def test_exits_zero(self, router_env):
        assert run(router_env, "--backends").returncode == 0

    def test_lists_faster_whisper(self, router_env):
        assert "faster-whisper" in run(router_env, "--backends").stdout

    def test_lists_parakeet(self, router_env):
        assert "parakeet" in run(router_env, "--backends").stdout


# ── --check ───────────────────────────────────────────────────────────────────

class TestCheckHealth:
    def test_check_json_exits_0_when_ready(self, router_env):
        r = run(router_env, "--check", "--json")
        assert r.returncode == 0

    def test_check_json_has_required_fields(self, router_env):
        r = run(router_env, "--check", "--json")
        data = json.loads(r.stdout)
        for field in ("ready", "action", "backends"):
            assert field in data, f"missing: {field}"

    def test_check_json_ready_true_when_backends_installed(self, router_env):
        r = run(router_env, "--check", "--json")
        assert json.loads(r.stdout)["ready"] is True

    def test_check_json_exits_2_when_no_backends(self, router_env_no_backends):
        r = run(router_env_no_backends, "--check", "--json")
        assert r.returncode == 2

    def test_check_json_ready_false_when_no_backends(self, router_env_no_backends):
        r = run(router_env_no_backends, "--check", "--json")
        data = json.loads(r.stdout)
        assert data["ready"] is False


# ── --probe ───────────────────────────────────────────────────────────────────

class TestProbe:
    def test_probe_tone_wav_returns_json(self, router_env):
        r = run(router_env, "--probe", str(TONE_WAV))
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "duration" in data and "file" in data

    def test_probe_duration_approximately_1_second(self, router_env):
        r = run(router_env, "--probe", str(TONE_WAV))
        data = json.loads(r.stdout)
        # Probe may use soundfile fallback; duration should be ~1.0
        assert 0.5 <= data["duration"] <= 2.0

    def test_probe_no_file_exits_nonzero(self, router_env):
        r = run(router_env, "--probe")
        assert r.returncode != 0


# ── Routing ───────────────────────────────────────────────────────────────────

class TestRouting:
    def test_translate_routes_to_faster_whisper(self, router_env, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.touch()
        r = run(router_env, "--translate", str(audio))
        assert "BACKEND=faster-whisper" in r.stdout

    def test_fast_routes_to_parakeet(self, router_env, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.touch()
        r = run(router_env, "--fast", str(audio))
        assert "BACKEND=parakeet" in r.stdout

    def test_long_form_routes_to_parakeet(self, router_env, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.touch()
        r = run(router_env, "--long-form", str(audio))
        assert "BACKEND=parakeet" in r.stdout

    def test_streaming_routes_to_parakeet(self, router_env, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.touch()
        r = run(router_env, "--streaming", str(audio))
        assert "BACKEND=parakeet" in r.stdout

    def test_non_eu_language_routes_to_fw(self, router_env, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.touch()
        r = run(router_env, "-l", "ja", str(audio))
        assert "BACKEND=faster-whisper" in r.stdout

    def test_eu_language_allows_parakeet(self, router_env, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.touch()
        r = run(router_env, "-l", "fr", str(audio))
        # fr is in PARAKEET_LANGS → Parakeet is the default
        assert "BACKEND=parakeet" in r.stdout

    def test_default_routes_to_parakeet(self, router_env, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.touch()
        r = run(router_env, str(audio))
        assert "BACKEND=parakeet" in r.stdout


# ── Conflicting / invalid flags ───────────────────────────────────────────────

class TestFlagErrors:
    def test_conflicting_fw_and_pk_flags_exits_3(self, router_env, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.touch()
        r = run(router_env, "--translate", "--long-form", str(audio))
        assert r.returncode == 3

    def test_backend_fw_alias(self, router_env, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.touch()
        r = run(router_env, "--backend", "fw", str(audio))
        assert "BACKEND=faster-whisper" in r.stdout

    def test_backend_pk_alias(self, router_env, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.touch()
        r = run(router_env, "--backend", "pk", str(audio))
        assert "BACKEND=parakeet" in r.stdout

    def test_backend_bogus_exits_3(self, router_env, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.touch()
        r = run(router_env, "--backend", "bogus", str(audio))
        assert r.returncode == 3

    def test_setup_without_value_exits_nonzero(self, router_env):
        r = run(router_env, "--setup")
        assert r.returncode != 0
```

**Step 2: Run tests**

```bash
pytest tests/unit/test_router.py -v
```

Expected: all PASS. The `--probe` test requires Python 3 in PATH (available). The routing tests use stub backends and shouldn't require real models.

**Step 3: Commit**

```bash
git add tests/unit/test_router.py
git commit -m "test: add test_router.py (routing, --check, --probe, --version, flag errors)"
```

---

### Task 7: Integration test infrastructure

**Files:**
- Create: `tests/integration/conftest.py`

**Step 1: Write `tests/integration/conftest.py`**

```python
"""
Integration test conftest: session-scoped skip fixtures for backend availability.
Tests in this directory are marked @pytest.mark.integration and require
the ML backends to be installed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).parent.parent.parent

FW_VENV_PYTHON = _REPO / "scripts/backends/faster-whisper/.venv/bin/python"
PK_VENV_PYTHON = _REPO / "scripts/backends/parakeet/venv/bin/python"
FW_TRANSCRIBE  = _REPO / "scripts/backends/faster-whisper/transcribe"
PK_TRANSCRIBE  = _REPO / "scripts/backends/parakeet/transcribe"
ROUTER         = _REPO / "scripts/transcribe"
TONE_WAV       = _REPO / "tests/fixtures/tone.wav"


@pytest.fixture(scope="session")
def require_fw():
    """Skip the entire file if faster-whisper is not installed."""
    if not FW_VENV_PYTHON.exists():
        pytest.skip(
            "faster-whisper backend not installed. "
            "Run: scripts/backends/faster-whisper/setup.sh"
        )


@pytest.fixture(scope="session")
def require_pk():
    """Skip the entire file if parakeet is not installed."""
    if not PK_VENV_PYTHON.exists():
        pytest.skip(
            "parakeet backend not installed. "
            "Run: scripts/backends/parakeet/setup.sh"
        )
```

**Step 2: Verify integration conftest is importable**

```bash
pytest tests/integration/ --collect-only -m integration
```

Expected: `no tests ran` (no test files yet). No import errors.

**Step 3: Commit**

```bash
git add tests/integration/conftest.py
git commit -m "test: add integration conftest with backend skip fixtures"
```

---

### Task 8: test_fw_e2e.py

**Files:**
- Create: `tests/integration/test_fw_e2e.py`

**Step 1: Write the test file**

```python
"""
End-to-end integration tests for the faster-whisper backend.

These tests run the actual backend script against a 1-second silent/tone WAV.
They skip automatically when the backend is not installed.

Run with: pytest -m integration tests/integration/test_fw_e2e.py -v
"""
from __future__ import annotations

import json
import re
import subprocess

import pytest

from tests.integration.conftest import (
    FW_TRANSCRIBE,
    ROUTER,
    TONE_WAV,
)

pytestmark = pytest.mark.integration


# ── Helper ────────────────────────────────────────────────────────────────────

def run_fw(*args, timeout=120):
    return subprocess.run(
        [str(FW_TRANSCRIBE), *args],
        capture_output=True, text=True, timeout=timeout,
    )


def run_router(*args, timeout=120):
    return subprocess.run(
        [str(ROUTER), "--backend", "faster-whisper", *args],
        capture_output=True, text=True, timeout=timeout,
    )


# ── Skip guard ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def _skip_if_no_fw(require_fw):
    """Module-level skip when backend absent."""


# ── Basic transcription ───────────────────────────────────────────────────────

class TestFwBasic:
    def test_exits_zero(self):
        r = run_fw(str(TONE_WAV))
        assert r.returncode == 0

    def test_stdout_or_silence_ok(self):
        r = run_fw(str(TONE_WAV))
        # Silence/tones may produce empty transcript — that's fine
        assert r.returncode == 0

    def test_nonexistent_file_exits_3(self):
        r = run_fw("/no/such/file.wav")
        assert r.returncode == 3

    def test_invalid_format_exits_nonzero(self):
        r = run_fw(str(TONE_WAV), "--format", "INVALID_FORMAT")
        assert r.returncode != 0


# ── Output formats ────────────────────────────────────────────────────────────

class TestFwFormats:
    def test_srt_format_structure(self):
        r = run_fw(str(TONE_WAV), "--format", "srt")
        assert r.returncode == 0
        if r.stdout.strip():  # non-empty transcript only
            lines = r.stdout.strip().split("\n")
            # First line should be cue number "1"
            assert lines[0] == "1"
            assert "-->" in r.stdout

    def test_json_format_parseable(self):
        r = run_fw(str(TONE_WAV), "--format", "json")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        for key in ("text", "language", "segments", "duration"):
            assert key in data, f"missing key: {key}"

    def test_json_duration_approximately_1_second(self):
        r = run_fw(str(TONE_WAV), "--format", "json")
        data = json.loads(r.stdout)
        assert 0.5 <= data["duration"] <= 2.0


# ── Agent mode ────────────────────────────────────────────────────────────────

class TestFwAgent:
    def test_agent_exits_zero(self):
        r = run_fw(str(TONE_WAV), "--agent")
        assert r.returncode == 0

    def test_agent_single_line_json(self):
        r = run_fw(str(TONE_WAV), "--agent")
        lines = [l for l in r.stdout.strip().split("\n") if l]
        assert len(lines) == 1

    def test_agent_has_required_fields(self):
        r = run_fw(str(TONE_WAV), "--agent")
        j = json.loads(r.stdout.strip())
        for field in ("text", "duration", "language", "backend", "segments", "word_count"):
            assert field in j, f"missing: {field}"

    def test_agent_backend_is_faster_whisper(self):
        r = run_fw(str(TONE_WAV), "--agent")
        j = json.loads(r.stdout.strip())
        assert j["backend"] == "faster-whisper"


# ── Probe ─────────────────────────────────────────────────────────────────────

class TestFwProbe:
    def test_probe_exits_zero(self):
        r = run_router("--probe", str(TONE_WAV))
        assert r.returncode == 0

    def test_probe_json_has_duration(self):
        r = run_router("--probe", str(TONE_WAV))
        data = json.loads(r.stdout)
        assert "duration" in data
        assert 0.5 <= data["duration"] <= 2.0


# ── FW-specific features ──────────────────────────────────────────────────────

class TestFwSpecific:
    def test_detect_language_only(self):
        r = run_fw(str(TONE_WAV), "--detect-language-only")
        assert r.returncode == 0
        # Output should mention "Language:"
        combined = r.stdout + r.stderr
        assert "Language:" in combined or "language" in combined.lower()

    def test_skip_existing(self, tmp_path):
        import shutil
        audio = tmp_path / "tone.wav"
        shutil.copy(TONE_WAV, audio)
        out = tmp_path / "tone.txt"

        # First run: creates output
        r1 = run_fw(str(audio), "--format", "text", "-o", str(out))
        assert r1.returncode == 0

        # Second run: --skip-existing should skip
        r2 = run_fw(str(audio), "--format", "text", "-o", str(out), "--skip-existing")
        assert r2.returncode == 0
        combined2 = r2.stdout + r2.stderr
        assert "skip" in combined2.lower() or "exist" in combined2.lower()
```

**Step 2: Run integration tests (skips if fw not installed)**

```bash
pytest tests/integration/test_fw_e2e.py -v -m integration
```

Expected:
- If faster-whisper **not** installed: `SKIP` with descriptive message.
- If faster-whisper **is** installed: all PASS (or investigate failures).

**Step 3: Commit**

```bash
git add tests/integration/test_fw_e2e.py
git commit -m "test: add test_fw_e2e.py (faster-whisper end-to-end, model-gated)"
```

---

### Task 9: test_pk_e2e.py

**Files:**
- Create: `tests/integration/test_pk_e2e.py`

**Step 1: Write the test file**

```python
"""
End-to-end integration tests for the Parakeet (NeMo) backend.

Skip automatically when the backend is not installed.

Run with: pytest -m integration tests/integration/test_pk_e2e.py -v
"""
from __future__ import annotations

import json
import subprocess

import pytest

from tests.integration.conftest import (
    PK_TRANSCRIBE,
    ROUTER,
    TONE_WAV,
)

pytestmark = pytest.mark.integration


# ── Helper ────────────────────────────────────────────────────────────────────

def run_pk(*args, timeout=180):
    return subprocess.run(
        [str(PK_TRANSCRIBE), *args],
        capture_output=True, text=True, timeout=timeout,
    )


def run_router(*args, timeout=180):
    return subprocess.run(
        [str(ROUTER), "--backend", "parakeet", *args],
        capture_output=True, text=True, timeout=timeout,
    )


# ── Skip guard ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def _skip_if_no_pk(require_pk):
    """Module-level skip when parakeet absent."""


# ── Basic transcription ───────────────────────────────────────────────────────

class TestPkBasic:
    def test_exits_zero(self):
        r = run_pk(str(TONE_WAV))
        assert r.returncode == 0

    def test_nonexistent_file_exits_3(self):
        r = run_pk("/no/such/file.wav")
        assert r.returncode == 3


# ── Output formats ────────────────────────────────────────────────────────────

class TestPkFormats:
    def test_srt_format_structure(self):
        r = run_pk(str(TONE_WAV), "--format", "srt")
        assert r.returncode == 0
        if r.stdout.strip():
            assert "-->" in r.stdout

    def test_json_format_parseable(self):
        r = run_pk(str(TONE_WAV), "--format", "json")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        for key in ("text", "language", "segments", "duration"):
            assert key in data, f"missing key: {key}"

    def test_json_duration_approximately_1_second(self):
        r = run_pk(str(TONE_WAV), "--format", "json")
        data = json.loads(r.stdout)
        assert 0.5 <= data["duration"] <= 2.0


# ── Agent mode ────────────────────────────────────────────────────────────────

class TestPkAgent:
    def test_agent_exits_zero(self):
        assert run_pk(str(TONE_WAV), "--agent").returncode == 0

    def test_agent_single_line_json(self):
        r = run_pk(str(TONE_WAV), "--agent")
        lines = [l for l in r.stdout.strip().split("\n") if l]
        assert len(lines) == 1

    def test_agent_has_required_fields(self):
        r = run_pk(str(TONE_WAV), "--agent")
        j = json.loads(r.stdout.strip())
        for field in ("text", "duration", "language", "backend", "segments", "word_count"):
            assert field in j, f"missing: {field}"

    def test_agent_backend_is_parakeet(self):
        r = run_pk(str(TONE_WAV), "--agent")
        j = json.loads(r.stdout.strip())
        assert j["backend"] == "parakeet"


# ── Probe ─────────────────────────────────────────────────────────────────────

class TestPkProbe:
    def test_probe_exits_zero(self):
        assert run_router("--probe", str(TONE_WAV)).returncode == 0

    def test_probe_json_has_duration(self):
        r = run_router("--probe", str(TONE_WAV))
        data = json.loads(r.stdout)
        assert 0.5 <= data["duration"] <= 2.0


# ── Parakeet-specific features ────────────────────────────────────────────────

class TestPkSpecific:
    def test_fast_flag_exits_zero(self):
        # --fast uses the 110M model (smaller, faster)
        r = run_pk(str(TONE_WAV), "--fast")
        assert r.returncode == 0

    def test_no_align_flag_exits_zero(self):
        # --no-align skips wav2vec2 alignment
        r = run_pk(str(TONE_WAV), "--no-align")
        assert r.returncode == 0
```

**Step 2: Run integration tests**

```bash
pytest tests/integration/test_pk_e2e.py -v -m integration
```

Expected:
- If parakeet **not** installed: `SKIP` with descriptive message.
- If parakeet **is** installed: all PASS.

**Step 3: Final verification — run all unit tests together**

```bash
pytest tests/unit/ -v
```

Expected: all PASS (no integration markers needed).

**Step 4: Commit**

```bash
git add tests/integration/test_pk_e2e.py
git commit -m "test: add test_pk_e2e.py (parakeet end-to-end, model-gated)"
```

---

## Final verification

After all tasks, run the full suite:

```bash
# Unit tests only (default — CI-safe):
pytest -v

# Check test count and markers:
pytest --collect-only | tail -5

# Integration tests (requires backends):
pytest -m integration -v
```

Expected unit test count: **~100 tests** across 6 files, all green.

Expected integration: skipped or passing depending on installed backends.
