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
        # word has no "speaker" key — it should not gain one
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
