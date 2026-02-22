"""Unit tests for lib/audio.py — all subprocess calls mocked."""
from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

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
        # audio.py prints "Warning: skipping non-audio file: ..."
        assert "skipping" in captured.err.lower() or "warning" in captured.err.lower()

    def test_missing_file_warning(self, capsys):
        resolve_inputs(["/no/such/file.mp3"])
        captured = capsys.readouterr()
        assert captured.err  # some warning emitted


# ── probe_audio ───────────────────────────────────────────────────────────────

def _ffprobe_json(duration="2.5", channels=1, sample_rate="16000"):
    """Build a minimal ffprobe JSON string for mocking subprocess.run.stdout."""
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

        # AIDEV-NOTE: also mock soundfile so the fallback import doesn't pull in real data
        with patch("lib.audio.shutil.which", return_value=None):
            with patch.dict("sys.modules", {"soundfile": None}):
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
             patch("lib.audio.subprocess.run") as mock_run:
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
        # pan=mono|c0=c0 is the left-channel filter
        assert "c0=c0" in cmd

    def test_right_uses_c1_filter(self, tmp_path):
        f = tmp_path / "audio.wav"
        f.touch()

        with patch("lib.audio.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("lib.audio.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            extract_channel(str(f), "right", quiet=True)

        cmd = " ".join(mock_run.call_args[0][0])
        # pan=mono|c0=c1 is the right-channel filter
        assert "c0=c1" in cmd

    def test_no_ffmpeg_returns_original(self, tmp_path):
        f = tmp_path / "audio.wav"
        f.touch()

        with patch("lib.audio.shutil.which", return_value=None):
            path, tmp = extract_channel(str(f), "left", quiet=True)

        assert path == str(f) and tmp is None

    def test_ffmpeg_failure_returns_original(self, tmp_path):
        f = tmp_path / "audio.wav"
        f.touch()

        with patch("lib.audio.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("lib.audio.subprocess.run") as mock_run, \
             patch("lib.audio.os.path.exists", return_value=False):
            mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg")
            path, tmp = extract_channel(str(f), "left", quiet=True)

        assert path == str(f) and tmp is None
