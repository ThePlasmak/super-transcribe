"""
End-to-end integration tests for the faster-whisper backend.

These tests run the actual backend script against a 1-second silent/tone WAV.
They skip automatically when the backend is not installed.

Run with: pytest -m integration tests/integration/test_fw_e2e.py -v
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.integration.conftest import (
    FW_TRANSCRIBE,
    ROUTER,
    TONE_WAV,
)

pytestmark = pytest.mark.integration


# ── Helpers ───────────────────────────────────────────────────────────────────

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
        # Silence/tones may produce empty transcript — that's fine
        r = run_fw(str(TONE_WAV))
        assert r.returncode == 0

    def test_nonexistent_file_exits_3(self):
        # EXIT_BAD_INPUT = 3: file not found triggers resolve_inputs → empty list → sys.exit(3)
        r = run_fw("/no/such/file.wav")
        assert r.returncode == 3

    def test_invalid_format_exits_nonzero(self):
        # argparse p.error() exits with 2 for invalid --format values
        r = run_fw(str(TONE_WAV), "--format", "INVALID_FORMAT")
        assert r.returncode != 0


# ── Output formats ────────────────────────────────────────────────────────────

class TestFwFormats:
    def test_srt_format_structure(self):
        r = run_fw(str(TONE_WAV), "--format", "srt")
        assert r.returncode == 0
        if r.stdout.strip():  # non-empty transcript only
            lines = r.stdout.strip().split("\n")
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
        lines = [line for line in r.stdout.strip().split("\n") if line]
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
        # --detect-language-only: prints "Language: XX (probability: Y.YYY)" to stdout
        r = run_fw(str(TONE_WAV), "--detect-language-only")
        assert r.returncode == 0
        combined = r.stdout + r.stderr
        assert "Language:" in combined or "language" in combined.lower()

    def test_skip_existing(self, tmp_path):
        # AIDEV-NOTE: --skip-existing only fires when -o is a directory (the backend
        # checks out_dir.is_dir()). Pass a dir path so the skip logic activates.
        audio = tmp_path / "tone.wav"
        shutil.copy(TONE_WAV, audio)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        r1 = run_fw(str(audio), "--format", "text", "-o", str(out_dir))
        assert r1.returncode == 0

        r2 = run_fw(str(audio), "--format", "text", "-o", str(out_dir), "--skip-existing")
        assert r2.returncode == 0
        combined2 = r2.stdout + r2.stderr
        assert "skip" in combined2.lower() or "exist" in combined2.lower()
