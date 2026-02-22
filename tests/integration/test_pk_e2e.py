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


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _extract_json(text: str) -> dict:
    """Extract the first valid JSON object from stdout that may contain NeMo log noise.

    NeMo's logger can write [NeMo I ...] lines to stdout even when
    NEMO_LOG_LEVEL=ERROR is set (env var is applied after NeMo imports its
    logger).  NeMo also logs Python dicts (single-quoted) that look like
    JSON but are not.  We filter out NeMo log lines first, then locate
    the first '{...}' block that parses as valid JSON.

    Works for both compact single-line JSON (--agent) and indented multi-line
    JSON (--format json).
    """
    # AIDEV-NOTE: NeMo log lines always start with '[NeMo'; strip them so that
    # Python-repr dicts inside log messages don't confuse the JSON scanner.
    clean_lines = [line for line in text.splitlines() if not line.lstrip().startswith("[NeMo")]
    clean = "\n".join(clean_lines)

    # Scan for every '{' and attempt to parse the matching block
    search_from = 0
    while True:
        start = clean.find("{", search_from)
        if start == -1:
            raise ValueError(
                f"No valid JSON object found in output.\nstdout (first 800):\n{text[:800]}"
            )
        # Walk forward tracking brace depth to find the matching '}'
        depth = 0
        end = start
        in_string = False
        escape_next = False
        for idx in range(start, len(clean)):
            ch = clean[idx]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = idx
                        break
        candidate = clean[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Not valid JSON (e.g. a Python-repr dict); move past this '{'
            search_from = start + 1


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
        # EXIT_BAD_INPUT=3: resolve_inputs returns empty list → sys.exit(EXIT_BAD_INPUT)
        r = run_pk("/no/such/file.wav")
        assert r.returncode == 3


# ── Output formats ────────────────────────────────────────────────────────────

class TestPkFormats:
    def test_srt_format_structure(self):
        # AIDEV-NOTE: tone.wav is silence — parakeet produces an empty transcript;
        # the SRT output will be empty.  We only verify the process exits cleanly
        # and that any SRT cues present are structurally valid (contain '-->').
        r = run_pk(str(TONE_WAV), "--format", "srt")
        assert r.returncode == 0
        # If any SRT timestamp cue lines are present they must contain '-->'
        srt_cues = [line for line in r.stdout.splitlines() if "-->" in line]
        numeric_cues = [line for line in r.stdout.splitlines() if line.strip().isdigit()]
        if numeric_cues:
            assert srt_cues, "SRT sequence numbers found but no '-->' timestamp lines"

    def test_json_format_parseable(self):
        # AIDEV-NOTE: NeMo INFO-level logs may appear on stdout before the JSON;
        # _extract_json() locates the first '{...}' block regardless of log noise.
        # AIDEV-NOTE: 'language' is only present when speech is detected; tone.wav
        # is silent so 'language' may be absent.  Check only always-present keys.
        r = run_pk(str(TONE_WAV), "--format", "json")
        assert r.returncode == 0
        data = _extract_json(r.stdout)
        for key in ("text", "segments", "duration"):
            assert key in data, f"missing key: {key}"

    def test_json_duration_approximately_1_second(self):
        r = run_pk(str(TONE_WAV), "--format", "json")
        data = _extract_json(r.stdout)
        assert 0.5 <= data["duration"] <= 2.0


# ── Agent mode ────────────────────────────────────────────────────────────────

class TestPkAgent:
    def test_agent_exits_zero(self):
        assert run_pk(str(TONE_WAV), "--agent").returncode == 0

    def test_agent_single_line_json(self):
        # AIDEV-NOTE: NeMo log noise may appear; count only lines that start with '{'.
        r = run_pk(str(TONE_WAV), "--agent")
        json_lines = [
            line for line in r.stdout.strip().split("\n")
            if line.strip().startswith("{")
        ]
        assert len(json_lines) == 1

    def test_agent_has_required_fields(self):
        # AIDEV-NOTE: 'language' is safe to assert here even for silent audio because
        # format_agent_json() always emits the key via result.get("language") — it
        # serialises as JSON null when absent from the parakeet result dict, so the
        # key is always present (unlike --format json which omits the key entirely).
        r = run_pk(str(TONE_WAV), "--agent")
        j = _extract_json(r.stdout)
        for field in ("text", "duration", "language", "backend", "segments", "word_count"):
            assert field in j, f"missing: {field}"

    def test_agent_backend_is_parakeet(self):
        r = run_pk(str(TONE_WAV), "--agent")
        j = _extract_json(r.stdout)
        assert j["backend"] == "parakeet"


# ── Probe ─────────────────────────────────────────────────────────────────────

class TestPkProbe:
    def test_probe_exits_zero(self):
        assert run_router("--probe", str(TONE_WAV)).returncode == 0

    def test_probe_json_has_duration(self):
        r = run_router("--probe", str(TONE_WAV))
        data = json.loads(r.stdout)
        assert "duration" in data
        assert 0.5 <= data["duration"] <= 2.0


# ── Parakeet-specific features ────────────────────────────────────────────────

class TestPkSpecific:
    def test_fast_flag_exits_zero(self):
        # --fast selects the 110M model (nvidia/parakeet-tdt_ctc-110m); smaller and faster
        r = run_pk(str(TONE_WAV), "--fast")
        assert r.returncode == 0

    def test_no_align_flag_exits_zero(self):
        # --no-align skips wav2vec2 alignment refinement
        r = run_pk(str(TONE_WAV), "--no-align")
        assert r.returncode == 0
