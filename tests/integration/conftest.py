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
