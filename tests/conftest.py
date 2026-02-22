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
