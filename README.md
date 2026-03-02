# 🎙️ Super-Transcribe

A speech-to-text skill with two bundled engines for the best speed and featureset — [NVIDIA Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (NeMo) for the former, and [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) for the latter.

The setup script installs the fastest backend for your setup, and the main script auto-selects the best backend for your task (and loads the other backend if necessary).

## First-Time Setup

If you set this up with your agent, it will use the setup script, which detects your platform, GPU, Python version, and optional dependencies. Next, it will install and configure the optimal backend(s).

See **Prerequisites** below for more information about backend compatibility.

### Prerequisites

| Dependency | Required? | Install |
|---|---|---|
| Python 3.10+ | **Required** | `sudo apt install python3 python3-venv` / `brew install python@3.12` |
| NVIDIA GPU + CUDA | Highly recommended (transcriptions are 50 to 100× faster with a GPU) | [WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/) / install nvidia-driver |
| ffmpeg | Recommended (needed for input formats other than WAV and features like audio preprocessing and burn-in) | `sudo apt install ffmpeg` / `brew install ffmpeg` |
| yt-dlp | Optional (for downloading media) | `pipx install yt-dlp` |
| HuggingFace token | Optional (for faster-whisper's speaker diarization) | `huggingface-cli login` + [accept model](https://hf.co/pyannote/speaker-diarization-3.1) |

## Why Two Backends?

In general, Parakeet has better speed and accuracy, while faster-whisper has more features.

| | 🦜 Parakeet (default) | 🗣️ faster-whisper |
|---|---|---|
| **Accuracy** | Best (6.34% WER) | Good (7.08% WER) |
| **Speed** | ~3380× realtime | ~20× realtime |
| **Auto-punctuation** | Built-in | No |
| **Languages** | 25 European | 99+ worldwide |
| **Translation** | Canary (between the 25 European languages) | Any → English |
| **Best for** | Standard transcription | Translation, non-European languages, prompting |

The router automatically picks the most suitable backend for your task. The agent can override this with `--backend parakeet` or `--backend faster-whisper`.

## Features (Selected)

**Shared (both backends):**
- 10 output formats
- Speaker diarization
- Chapter detection
- Filler removal
- Denoise/normalize audio
- Check RSS/podcast feeds
- Batch processing

**Parakeet-only (selected features):**
- `--fast` (110M model)
- `--multitalker` (overlapping speech)
- Canary translation (allowing translation between European languages, not just translating to English like in faster-whisper)

**faster-whisper-only (selected features):**
- `--translate` (to English only)
- `--initial-prompt` (makes decoder focus on transcribing certain words correctly)
- `--hotwords` (like `--initial-prompt` but more focused on getting the exact spellings right)
- `--multilingual` (supports multiple languages in the same file)
- 99+ languages

### Agent Integration

This skill is primarily designed for [OpenClaw](https://openclaw.ai) agents.

Key features to aid agent use:

- **`--probe`** — Check audio duration/format before committing to transcription

    ```bash
    # Probe before deciding to transcribe
    ./scripts/transcribe --probe recording.mp3
    # → {"file":"recording.mp3","duration":2714.5,"duration_human":"45m 14s",...}
    ```

- **`--agent`** — Compact JSON output with text, duration, language, confidence, speaker info, and summary hints

    ```bash
    # Agent mode with file output
    ./scripts/transcribe --agent -o /tmp/transcript.txt audio.ogg
    # → {"text":"...","duration":4.2,"avg_confidence":0.94,"summary_hint":{"first":"...","last":"..."},...}
    ```

- **Exit codes** — 0 (success), 1 (error), 2 (missing dep), 3 (bad input), 4 (GPU OOM)
- **First-run messaging** — Notifies the agent when a backend is setting up for the first time

## Output Formats

`text` (default) · `json` · `srt` · `vtt` · `ass` · `lrc` · `ttml` · `csv` · `tsv` · `html`

## Requirements

- Python 3.10+
- NVIDIA GPU + CUDA (highly recommended — CPU is 50-100× slower)
- Optional: ffmpeg (for non-WAV input, preprocessing, burn-in)
- Optional: yt-dlp (for YouTube/URL input)

## Platform Support

| Platform | GPU Acceleration | Speed |
|---|---|---|
| Linux + NVIDIA GPU | CUDA | Full speed |
| WSL2 + NVIDIA GPU | CUDA | Full speed |
| macOS Apple Silicon | CPU only | ~3-5× RT (faster-whisper only) |
| Linux (no GPU) | CPU | ~1× RT |

## Documentation

- **[SKILL.md](SKILL.md)** — Full reference with all options, model tables, and agent guidance
- **[CHANGELOG.md](CHANGELOG.md)** — Version history

## Licenses

- Parakeet TDT v3: CC-BY-4.0
- faster-whisper: MIT
