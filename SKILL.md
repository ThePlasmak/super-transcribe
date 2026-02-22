---
name: super-transcribe
description: "Unified speech-to-text that auto-routes to the best backend (faster-whisper or parakeet) based on what the task needs. Lazy-loads backends — no need to set up both. Parakeet for best accuracy/speed/auto-punctuation, faster-whisper for translation/99+ languages/advanced inference tuning. Both backends support diarization, all output formats, search, chapters, preprocessing, and more. Use when: user asks to transcribe audio/video, generate subtitles, identify speakers, translate speech, search transcripts, or any speech-to-text task. Don't use when: user wants text-to-speech (use tts), audio editing, or music generation."
version: 0.1.0
author: Sarah Mak
tags:
  [
    "audio",
    "transcription",
    "whisper",
    "parakeet",
    "speech-to-text",
    "ml",
    "cuda",
    "gpu",
    "subtitles",
    "diarization",
    "podcast",
    "unified",
  ]
platforms: ["linux", "wsl2"]
metadata:
  {
    "openclaw":
      {
        "emoji": "🎙️",
        "requires":
          {
            "bins": ["python3"],
            "optionalBins": ["ffmpeg", "yt-dlp"],
            "optionalPaths": ["~/.cache/huggingface/token"],
          },
      },
  }
---

# 🎙️ Super-Transcribe — Unified Speech-to-Text

A self-contained transcription skill with two bundled backends that intelligently routes based on task requirements:

- **🦜 Parakeet** (NVIDIA NeMo) — best accuracy (6.34% WER), ~3380× realtime, auto-punctuation, 25 European languages, NeMo diarization
- **🗣️ faster-whisper** (CTranslate2) — translation, 99+ languages, initial prompting, advanced inference tuning

**Shared features** (work on both backends): diarization, all 10 output formats, search, chapters, filler removal, preprocessing, burn-in, RSS, speaker export.

**Lazy loading:** each backend sets up its own venv on first use. No pre-configuration needed — just transcribe and the right backend installs itself.

## When to Use

Use this skill for **any speech-to-text task**. It replaces both the `faster-whisper` and `parakeet` skills as your single entry point.

**Trigger phrases:**
"transcribe this", "speech to text", "what did they say", "make a transcript",
"subtitle this", "who's speaking", "translate this audio", "transcribe this podcast",
"transcribe with parakeet", "transcribe with whisper", "best accuracy transcription",
"transcribe in French", "diarize this meeting", "find where X is mentioned"

## Auto-Routing Logic

The router picks the backend automatically:

```
┌─────────────────────────────────────────────┐
│             --backend specified?             │
│         YES → use that backend              │
│         NO  ↓                               │
├─────────────────────────────────────────────┤
│     Needs faster-whisper-only feature?      │
│  (translate, initial-prompt, hotwords,      │
│   multilingual, non-EU language, advanced   │
│   inference tuning params)                  │
│         YES → faster-whisper                │
│         NO  ↓                               │
├─────────────────────────────────────────────┤
│      Needs parakeet-only feature?           │
│  (--long-form, --streaming)                 │
│         YES → parakeet                      │
│         NO  ↓                               │
├─────────────────────────────────────────────┤
│  Default: prefer Parakeet (best accuracy    │
│  + speed + auto-punctuation)                │
│  Fall back to faster-whisper if Parakeet    │
│  not installed                              │
└─────────────────────────────────────────────┘
```

## Quick Reference

### Basic (auto-routes to best available)

| Task | Command |
|---|---|
| **Basic transcription** | `./scripts/transcribe audio.mp3` |
| **SRT subtitles** | `./scripts/transcribe audio.mp3 --format srt -o subs.srt` |
| **VTT subtitles** | `./scripts/transcribe audio.mp3 --format vtt -o subs.vtt` |
| **ASS subtitles** | `./scripts/transcribe audio.mp3 --format ass -o subs.ass` |
| **LRC lyrics** | `./scripts/transcribe audio.mp3 --format lrc -o lyrics.lrc` |
| **TTML broadcast** | `./scripts/transcribe audio.mp3 --format ttml -o subs.ttml` |
| **CSV spreadsheet** | `./scripts/transcribe audio.mp3 --format csv -o out.csv` |
| **JSON output** | `./scripts/transcribe audio.mp3 --format json -o out.json` |
| **YouTube/URL** | `./scripts/transcribe https://youtube.com/watch?v=...` |
| **Batch process** | `./scripts/transcribe *.mp3 -o ./transcripts/` |
| **Force backend** | `./scripts/transcribe --backend parakeet audio.mp3` |
| **List backends** | `./scripts/transcribe --backends` |
| **Speaker diarization** | `./scripts/transcribe meeting.wav --diarize` |
| **Search transcript** | `./scripts/transcribe audio.mp3 --search "keyword"` |
| **Detect chapters** | `./scripts/transcribe audio.mp3 --detect-chapters` |
| **Clean filler words** | `./scripts/transcribe audio.mp3 --clean-filler` |
| **Denoise audio** | `./scripts/transcribe audio.mp3 --denoise` |
| **Podcast RSS feed** | `./scripts/transcribe --rss https://feed.url` |
| **Burn subtitles** | `./scripts/transcribe video.mp4 --burn-in out.mp4` |
| **Name speakers** | `./scripts/transcribe audio.mp3 --diarize --speaker-names "Alice,Bob"` |
| **Export speaker audio** | `./scripts/transcribe audio.mp3 --diarize --export-speakers ./spk/` |

### Routes to faster-whisper automatically

| Task | Command | Why |
|---|---|---|
| **Translate → English** | `./scripts/transcribe audio.mp3 --translate` | Whisper-specific feature |
| **Domain jargon** | `./scripts/transcribe audio.mp3 --initial-prompt "Kubernetes"` | Whisper-specific prompting |
| **Non-European language** | `./scripts/transcribe audio.mp3 -l ja` | Parakeet: 25 EU langs only |
| **Multilingual mode** | `./scripts/transcribe audio.mp3 --multilingual` | Whisper-specific feature |

### Routes to Parakeet automatically

| Task | Command | Why |
|---|---|---|
| **Long audio (>24 min)** | `./scripts/transcribe lecture.wav --long-form` | Parakeet-only: local attention mode |
| **Streaming output** | `./scripts/transcribe audio.wav --streaming` | Parakeet-only: chunked inference |

## ⚠️ Agent Guidance — Keep Invocations Minimal

**CORE RULE:** The default command (`./scripts/transcribe audio.mp3`) is the fastest path. Add flags only when the user explicitly asks for that capability. The router handles backend selection automatically.

**⚠️ Model preference:** When the router selects faster-whisper, it uses `distil-large-v3.5` by default — this is the preferred model (fastest, better accuracy than large-v3-turbo). Don't override unless the user asks.

**Do NOT:**
- Add `--backend` unless the user specifically requests a backend
- Add `--diarize` unless the user asks "who said what" / "identify speakers"
- Add `--translate` unless the user wants audio translated to English
- Add `--format srt/vtt/ass/etc.` unless the user asks for subtitles in that format
- Add `--long-form` unless the audio is confirmed >24 minutes
- Add `--denoise`/`--normalize` unless the user mentions bad audio quality
- Add `--initial-prompt` unless there's domain-specific jargon to prime
- Add `--search` unless the user asks to find/locate a word in audio
- Add `--detect-chapters` unless the user asks for chapters/sections
- Add `--clean-filler` unless the user asks to remove filler words

**Don't use this skill when:**
- User wants **text-to-speech** (use the `tts` tool instead)
- User wants **audio editing** or music generation
- User wants to **summarize** a YouTube video without needing a raw transcript (use `summarize` skill)

**Output handling:**
- **Text transcript** → show directly to user (summarise long ones)
- **Subtitle files** (SRT, VTT, ASS, LRC, TTML) → write to `-o` file, tell user the path
- **Data formats** (CSV, TSV, JSON, HTML) → write to `-o` file, tell user the path
- **Search results** → show directly (human-readable)
- **Chapter output** → show directly or write to `--chapters-file`

## Backend Comparison

| Feature | 🦜 Parakeet | 🗣️ faster-whisper |
|---|---|---|
| **Accuracy** | ✅ Best (6.34% avg WER) | Good (distil: 7.08% WER) |
| **Speed** | ✅ ~3380× realtime | ~20× realtime |
| **Auto punctuation** | ✅ Built-in | ❌ Requires post-processing |
| **Languages** | 25 European | ✅ 99+ worldwide |
| **Diarization** | ✅ NeMo ClusteringDiarizer | ✅ pyannote speaker ID |
| **Translation** | ❌ | ✅ Any → English |
| **Chapters/search** | ✅ Shared | ✅ Shared |
| **Output formats** | ✅ All 10 formats | ✅ All 10 formats |
| **Audio preprocessing** | ✅ --denoise, --normalize | ✅ --denoise, --normalize |
| **Filler removal** | ✅ --clean-filler | ✅ --clean-filler |
| **Long audio** | ✅ Up to 3 hours | Limited by VRAM |
| **Streaming** | ✅ Chunked inference | ✅ Segment streaming |
| **RSS/podcast** | ✅ --rss | ✅ --rss |
| **VRAM usage** | ~2GB | ~1.5GB (distil) |
| **Burn-in subtitles** | ✅ --burn-in | ✅ --burn-in |
| **Initial prompt** | ❌ | ✅ Domain jargon priming |
| **Translation** | ❌ | ✅ Any → English |

## Setup

**No manual setup required.** Both backends are bundled and lazy-load on first use — the first transcription with a backend triggers its `setup.sh` automatically (creates a venv, installs dependencies, detects GPU).

To check backend status or pre-install:
```bash
./scripts/transcribe --backends                  # Show what's installed
./scripts/transcribe --setup faster-whisper      # Pre-install faster-whisper
./scripts/transcribe --setup parakeet            # Pre-install parakeet
./scripts/transcribe --setup all                 # Pre-install both
```

### Requirements

- Python 3.10+
- NVIDIA GPU + CUDA (strongly recommended — CPU is 50-100× slower)
- Optional: ffmpeg (needed for mp3/m4a/mp4 input with parakeet; also for --denoise, --normalize, --burn-in)
- Optional: yt-dlp (for YouTube/URL input)
- Optional: HuggingFace token at `~/.cache/huggingface/token` (for pyannote diarization fallback)

## Options

### Super-transcribe routing options

```
--backend <name>     Force a backend: faster-whisper | parakeet | fw | pk
--backends           List available/installed backends and exit
--help-routing       Show detailed auto-routing logic
```

### Shared options (work with both backends)

```
AUDIO                 Audio file(s), directory, glob, or URL
-f, --format FMT      text | json | srt | vtt | ass | lrc | ttml | csv | tsv | html
-o, --output PATH     Output file or directory
-m, --model NAME      Model name (backend-specific)
-l, --language CODE   Language code
--max-words-per-line  Subtitle word wrapping
--max-chars-per-line  Subtitle character wrapping
--batch-size N        Inference batch size
--skip-existing       Skip already-transcribed files
--device DEV          auto | cpu | cuda
-q, --quiet           Suppress progress messages
--version             Print version info
--diarize             Speaker diarization
--min-speakers N      Min speakers hint for diarization
--max-speakers N      Max speakers hint for diarization
--speaker-names LIST  Replace SPEAKER_1, SPEAKER_2 with names
--export-speakers DIR Export each speaker's audio to WAV files
--search TERM         Search transcript for TERM
--search-fuzzy        Fuzzy/approximate search matching
--detect-chapters     Detect chapter breaks from silence gaps
--chapter-gap SEC     Min silence gap for chapter break (default: 8.0)
--chapters-file PATH  Write chapter markers to file
--chapter-format FMT  youtube | text | json (default: youtube)
--clean-filler        Remove hesitation fillers (um, uh, etc.)
--filter-hallucinations  Filter common hallucination patterns
--merge-sentences     Merge segments into sentence chunks
--detect-paragraphs   Insert paragraph breaks based on gaps
--paragraph-gap SEC   Min gap for paragraph break (default: 3.0)
--normalize           EBU R128 volume normalization
--denoise             High-pass + FFT noise reduction
--channel CH          left | right | mix (default: mix)
--burn-in OUTPUT      Burn subtitles into video file
--rss URL             Podcast RSS feed to transcribe
--rss-latest N        Latest N episodes from RSS (default: 5)
--stats-file PATH     Write performance stats JSON sidecar
```

### faster-whisper only options

```
--translate           Translate to English
--multilingual        Enable multilingual/code-switching mode
--initial-prompt TEXT Prompt to condition the model
--hotwords WORDS      Hotwords to boost recognition
--prefix TEXT         Prefix to condition first segment
--word-timestamps     Word-level timestamps
--stream              Output segments as transcribed (streaming)
--language-map MAP    Per-file language overrides for batch mode
--parallel N          Parallel workers for batch processing
--hf-token TOKEN      HuggingFace token override
--revision REV        Model revision to pin
--model-dir PATH      Custom model cache directory
--compute-type TYPE   Quantization: auto | int8 | float16 | etc.
--threads N           CPU threads for CTranslate2
--output-template TPL Output filename template
--beam-size N         Beam search size
--temperature T       Sampling temperature
--best-of N           Candidates when sampling
--patience F          Beam search patience
--repetition-penalty  Penalty for repeated tokens
--no-repeat-ngram-size  Prevent n-gram repetition
--no-vad              Disable VAD
--vad-threshold T     VAD speech probability threshold
--min-confidence PROB Drop segments below confidence
--clip-timestamps     Transcribe specific time ranges
--hallucination-silence-threshold SEC  Skip hallucinated silence
--no-batch            Disable batched inference
--no-timestamps       No timing information
--retries N           Retry failed files
--keep-temp           Keep downloaded temp files
--log-level LEVEL     debug | info | warning | error
--stats-file PATH     Write stats JSON sidecar
--detect-language-only  Detect language and exit
--progress            Show progress bar
```

### Parakeet only options

```
--long-form           Local attention for audio >24 min (up to ~3 hours)
--streaming           Print segments as they're transcribed (chunked inference)
--timestamps          Enable word/segment/char timestamps (auto-enabled for timed formats)
```

## Architecture

Both backends share a common library to avoid code duplication:

```
scripts/
├── transcribe           # Unified entry point (router)
└── backends/
    ├── lib/             # Shared code (imported by both backends)
    │   ├── formatters.py    # All 10 output formats
    │   ├── postprocess.py   # Search, chapters, filler, paragraphs, merge, hallucination filter
    │   ├── audio.py         # Preprocessing, conversion, download, burn-in
    │   ├── speakers.py      # Speaker name mapping, audio export
    │   └── rss.py           # RSS feed parsing
    ├── faster-whisper/  # CTranslate2 backend (imports from lib/)
    └── parakeet/        # NeMo backend (imports from lib/)
```
