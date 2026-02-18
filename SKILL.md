---
name: super-transcribe
description: "Unified speech-to-text that auto-routes to the best backend (faster-whisper or parakeet) based on what the task needs. Lazy-loads backends — no need to set up both. Parakeet for best accuracy/speed/auto-punctuation, faster-whisper for diarization/translation/99+ languages/advanced features. Use when: user asks to transcribe audio/video, generate subtitles, identify speakers, translate speech, search transcripts, or any speech-to-text task. Don't use when: user wants text-to-speech (use tts), audio editing, or music generation."
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

- **🦜 Parakeet** (NVIDIA NeMo) — best accuracy (6.34% WER), ~3380× realtime, auto-punctuation, 25 European languages
- **🗣️ faster-whisper** (CTranslate2) — diarization, translation, 99+ languages, chapters, search, 12+ output formats, audio preprocessing

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
│  (diarize, translate, search, chapters,     │
│   RSS, ASS/LRC/TTML/CSV formats, denoise,  │
│   filler removal, burn-in, language-map,    │
│   initial-prompt, multilingual, etc.)       │
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
| **JSON output** | `./scripts/transcribe audio.mp3 --format json -o out.json` |
| **YouTube/URL** | `./scripts/transcribe https://youtube.com/watch?v=...` |
| **Batch process** | `./scripts/transcribe *.mp3 -o ./transcripts/` |
| **Force backend** | `./scripts/transcribe --backend parakeet audio.mp3` |
| **List backends** | `./scripts/transcribe --backends` |

### Routes to faster-whisper automatically

| Task | Command | Why |
|---|---|---|
| **Speaker diarization** | `./scripts/transcribe meeting.wav --diarize` | Parakeet has no diarization |
| **Translate → English** | `./scripts/transcribe audio.mp3 --translate` | Parakeet has no translation |
| **ASS subtitles** | `./scripts/transcribe audio.mp3 --format ass -o s.ass` | Parakeet: text/json/srt/vtt only |
| **LRC lyrics** | `./scripts/transcribe audio.mp3 --format lrc -o lyrics.lrc` | Parakeet: text/json/srt/vtt only |
| **TTML broadcast** | `./scripts/transcribe audio.mp3 --format ttml -o s.ttml` | Parakeet: text/json/srt/vtt only |
| **CSV spreadsheet** | `./scripts/transcribe audio.mp3 --format csv -o out.csv` | Parakeet: text/json/srt/vtt only |
| **Search transcript** | `./scripts/transcribe audio.mp3 --search "keyword"` | Parakeet has no search |
| **Detect chapters** | `./scripts/transcribe audio.mp3 --detect-chapters` | Parakeet has no chapters |
| **Podcast RSS feed** | `./scripts/transcribe --rss https://feed.url` | Parakeet has no RSS |
| **Denoise audio** | `./scripts/transcribe audio.mp3 --denoise` | Parakeet has no preprocessing |
| **Clean filler words** | `./scripts/transcribe audio.mp3 --clean-filler` | Parakeet has no filler removal |
| **Domain jargon** | `./scripts/transcribe audio.mp3 --initial-prompt "Kubernetes"` | Parakeet has no prompting |
| **Burn subtitles** | `./scripts/transcribe video.mp4 --burn-in out.mp4` | Parakeet has no burn-in |
| **Name speakers** | `./scripts/transcribe audio.mp3 --diarize --speaker-names "Alice,Bob"` | Requires diarization |
| **Export speaker audio** | `./scripts/transcribe audio.mp3 --diarize --export-speakers ./spk/` | Requires diarization |
| **Non-European language** | `./scripts/transcribe audio.mp3 -l ja` | Parakeet: 25 EU langs only |

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
- **Data formats** (CSV, JSON, HTML) → write to `-o` file, tell user the path
- **Search results** → show directly (human-readable)
- **Chapter output** → show directly or write to `--chapters-file`

## Backend Comparison

| Feature | 🦜 Parakeet | 🗣️ faster-whisper |
|---|---|---|
| **Accuracy** | ✅ Best (6.34% avg WER) | Good (distil: 7.08% WER) |
| **Speed** | ✅ ~3380× realtime | ~20× realtime |
| **Auto punctuation** | ✅ Built-in | ❌ Requires post-processing |
| **Languages** | 25 European | ✅ 99+ worldwide |
| **Diarization** | ❌ | ✅ pyannote speaker ID |
| **Translation** | ❌ | ✅ Any → English |
| **Chapters/search** | ❌ | ✅ Built-in |
| **Output formats** | text, JSON, SRT, VTT | ✅ text, JSON, SRT, VTT, ASS, LRC, TTML, CSV, TSV, HTML |
| **Audio preprocessing** | ❌ | ✅ --denoise, --normalize |
| **Filler removal** | ❌ | ✅ --clean-filler |
| **Long audio** | ✅ Up to 3 hours | Limited by VRAM |
| **Streaming** | ✅ Chunked inference | ✅ Segment streaming |
| **RSS/podcast** | ❌ | ✅ --rss |
| **VRAM usage** | ~2GB | ~1.5GB (distil) |
| **Burn-in subtitles** | ❌ | ✅ --burn-in |

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
- Optional: ffmpeg (needed for mp3/m4a/mp4 input with parakeet; also for --denoise, --normalize, --burn-in with faster-whisper)
- Optional: yt-dlp (for YouTube/URL input)
- Optional: HuggingFace token at `~/.cache/huggingface/token` (for speaker diarization with pyannote)

## Options

### Super-transcribe routing options

```
--backend <name>     Force a backend: faster-whisper | parakeet | fw | pk
--backends           List available/installed backends and exit
--help-routing       Show detailed auto-routing logic
```

### All other options

All flags are passed through to the selected backend. The full option set depends on which backend is selected:

**Shared options** (work with both backends):
```
AUDIO                 Audio file(s), directory, glob, or URL
-f, --format FMT      text | json | srt | vtt (+ more with faster-whisper)
-o, --output PATH     Output file or directory
-m, --model NAME      Model name (backend-specific)
-l, --language CODE   Language code
--word-timestamps     Word-level timestamps (fw) / --timestamps (pk, auto-enabled)
--max-words-per-line  Subtitle word wrapping
--max-chars-per-line  Subtitle character wrapping
--batch-size N        Inference batch size
--skip-existing       Skip already-transcribed files
--device DEV          auto | cpu | cuda
-q, --quiet           Suppress progress messages
--version             Print version info
```

**faster-whisper only** — see the faster-whisper SKILL.md for full docs:
```
--diarize, --min-speakers, --max-speakers, --speaker-names, --export-speakers
--translate, --multilingual
--search, --search-fuzzy
--detect-chapters, --chapter-gap, --chapters-file, --chapter-format
--rss, --rss-latest
--format ass|lrc|ttml|csv|tsv|html
--clean-filler, --normalize, --denoise
--burn-in, --filter-hallucinations
--initial-prompt, --hotwords, --prefix
--language-map, --parallel, --channel
--merge-sentences, --detect-paragraphs, --paragraph-gap
--stream, --clip-timestamps, --temperature, --beam-size
--hallucination-silence-threshold, --min-confidence
--vad-threshold, --no-vad, --no-batch
--stats-file, --retries, --keep-temp
--hf-token, --revision, --model-dir, --compute-type, --threads
--log-level
```

**Parakeet only** — see the parakeet SKILL.md for full docs:
```
--long-form           Local attention for audio >24 min (up to ~3 hours)
--streaming           Print segments as they're transcribed (chunked inference)
```
