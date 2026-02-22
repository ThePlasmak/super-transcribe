"""
Shared audio processing functions for super-transcribe backends.
Includes: preprocessing (denoise/normalize), channel extraction,
subtitle burn-in, URL download, audio conversion.
"""

import os
import subprocess
import shutil
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------

def get_audio_duration(audio_path):
    """Get audio duration in seconds using ffprobe or soundfile."""
    if shutil.which("ffprobe"):
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
                capture_output=True, text=True, check=True,
            )
            return float(result.stdout.strip())
        except Exception:
            pass

    try:
        import soundfile as sf
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        pass

    return 0.0


# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------

def preprocess_audio(audio_path, normalize=False, denoise=False, quiet=False):
    """Preprocess audio with ffmpeg filters (normalize volume, reduce noise).
    Returns (processed_path, tmp_path_to_cleanup_or_None).
    """
    if not normalize and not denoise:
        return audio_path, None

    if not shutil.which("ffmpeg"):
        if not quiet:
            import sys
            print("⚠️  ffmpeg not found — skipping preprocessing", file=sys.stderr)
        return audio_path, None

    filters = []
    if denoise:
        filters.append("highpass=f=200")
        filters.append("afftdn=nf=-25")
    if normalize:
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")

    tmp_path = audio_path + ".preprocessed.wav"
    filter_str = ",".join(filters)
    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-af", filter_str,
        "-ar", "16000", "-ac", "1",
        tmp_path,
    ]

    if not quiet:
        import sys
        labels = []
        if normalize:
            labels.append("normalizing")
        if denoise:
            labels.append("denoising")
        print(f"🔧 Preprocessing: {' + '.join(labels)}...", file=sys.stderr)

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return tmp_path, tmp_path
    except subprocess.CalledProcessError:
        if not quiet:
            import sys
            print("⚠️  Preprocessing failed, using original audio", file=sys.stderr)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return audio_path, None


# ---------------------------------------------------------------------------
# Channel extraction
# ---------------------------------------------------------------------------

def extract_channel(audio_path, channel, quiet=False):
    """Extract a stereo channel from audio using ffmpeg.
    channel: 'left' (c0), 'right' (c1), or 'mix' (no-op).
    Returns (output_path, tmp_path_to_cleanup_or_None).
    """
    if channel == "mix":
        return audio_path, None

    if not shutil.which("ffmpeg"):
        if not quiet:
            import sys
            print("⚠️  ffmpeg not found — cannot extract channel; using full mix", file=sys.stderr)
        return audio_path, None

    pan = "c0" if channel == "left" else "c1"
    tmp_path = audio_path + f".{channel}.wav"
    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-af", f"pan=mono|c0={pan}",
        "-ar", "16000",
        tmp_path,
    ]
    if not quiet:
        import sys
        print(f"🎚️  Extracting {channel} channel...", file=sys.stderr)
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return tmp_path, tmp_path
    except subprocess.CalledProcessError:
        if not quiet:
            import sys
            print("⚠️  Channel extraction failed; using full mix", file=sys.stderr)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return audio_path, None


# ---------------------------------------------------------------------------
# Subtitle burn-in
# ---------------------------------------------------------------------------

def burn_subtitles(video_path, srt_content, output_path, quiet=False):
    """Burn SRT subtitles into a video file using ffmpeg."""
    import sys
    tmp_srt = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".srt", delete=False, encoding="utf-8"
        ) as f:
            f.write(srt_content)
            tmp_srt = f.name

        escaped = tmp_srt.replace("\\", "/").replace(":", "\\:")
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"subtitles={escaped}",
            "-c:a", "copy",
            output_path,
        ]
        if not quiet:
            print(f"🎬 Burning subtitles into {output_path}...", file=sys.stderr)
            subprocess.run(cmd, check=True)
        else:
            subprocess.run(cmd, check=True, capture_output=True)
        if not quiet:
            print(f"✅ Burned: {output_path}", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Burn-in failed: {e}", file=sys.stderr)
    finally:
        if tmp_srt and os.path.exists(tmp_srt):
            os.unlink(tmp_srt)


# ---------------------------------------------------------------------------
# URL download
# ---------------------------------------------------------------------------

def is_url(path):
    """Check if the input looks like a URL."""
    return path.startswith(("http://", "https://", "www."))


def download_url(url, audio_format="wav", quiet=False):
    """Download audio from URL using yt-dlp. Returns (audio_path, tmpdir)."""
    import sys
    ytdlp = shutil.which("yt-dlp")
    if not ytdlp:
        pipx_path = Path.home() / ".local/share/pipx/venvs/yt-dlp/bin/yt-dlp"
        if pipx_path.exists():
            ytdlp = str(pipx_path)
        else:
            print("Error: yt-dlp not found. Install with: pipx install yt-dlp", file=sys.stderr)
            sys.exit(1)

    tmpdir = tempfile.mkdtemp(prefix="super-transcribe-")
    out_tmpl = os.path.join(tmpdir, "audio.%(ext)s")

    cmd = [ytdlp, "-x", "--audio-format", audio_format, "-o", out_tmpl, "--no-playlist"]
    if quiet:
        cmd.append("-q")
    cmd.append(url)

    if not quiet:
        print("⬇️  Downloading audio from URL...", file=sys.stderr)

    try:
        subprocess.run(cmd, check=True, capture_output=quiet)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading URL: {e}", file=sys.stderr)
        shutil.rmtree(tmpdir, ignore_errors=True)
        sys.exit(1)

    files = list(Path(tmpdir).glob("audio.*"))
    if not files:
        print("Error: No audio file downloaded", file=sys.stderr)
        shutil.rmtree(tmpdir, ignore_errors=True)
        sys.exit(1)

    return str(files[0]), tmpdir


# ---------------------------------------------------------------------------
# Audio format conversion (for NeMo: prefers .wav 16kHz mono)
# ---------------------------------------------------------------------------

NATIVE_EXTS = {".wav", ".flac"}
CONVERTIBLE_EXTS = {
    ".mp3", ".m4a", ".mp4", ".mkv", ".avi", ".wma", ".aac",
    ".ogg", ".webm", ".opus",
}
AUDIO_EXTS = NATIVE_EXTS | CONVERTIBLE_EXTS


def convert_to_wav(audio_path, quiet=False):
    """Convert non-wav/flac audio to 16kHz mono WAV using ffmpeg.
    Returns (wav_path, tmp_path_to_cleanup_or_None).
    """
    ext = Path(audio_path).suffix.lower()
    if ext in NATIVE_EXTS:
        return audio_path, None

    if not shutil.which("ffmpeg"):
        import sys
        print(
            f"⚠️  ffmpeg not found — cannot convert {ext} to WAV.",
            file=sys.stderr,
        )
        return audio_path, None

    tmp_path = audio_path + ".converted.wav"
    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        tmp_path,
    ]
    if not quiet:
        import sys
        print(f"🔄 Converting {ext} → WAV (16kHz mono)...", file=sys.stderr)

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return tmp_path, tmp_path
    except subprocess.CalledProcessError as e:
        if not quiet:
            import sys
            print(f"⚠️  Conversion failed: {e}. Trying original file.", file=sys.stderr)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return audio_path, None
