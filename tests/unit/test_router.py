"""
Unit tests for scripts/transcribe bash router.

Strategy: copy the router to tmp_path, create stub backend scripts that
echo 'BACKEND=faster-whisper' or 'BACKEND=parakeet', and create fake
venv python files so backend_ready() passes.

The router uses SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
so the entire directory tree must be replicated under tmp_path.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

# ── Constants ──────────────────────────────────────────────────────────────────

ROUTER_SRC = Path(__file__).parent.parent.parent / "scripts" / "transcribe"
LIB_SRC = Path(__file__).parent.parent.parent / "scripts" / "backends" / "lib"
TONE_WAV = Path(__file__).parent.parent / "fixtures" / "tone.wav"

# AIDEV-NOTE: Stub scripts just report which backend was selected; no real transcription.
_STUB_FW = """\
#!/usr/bin/env bash
echo "BACKEND=faster-whisper"
echo "ARGS=$*"
exit 0
"""

_STUB_PK = """\
#!/usr/bin/env bash
echo "BACKEND=parakeet"
echo "ARGS=$*"
exit 0
"""

# Minimal python3 stub used for the fake venv pythons — just prints nothing and exits.
# backend_ready() only checks that the file *exists* (not that it's a real interpreter),
# but --version and check_health call it to query package versions. The fake python
# prints "unknown" for any invocation, which is acceptable in these tests.
_STUB_PYTHON = """\
#!/usr/bin/env bash
# Stub python used by backend_ready() path checks and --version queries.
echo "unknown"
exit 0
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_router_env(tmp_path: Path, fw_ready: bool = True, pk_ready: bool = True) -> Path:
    """Build an isolated router environment under tmp_path.

    Replicates the directory structure that SCRIPT_DIR resolves to:
        tmp_path/transcribe          ← router script (copy)
        tmp_path/backends/lib/       ← shared lib (copy; needed for --probe)
        tmp_path/backends/faster-whisper/transcribe   ← stub
        tmp_path/backends/faster-whisper/.venv/bin/python  (if fw_ready)
        tmp_path/backends/parakeet/transcribe          ← stub
        tmp_path/backends/parakeet/venv/bin/python     (if pk_ready)

    Returns tmp_path for use as the working env root.
    """
    # Copy router script
    router = tmp_path / "transcribe"
    shutil.copy(ROUTER_SRC, router)
    router.chmod(0o755)

    # Copy shared lib so --probe (python3 inline script) can import lib.audio
    shutil.copytree(LIB_SRC, tmp_path / "backends" / "lib")

    # ── faster-whisper backend ────────────────────────────────────────────────
    fw_dir = tmp_path / "backends" / "faster-whisper"
    fw_dir.mkdir(parents=True)
    fw_stub = fw_dir / "transcribe"
    fw_stub.write_text(_STUB_FW)
    fw_stub.chmod(0o755)
    if fw_ready:
        # AIDEV-NOTE: backend_ready("faster-whisper") checks $FW_DIR/.venv/bin/python
        venv_bin = fw_dir / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        fake_py = venv_bin / "python"
        fake_py.write_text(_STUB_PYTHON)
        fake_py.chmod(0o755)

    # ── parakeet backend ──────────────────────────────────────────────────────
    pk_dir = tmp_path / "backends" / "parakeet"
    pk_dir.mkdir(parents=True)
    pk_stub = pk_dir / "transcribe"
    pk_stub.write_text(_STUB_PK)
    pk_stub.chmod(0o755)
    if pk_ready:
        # AIDEV-NOTE: backend_ready("parakeet") checks $PK_DIR/venv/bin/python (no leading dot)
        venv_bin = pk_dir / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        fake_py = venv_bin / "python"
        fake_py.write_text(_STUB_PYTHON)
        fake_py.chmod(0o755)

    return tmp_path


def run(env: Path, *args: str, timeout: int = 10) -> subprocess.CompletedProcess:
    """Run the router with the given args; always capture stdout + stderr."""
    return subprocess.run(
        ["bash", str(env / "transcribe"), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def router_env(tmp_path: Path) -> Path:
    """Router environment: both backends available and ready."""
    return _make_router_env(tmp_path)


@pytest.fixture
def router_env_fw_only(tmp_path: Path) -> Path:
    """Router environment: only faster-whisper ready; parakeet available but not ready."""
    return _make_router_env(tmp_path, fw_ready=True, pk_ready=False)


@pytest.fixture
def router_env_pk_only(tmp_path: Path) -> Path:
    """Router environment: only parakeet ready; faster-whisper available but not ready."""
    return _make_router_env(tmp_path, fw_ready=False, pk_ready=True)


@pytest.fixture
def router_env_no_backends(tmp_path: Path) -> Path:
    """Router environment: neither backend has a venv (not ready)."""
    return _make_router_env(tmp_path, fw_ready=False, pk_ready=False)


# ── Test: --version ───────────────────────────────────────────────────────────


class TestVersion:
    def test_version_prints_version_string(self, router_env: Path) -> None:
        """--version must print 'super-transcribe' and exit 0."""
        result = run(router_env, "--version")
        assert result.returncode == 0
        assert "super-transcribe" in result.stdout

    def test_version_contains_semver(self, router_env: Path) -> None:
        """--version output must contain a semver-like string."""
        result = run(router_env, "--version")
        # E.g. "super-transcribe 0.1.0"
        import re
        assert re.search(r"\d+\.\d+\.\d+", result.stdout), (
            f"No semver found in: {result.stdout!r}"
        )


# ── Test: --backends ──────────────────────────────────────────────────────────


class TestBackendsList:
    def test_both_ready(self, router_env: Path) -> None:
        """--backends with both ready should show both as installed."""
        result = run(router_env, "--backends")
        assert result.returncode == 0
        assert "faster-whisper" in result.stdout
        assert "parakeet" in result.stdout
        # Both should be marked as installed & ready
        assert "installed" in result.stdout

    def test_fw_only_ready(self, router_env_fw_only: Path) -> None:
        """When only faster-whisper is ready, parakeet shows as not set up."""
        result = run(router_env_fw_only, "--backends")
        assert result.returncode == 0
        assert "faster-whisper" in result.stdout
        # Parakeet is bundled (dir exists) but not yet set up
        assert "parakeet" in result.stdout

    def test_no_backends_ready(self, router_env_no_backends: Path) -> Path:
        """When neither is ready, --backends still exits 0 (informational)."""
        result = run(router_env_no_backends, "--backends")
        assert result.returncode == 0
        # Should still list both as bundled
        assert "faster-whisper" in result.stdout
        assert "parakeet" in result.stdout


# ── Test: default routing ─────────────────────────────────────────────────────


class TestDefaultRouting:
    def test_default_prefers_parakeet_when_available(self, router_env: Path) -> None:
        """When both backends are available, the router defaults to parakeet."""
        result = run(router_env, str(TONE_WAV))
        assert result.returncode == 0
        assert "BACKEND=parakeet" in result.stdout

    def test_default_falls_back_to_fw_when_parakeet_unavailable(
        self, router_env_fw_only: Path
    ) -> None:
        """When only faster-whisper is ready, the router routes there by default.

        Note: backend_ready() is not what drives the default route — the router
        uses backend_available() (checks directory + script exist) for selection,
        then auto-installs if not ready. Here both dirs exist, so parakeet is
        'available'. The routing still picks parakeet (preferred), but parakeet is
        not ready so it will attempt auto-setup. To test FW fallback cleanly we
        need to remove the parakeet backend directory entirely.
        """
        # Remove the parakeet backend directory so backend_available("parakeet") → false
        pk_dir = router_env_fw_only / "backends" / "parakeet"
        shutil.rmtree(pk_dir)

        result = run(router_env_fw_only, str(TONE_WAV))
        assert result.returncode == 0
        assert "BACKEND=faster-whisper" in result.stdout

    def test_no_backends_exits_nonzero(self, tmp_path: Path) -> None:
        """When no backends are present at all, the router exits with code 2."""
        # Copy router only, no backends/ directory
        router = tmp_path / "transcribe"
        shutil.copy(ROUTER_SRC, router)
        router.chmod(0o755)

        result = run(tmp_path, str(TONE_WAV))
        assert result.returncode == 2


# ── Test: --backend flag ──────────────────────────────────────────────────────


class TestBackendFlag:
    """--backend accepts: faster-whisper, fw, whisper, parakeet, pk, nemo."""

    @pytest.mark.parametrize("alias", ["faster-whisper", "fw", "whisper"])
    def test_backend_fw_aliases(self, router_env: Path, alias: str) -> None:
        result = run(router_env, "--backend", alias, str(TONE_WAV))
        assert result.returncode == 0
        assert "BACKEND=faster-whisper" in result.stdout

    @pytest.mark.parametrize("alias", ["parakeet", "pk", "nemo"])
    def test_backend_pk_aliases(self, router_env: Path, alias: str) -> None:
        result = run(router_env, "--backend", alias, str(TONE_WAV))
        assert result.returncode == 0
        assert "BACKEND=parakeet" in result.stdout

    def test_backend_equals_form_fw(self, router_env: Path) -> None:
        """--backend=fw should also work."""
        result = run(router_env, "--backend=fw", str(TONE_WAV))
        assert result.returncode == 0
        assert "BACKEND=faster-whisper" in result.stdout

    def test_backend_equals_form_pk(self, router_env: Path) -> None:
        """--backend=pk should also work."""
        result = run(router_env, "--backend=pk", str(TONE_WAV))
        assert result.returncode == 0
        assert "BACKEND=parakeet" in result.stdout

    def test_backend_unknown_exits_3(self, router_env: Path) -> None:
        """Unknown --backend value exits with code 3."""
        result = run(router_env, "--backend", "unknown-backend", str(TONE_WAV))
        assert result.returncode == 3

    def test_backend_flag_not_passed_through(self, router_env: Path) -> None:
        """--backend is consumed by the router and NOT forwarded to the backend stub."""
        result = run(router_env, "--backend", "fw", str(TONE_WAV))
        assert "--backend" not in result.stdout


# ── Test: faster-whisper-only flags ──────────────────────────────────────────


class TestFWOnlyFlags:
    """Flags in FW_ONLY_NOVAL / FW_ONLY_WITHVAL must route to faster-whisper."""

    def test_translate_routes_to_fw(self, router_env: Path) -> None:
        result = run(router_env, "--translate", str(TONE_WAV))
        assert result.returncode == 0
        assert "BACKEND=faster-whisper" in result.stdout

    def test_multilingual_routes_to_fw(self, router_env: Path) -> None:
        result = run(router_env, "--multilingual", str(TONE_WAV))
        assert result.returncode == 0
        assert "BACKEND=faster-whisper" in result.stdout

    def test_word_timestamps_routes_to_fw(self, router_env: Path) -> None:
        result = run(router_env, "--word-timestamps", str(TONE_WAV))
        assert result.returncode == 0
        assert "BACKEND=faster-whisper" in result.stdout

    def test_initial_prompt_routes_to_fw(self, router_env: Path) -> None:
        result = run(router_env, "--initial-prompt", "Hello", str(TONE_WAV))
        assert result.returncode == 0
        assert "BACKEND=faster-whisper" in result.stdout

    def test_compute_type_routes_to_fw(self, router_env: Path) -> None:
        result = run(router_env, "--compute-type", "float16", str(TONE_WAV))
        assert result.returncode == 0
        assert "BACKEND=faster-whisper" in result.stdout

    def test_temperature_routes_to_fw(self, router_env: Path) -> None:
        result = run(router_env, "--temperature", "0.0", str(TONE_WAV))
        assert result.returncode == 0
        assert "BACKEND=faster-whisper" in result.stdout

    def test_no_vad_routes_to_fw(self, router_env: Path) -> None:
        result = run(router_env, "--no-vad", str(TONE_WAV))
        assert result.returncode == 0
        assert "BACKEND=faster-whisper" in result.stdout

    def test_log_level_routes_to_fw(self, router_env: Path) -> None:
        result = run(router_env, "--log-level", "debug", str(TONE_WAV))
        assert result.returncode == 0
        assert "BACKEND=faster-whisper" in result.stdout

    def test_fw_flags_passed_through_to_backend(self, router_env: Path) -> None:
        """FW-only flags should appear in ARGS passed to the backend stub."""
        result = run(router_env, "--translate", str(TONE_WAV))
        assert "--translate" in result.stdout


# ── Test: parakeet-only flags ─────────────────────────────────────────────────


class TestPKOnlyFlags:
    """Flags in PK_ONLY_NOVAL / PK_ONLY_WITHVAL must route to parakeet."""

    def test_long_form_routes_to_pk(self, router_env: Path) -> None:
        result = run(router_env, "--long-form", str(TONE_WAV))
        assert result.returncode == 0
        assert "BACKEND=parakeet" in result.stdout

    def test_streaming_routes_to_pk(self, router_env: Path) -> None:
        result = run(router_env, "--streaming", str(TONE_WAV))
        assert result.returncode == 0
        assert "BACKEND=parakeet" in result.stdout

    def test_fast_routes_to_pk(self, router_env: Path) -> None:
        result = run(router_env, "--fast", str(TONE_WAV))
        assert result.returncode == 0
        assert "BACKEND=parakeet" in result.stdout

    def test_multitalker_routes_to_pk(self, router_env: Path) -> None:
        result = run(router_env, "--multitalker", str(TONE_WAV))
        assert result.returncode == 0
        assert "BACKEND=parakeet" in result.stdout

    def test_source_lang_routes_to_pk(self, router_env: Path) -> None:
        result = run(router_env, "--source-lang", "fr", str(TONE_WAV))
        assert result.returncode == 0
        assert "BACKEND=parakeet" in result.stdout

    def test_target_lang_routes_to_pk(self, router_env: Path) -> None:
        result = run(router_env, "--target-lang", "en", str(TONE_WAV))
        assert result.returncode == 0
        assert "BACKEND=parakeet" in result.stdout

    def test_pk_flags_passed_through_to_backend(self, router_env: Path) -> None:
        """PK-only flags should appear in ARGS passed to the backend stub."""
        result = run(router_env, "--long-form", str(TONE_WAV))
        assert "--long-form" in result.stdout


# ── Test: language routing ─────────────────────────────────────────────────────


class TestLanguageRouting:
    """Non-EU languages force faster-whisper; EU/supported langs use default routing."""

    # AIDEV-NOTE: PARAKEET_LANGS = "bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk"

    @pytest.mark.parametrize("lang", ["ja", "zh", "ar", "ko", "hi"])
    def test_non_eu_language_routes_to_fw(self, router_env: Path, lang: str) -> None:
        """Non-European languages (not in PARAKEET_LANGS) force faster-whisper."""
        result = run(router_env, "-l", lang, str(TONE_WAV))
        assert result.returncode == 0
        assert "BACKEND=faster-whisper" in result.stdout, (
            f"Expected faster-whisper for lang={lang!r}, got: {result.stdout!r}"
        )

    @pytest.mark.parametrize("lang", ["en", "de", "fr", "es", "it", "pl", "ru"])
    def test_eu_language_uses_default_routing(self, router_env: Path, lang: str) -> None:
        """EU/Parakeet-supported languages don't force faster-whisper.

        With both backends available the default is parakeet.
        """
        result = run(router_env, "-l", lang, str(TONE_WAV))
        assert result.returncode == 0
        assert "BACKEND=parakeet" in result.stdout, (
            f"Expected parakeet (default) for EU lang={lang!r}, got: {result.stdout!r}"
        )

    def test_language_flag_long_form_non_eu(self, router_env: Path) -> None:
        """--language=ja (equals form) also forces faster-whisper."""
        result = run(router_env, "--language=ja", str(TONE_WAV))
        assert result.returncode == 0
        assert "BACKEND=faster-whisper" in result.stdout

    def test_language_flag_long_form_eu(self, router_env: Path) -> None:
        """--language=fr (equals form) uses default parakeet routing."""
        result = run(router_env, "--language=fr", str(TONE_WAV))
        assert result.returncode == 0
        assert "BACKEND=parakeet" in result.stdout


# ── Test: conflicting flags ────────────────────────────────────────────────────


class TestConflictingFlags:
    """Combining FW-only and PK-only flags must exit with code 3."""

    def test_translate_and_long_form_exits_3(self, router_env: Path) -> None:
        """--translate (FW) + --long-form (PK) is a conflicting combination."""
        result = run(router_env, "--translate", "--long-form", str(TONE_WAV))
        assert result.returncode == 3

    def test_translate_and_streaming_exits_3(self, router_env: Path) -> None:
        """--translate (FW) + --streaming (PK) is a conflicting combination."""
        result = run(router_env, "--translate", "--streaming", str(TONE_WAV))
        assert result.returncode == 3

    def test_word_timestamps_and_fast_exits_3(self, router_env: Path) -> None:
        """--word-timestamps (FW) + --fast (PK) is a conflicting combination."""
        result = run(router_env, "--word-timestamps", "--fast", str(TONE_WAV))
        assert result.returncode == 3

    def test_conflicting_error_message_on_stderr(self, router_env: Path) -> None:
        """Conflicting-flag error message goes to stderr."""
        result = run(router_env, "--translate", "--long-form", str(TONE_WAV))
        assert "Conflicting" in result.stderr or "conflicting" in result.stderr

    def test_non_eu_lang_and_long_form_exits_3(self, router_env: Path) -> None:
        """-l ja (forces FW) + --long-form (forces PK) is conflicting."""
        result = run(router_env, "-l", "ja", "--long-form", str(TONE_WAV))
        assert result.returncode == 3


# ── Test: --probe ──────────────────────────────────────────────────────────────


class TestProbeMode:
    def test_probe_with_valid_wav(self, router_env: Path) -> None:
        """--probe on a valid WAV file should exit 0 and emit JSON."""
        result = run(router_env, "--probe", str(TONE_WAV))
        assert result.returncode == 0
        data = json.loads(result.stdout.strip())
        # probe_audio returns at least duration and format information
        assert "duration" in data or "format" in data or "sample_rate" in data

    def test_probe_without_file_exits_3(self, router_env: Path) -> None:
        """--probe with no audio file should exit 3."""
        result = run(router_env, "--probe")
        assert result.returncode == 3

    def test_probe_does_not_invoke_backend(self, router_env: Path) -> None:
        """--probe must exit before routing to any backend."""
        result = run(router_env, "--probe", str(TONE_WAV))
        assert "BACKEND=" not in result.stdout


# ── Test: passthrough args ────────────────────────────────────────────────────


class TestPassthrough:
    """Miscellaneous and shared flags should pass through unchanged."""

    def test_format_flag_passes_through(self, router_env: Path) -> None:
        """--format is a shared flag and should be forwarded to the backend."""
        result = run(router_env, "--format", "srt", str(TONE_WAV))
        assert result.returncode == 0
        assert "--format" in result.stdout

    def test_diarize_passes_through(self, router_env: Path) -> None:
        """--diarize is a shared flag and should be forwarded to the backend."""
        result = run(router_env, "--diarize", str(TONE_WAV))
        assert result.returncode == 0
        assert "--diarize" in result.stdout

    def test_audio_file_passed_to_backend(self, router_env: Path) -> None:
        """The audio file path should appear in the backend's ARGS."""
        result = run(router_env, str(TONE_WAV))
        assert result.returncode == 0
        assert str(TONE_WAV) in result.stdout

    def test_multiple_audio_files_passed_through(self, router_env: Path, tmp_path: Path) -> None:
        """Multiple audio files should all be forwarded to the selected backend."""
        wav2 = tmp_path / "tone2.wav"
        shutil.copy(TONE_WAV, wav2)
        result = run(router_env, str(TONE_WAV), str(wav2))
        assert result.returncode == 0
        assert str(TONE_WAV) in result.stdout
        assert str(wav2) in result.stdout


# ── Test: backend readiness / auto-setup UX ───────────────────────────────────


class TestBackendReadiness:
    """When a backend is selected but not ready, the router emits a setup notice."""

    def test_info_message_when_backend_not_ready(self, router_env_no_backends: Path) -> None:
        """When the selected backend isn't ready, a setup message goes to stderr.

        With neither backend ready (but both available), the router picks parakeet
        by default and prints a first-time setup message to stderr.
        The stub setup invocation will fail (no setup.sh), so we just check the
        message appears before that failure.
        """
        result = run(router_env_no_backends, str(TONE_WAV))
        # The router should mention setup / first-time on stderr
        combined = result.stderr + result.stdout
        assert "setup" in combined.lower() or "first" in combined.lower() or "install" in combined.lower()

    def test_no_setup_message_when_backend_ready(self, router_env: Path) -> None:
        """When both backends are ready, no 'first-time setup' message appears."""
        result = run(router_env, str(TONE_WAV))
        assert result.returncode == 0
        # The first-run message contains "First-time setup" or "Setting up"
        assert "First-time setup" not in result.stderr
        assert "Setting up" not in result.stderr


# ── Test: --check and --check --json ──────────────────────────────────────────


def _extract_json_from_output(output: str) -> dict:
    """Extract the JSON object from output that may be prefixed with non-JSON lines.

    check_health --json calls fake venv python stubs that each print "unknown" to
    stdout before the actual JSON line (the router doesn't redirect those calls).
    This helper finds the first line that starts with '{' and parses it.
    """
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError(f"No JSON object found in output: {output!r}")


class TestCheckHealth:
    """--check and --check --json require python3 on PATH which is always present."""

    def test_check_exits_with_code_on_exit(self, router_env: Path) -> None:
        """--check exits 0 when a backend is ready, 2 otherwise."""
        result = run(router_env, "--check")
        # Both backends are ready, so exit code should be 0
        assert result.returncode == 0

    def test_check_json_outputs_valid_json(self, router_env: Path) -> None:
        """--check --json must produce parseable JSON on a line starting with '{'."""
        result = run(router_env, "--check", "--json")
        # Exit 0 because at least one backend is ready
        assert result.returncode == 0
        # AIDEV-NOTE: fake venv stubs print "unknown" before the JSON line; use helper
        data = _extract_json_from_output(result.stdout)
        assert "ready" in data
        assert "backends" in data

    def test_check_json_ready_true_when_backends_ready(self, router_env: Path) -> None:
        """--check --json must return ready=true when a backend is installed."""
        result = run(router_env, "--check", "--json")
        data = _extract_json_from_output(result.stdout)
        assert data["ready"] is True

    def test_check_json_shows_both_backends(self, router_env: Path) -> None:
        """--check --json backend fields reflect actual readiness."""
        result = run(router_env, "--check", "--json")
        data = _extract_json_from_output(result.stdout)
        backends = data["backends"]
        assert backends["faster_whisper"]["installed"] is True
        assert backends["parakeet"]["installed"] is True

    def test_check_json_not_ready_without_backends(self, tmp_path: Path) -> None:
        """--check --json exits 2 and ready=false when no backends are installed."""
        # Router with no backends/ directory at all
        router = tmp_path / "transcribe"
        shutil.copy(ROUTER_SRC, router)
        router.chmod(0o755)
        result = run(tmp_path, "--check", "--json")
        assert result.returncode == 2
        data = _extract_json_from_output(result.stdout)
        assert data["ready"] is False
