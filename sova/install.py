"""Install and remove helpers for global Sova binary usage."""

import argparse
import json
import os
import plistlib
import shutil
import stat
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sova.config import CONTEXT_MODEL_HF_FILE, CONTEXT_MODEL_HF_REPO
from sova.ui import report, report_error, report_mode, report_step

SERVICES: list[dict[str, Any]] = [
    {
        "name": "embedding",
        "label": "com.sova.embedding",
        "port": 8081,
        "hf_repo": "Qwen/Qwen3-Embedding-4B-GGUF",
        "hf_file": "Qwen3-Embedding-4B-Q8_0.gguf",
        "extra_args": [
            "--embedding",
            "--pooling",
            "last",
            "--flash-attn",
            "off",
            "--ctx-size",
            "4096",
            "--parallel",
            "1",
            "--cache-ram",
            "0",
            "--sleep-idle-seconds",
            "600",
        ],
        "env": {
            "MallocNanoZone": "0",
            "GGML_METAL_NO_RESIDENCY": "1",
        },
        "keep_alive": False,
    },
    {
        "name": "chat",
        "label": "com.sova.chat",
        "port": 8083,
        "hf_repo": CONTEXT_MODEL_HF_REPO,
        "hf_file": CONTEXT_MODEL_HF_FILE,
        "extra_args": [
            # Context generation is text-only. On the target 32 GiB Mac,
            # parallel=2 and MTP both reduced measured throughput under memory
            # pressure, so keep one slot and the model's native decode path.
            "--no-mmproj",
            "--ctx-size",
            "4096",
            "--parallel",
            "1",
            "--cache-ram",
            "0",
            "--reasoning",
            "on",
            "--reasoning-effort",
            "low",
            "--reasoning-budget",
            "64",
            "--sleep-idle-seconds",
            "900",
        ],
        "keep_alive": False,
    },
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_home() -> Path:
    return Path.home() / ".sova"


def _default_binary_path() -> Path:
    return Path.home() / ".local" / "bin" / "sova"


def _manifest_path(home: Path) -> Path:
    return home / "install-manifest.json"


def _resolve_link_target(link: Path) -> Path | None:
    if not link.is_symlink():
        return None
    raw = Path(os.readlink(link))
    if raw.is_absolute():
        return raw.resolve()
    return (link.parent / raw).resolve()


def _remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def _ensure_replaceable(path: Path, force: bool, label: str) -> None:
    if path.exists() or path.is_symlink():
        if not force:
            report_error(
                f"{label} already exists",
                cause=str(path),
                action="run sova-remove first",
            )
            raise SystemExit(1)
        _remove_path(path)


def _link_path(source: Path, destination: Path, force: bool, label: str) -> None:
    if destination.exists() or destination.is_symlink():
        if (
            destination.is_symlink()
            and _resolve_link_target(destination) == source.resolve()
        ):
            return
        _ensure_replaceable(destination, force, label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve())


def _copy_file(source: Path, destination: Path, force: bool, label: str) -> None:
    if destination.exists() or destination.is_symlink():
        _ensure_replaceable(destination, force, label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _install_binary(source: Path, destination: Path, force: bool) -> None:
    _copy_file(source, destination, force, "binary")
    mode = destination.stat().st_mode
    destination.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _run_build(repo_root: Path) -> None:
    build_script = repo_root / "scripts" / "build-macos-onefile.sh"
    if not build_script.exists():
        report_error(
            "build script not found",
            cause=str(build_script),
            action="ensure repository is complete and retry",
        )
        raise SystemExit(1)
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", str(repo_root / ".uv-cache"))
    env.setdefault("PYINSTALLER_CONFIG_DIR", str(repo_root / ".pyinstaller"))
    result = subprocess.run(
        [str(build_script)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        error_text = result.stderr.strip() or "build failed"
        report_error(
            "build failed",
            cause=error_text,
            action="fix build errors and retry sova-install",
        )
        raise SystemExit(1)


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError, UnicodeError, json.JSONDecodeError:
        return {}


def _plist_dir() -> Path:
    return Path.home() / "Library" / "LaunchAgents"


def _plist_path(label: str) -> Path:
    return _plist_dir() / f"{label}.plist"


def _logs_dir() -> Path:
    return _default_home() / "logs"


_COMPLETION_SHELLS = ("bash", "zsh", "fish")


def _install_shell_completion(shell: str) -> bool:
    """Install completion via Typer's installer with an explicit shell.

    Calling the typer API directly avoids shellingham process-tree detection,
    which is unreliable when sova-install itself runs under another shell.
    """
    from typer._completion_shared import install as typer_install_completion

    try:
        typer_install_completion(
            shell=shell, prog_name="sova", complete_var="_SOVA_COMPLETE"
        )
    except OSError:
        return False
    return True


def _detect_shells() -> list[str]:
    """Detect which supported shells are present on this system."""
    return [s for s in _COMPLETION_SHELLS if shutil.which(s)]


def _install_shell_completions() -> tuple[list[str], list[str]]:
    """Install completions for every detected shell.

    Returns (installed, failed) shell names.
    """
    installed: list[str] = []
    failed: list[str] = []
    for shell in _detect_shells():
        if _install_shell_completion(shell):
            installed.append(shell)
        else:
            failed.append(shell)
    return installed, failed


def _fish_completions_dir() -> Path:
    return Path.home() / ".config" / "fish" / "completions"


def _remove_shell_completions() -> None:
    """Remove completion artifacts that are safe to delete.

    Fish completions are standalone files. Bash/zsh scripts stay in place
    because their rc files source them by absolute path; deleting the script
    would break shell startup, while a stale script is inert.
    """
    comp_dir = _fish_completions_dir()
    # Includes legacy files written by older sova-install versions.
    for command in ("sova", "sova-install", "sova-remove"):
        path = comp_dir / f"{command}.fish"
        if path.exists():
            path.unlink()


def _generate_plist(service: dict[str, Any], llama_server_path: str) -> dict[str, Any]:
    logs = _logs_dir()
    name = service["name"]
    keep_alive = service.get("keep_alive", True)
    args = [llama_server_path, "-hf", service["hf_repo"]]
    if "hf_file" in service:
        args += ["-hff", service["hf_file"]]
    args += ["--port", str(service["port"])]
    args += service["extra_args"]
    plist: dict[str, Any] = {
        "Label": service["label"],
        "ProgramArguments": args,
        "RunAtLoad": keep_alive,
        "StandardOutPath": str(logs / f"{name}.log"),
        "StandardErrorPath": str(logs / f"{name}.err.log"),
    }
    env_vars = service.get("env")
    if env_vars:
        plist["EnvironmentVariables"] = env_vars
    if keep_alive:
        plist["KeepAlive"] = True
    return plist


_WATCHDOG_LABEL = "com.sova.watchdog"


def _generate_watchdog_plist(binary_path: Path) -> dict[str, Any]:
    """Generate a plist for the idle-service watchdog."""
    logs = _logs_dir()
    return {
        "Label": _WATCHDOG_LABEL,
        "ProgramArguments": [str(binary_path), "--_watchdog"],
        "StartInterval": 600,  # every 10 minutes.
        "StandardOutPath": str(logs / "watchdog.log"),
        "StandardErrorPath": str(logs / "watchdog.err.log"),
    }


def _install_services(
    llama_server_path: str, binary_path: Path, force: bool
) -> tuple[list[str], int, int]:
    _plist_dir().mkdir(parents=True, exist_ok=True)
    _logs_dir().mkdir(parents=True, exist_ok=True)
    installed: list[str] = []
    loaded_count = 0
    skipped_count = 0
    for svc in SERVICES:
        plist = _plist_path(svc["label"])
        if plist.exists():
            if not force:
                skipped_count += 1
                continue
            subprocess.run(
                ["launchctl", "unload", str(plist)],
                capture_output=True,
                check=False,
            )
        plist_data = _generate_plist(svc, llama_server_path)
        with open(plist, "wb") as f:
            plistlib.dump(plist_data, f)
        subprocess.run(["launchctl", "load", str(plist)], check=True)
        installed.append(str(plist))
        loaded_count += 1

    # Install watchdog that stops idle services automatically.
    watchdog_plist = _plist_path(_WATCHDOG_LABEL)
    if watchdog_plist.exists():
        subprocess.run(
            ["launchctl", "unload", str(watchdog_plist)],
            capture_output=True,
            check=False,
        )
    watchdog_data = _generate_watchdog_plist(binary_path)
    with open(watchdog_plist, "wb") as f:
        plistlib.dump(watchdog_data, f)
    subprocess.run(["launchctl", "load", str(watchdog_plist)], check=True)
    installed.append(str(watchdog_plist))
    loaded_count += 1

    return installed, loaded_count, skipped_count


def _remove_services() -> int:
    unloaded_count = 0
    # Remove watchdog first.
    watchdog_plist = _plist_path(_WATCHDOG_LABEL)
    if watchdog_plist.exists():
        subprocess.run(
            ["launchctl", "unload", str(watchdog_plist)],
            capture_output=True,
            check=False,
        )
        watchdog_plist.unlink()
        unloaded_count += 1

    for svc in SERVICES:
        plist = _plist_path(svc["label"])
        if not plist.exists():
            continue
        subprocess.run(
            ["launchctl", "unload", str(plist)],
            capture_output=True,
            check=False,
        )
        plist.unlink()
        unloaded_count += 1

    # Clean up activity files.
    activity_dir = _default_home() / "activity"
    if activity_dir.exists():
        shutil.rmtree(activity_dir)
    return unloaded_count


def install_main() -> None:
    parser = argparse.ArgumentParser(description="Build and install global sova binary")
    parser.parse_args()
    report_mode("install")

    if sys.platform != "darwin":
        report_error(
            "unsupported platform",
            cause="sova-install currently supports macOS only",
        )
        raise SystemExit(1)

    repo_root = _repo_root()
    home = _default_home()
    binary_path = _default_binary_path()
    manifest_path = _manifest_path(home)
    existing_manifest = _read_manifest(manifest_path)
    replace_managed = bool(existing_manifest)
    allow_replace = replace_managed

    report_step("build")
    _run_build(repo_root)

    built_binary = repo_root / "dist" / "sova"
    if not built_binary.exists():
        report_error(
            "built binary not found",
            cause=str(built_binary),
            action="rerun sova-install to rebuild binary",
        )
        raise SystemExit(1)

    report_step("binary")
    _install_binary(built_binary, binary_path, allow_replace)

    home.mkdir(parents=True, exist_ok=True)
    data_dir = home / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    llama_server = shutil.which("llama-server")
    service_paths: list[str] = []
    _loaded_services = 0
    skipped_services = 0
    if llama_server:
        report_step("services")
        service_paths, _loaded_services, skipped_services = _install_services(
            llama_server, binary_path, allow_replace
        )
    else:
        report(
            "warning",
            "llama-server not found in PATH; skipping service install",
        )
        report("hint", "install llama.cpp and re-run sova-install")

    manifest = {
        "installed_at": datetime.now(UTC).isoformat(),
        "repo_root": str(repo_root),
        "home": str(home),
        "binary_path": str(binary_path),
        "services": service_paths,
    }
    _write_manifest(manifest_path, manifest)

    if llama_server:
        summary = "loaded"
        if skipped_services:
            summary = "loaded (some skipped)"
        report("services", summary)
    installed_shells, failed_shells = _install_shell_completions()
    if installed_shells:
        report("shell", f"completions installed: {', '.join(installed_shells)}")
    if failed_shells:
        report("hint", "completions: sova --install-completion")
    report("status", "install complete")
    if str(binary_path.parent) not in os.environ.get("PATH", "").split(os.pathsep):
        report("hint", "add ~/.local/bin to PATH")


def remove_main() -> None:
    parser = argparse.ArgumentParser(description="Remove global sova binary install")
    parser.add_argument(
        "--purge-data",
        action="store_true",
        help="Also remove ~/.sova",
    )
    args = parser.parse_args()
    report_mode("remove")

    home = _default_home()
    manifest_path = _manifest_path(home)
    manifest = _read_manifest(manifest_path)

    _unloaded_services = _remove_services()

    binary_path = Path(manifest.get("binary_path", _default_binary_path())).expanduser()

    if binary_path.exists() or binary_path.is_symlink():
        _remove_path(binary_path)
    _remove_shell_completions()

    if args.purge_data:
        if home.exists():
            shutil.rmtree(home)
        report("services", "unloaded")
        report("status", "remove complete")
        return

    if manifest_path.exists():
        manifest_path.unlink()

    logs_dir = _logs_dir()
    if logs_dir.exists():
        shutil.rmtree(logs_dir)

    data_dir = home / "data"
    if data_dir.exists() and not any(data_dir.iterdir()):
        data_dir.rmdir()
    if home.exists() and not any(home.iterdir()):
        home.rmdir()

    report("services", "unloaded")
    report("status", "remove complete")
