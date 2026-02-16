"""Install and remove helpers for global Sova binary usage."""

import argparse
import json
import os
import plistlib
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SERVICES: list[dict[str, Any]] = [
    {
        "name": "embedding",
        "label": "com.sova.embedding",
        "port": 8081,
        "hf_repo": "Qwen/Qwen3-Embedding-4B-GGUF",
        "hf_file": "Qwen3-Embedding-4B-Q8_0.gguf",
        "extra_args": ["--embedding", "--pooling", "last"],
        "keep_alive": False,
    },
    {
        "name": "reranker",
        "label": "com.sova.reranker",
        "port": 8082,
        "hf_repo": "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF",
        "extra_args": ["--rerank"],
        "keep_alive": False,
    },
    {
        "name": "chat",
        "label": "com.sova.chat",
        "port": 8083,
        "hf_repo": "mistralai/Ministral-3-14B-Instruct-2512-GGUF",
        "hf_file": "Ministral-3-14B-Instruct-2512-Q8_0.gguf",
        "extra_args": ["--ctx-size", "4096"],
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
            raise SystemExit(
                f"{label} already exists: {path} (use sova-remove or --force)"
            )
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
        raise SystemExit(f"missing build script: {build_script}")
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", str(repo_root / ".uv-cache"))
    env.setdefault("PYINSTALLER_CONFIG_DIR", str(repo_root / ".pyinstaller"))
    subprocess.run([str(build_script)], cwd=repo_root, env=env, check=True)


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _plist_dir() -> Path:
    return Path.home() / "Library" / "LaunchAgents"


def _plist_path(label: str) -> Path:
    return _plist_dir() / f"{label}.plist"


def _logs_dir() -> Path:
    return _default_home() / "logs"


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
        "StartInterval": 600,  # every 10 minutes
        "StandardOutPath": str(logs / "watchdog.log"),
        "StandardErrorPath": str(logs / "watchdog.err.log"),
    }


def _install_services(
    llama_server_path: str, binary_path: Path, force: bool
) -> list[str]:
    _plist_dir().mkdir(parents=True, exist_ok=True)
    _logs_dir().mkdir(parents=True, exist_ok=True)
    installed: list[str] = []
    for svc in SERVICES:
        plist = _plist_path(svc["label"])
        if plist.exists():
            if not force:
                print(f"  skip {svc['label']}: plist exists (use --force)")
                continue
            subprocess.run(
                ["launchctl", "unload", str(plist)],
                capture_output=True,
            )
        plist_data = _generate_plist(svc, llama_server_path)
        with open(plist, "wb") as f:
            plistlib.dump(plist_data, f)
        subprocess.run(["launchctl", "load", str(plist)], check=True)
        installed.append(str(plist))
        print(f"  loaded {svc['label']} (port {svc['port']})")

    # Install watchdog that stops idle services automatically.
    watchdog_plist = _plist_path(_WATCHDOG_LABEL)
    if watchdog_plist.exists():
        subprocess.run(
            ["launchctl", "unload", str(watchdog_plist)], capture_output=True
        )
    watchdog_data = _generate_watchdog_plist(binary_path)
    with open(watchdog_plist, "wb") as f:
        plistlib.dump(watchdog_data, f)
    subprocess.run(["launchctl", "load", str(watchdog_plist)], check=True)
    installed.append(str(watchdog_plist))
    print(f"  loaded {_WATCHDOG_LABEL} (idle watchdog)")

    return installed


def _remove_services() -> None:
    # Remove watchdog first.
    watchdog_plist = _plist_path(_WATCHDOG_LABEL)
    if watchdog_plist.exists():
        subprocess.run(
            ["launchctl", "unload", str(watchdog_plist)], capture_output=True
        )
        watchdog_plist.unlink()
        print(f"  unloaded {_WATCHDOG_LABEL}")

    for svc in SERVICES:
        plist = _plist_path(svc["label"])
        if not plist.exists():
            continue
        subprocess.run(
            ["launchctl", "unload", str(plist)],
            capture_output=True,
        )
        plist.unlink()
        print(f"  unloaded {svc['label']}")

    # Clean up activity files.
    activity_dir = _default_home() / "activity"
    if activity_dir.exists():
        shutil.rmtree(activity_dir)


def install_main() -> None:
    parser = argparse.ArgumentParser(description="Build and install global sova binary")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip build step and install existing dist/sova",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing installed files/symlinks",
    )
    args = parser.parse_args()

    if sys.platform != "darwin":
        raise SystemExit("sova-install currently supports macOS only")

    repo_root = _repo_root()
    home = _default_home()
    binary_path = _default_binary_path()
    manifest_path = _manifest_path(home)
    existing_manifest = _read_manifest(manifest_path)
    replace_managed = bool(existing_manifest)
    allow_replace = args.force or replace_managed

    if not args.skip_build:
        _run_build(repo_root)

    built_binary = repo_root / "dist" / "sova"
    if not built_binary.exists():
        raise SystemExit(f"missing built binary: {built_binary}")

    _install_binary(built_binary, binary_path, allow_replace)

    home.mkdir(parents=True, exist_ok=True)
    data_dir = home / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    llama_server = shutil.which("llama-server")
    service_paths: list[str] = []
    if llama_server:
        print("installing launchd services...")
        service_paths = _install_services(llama_server, binary_path, allow_replace)
    else:
        print(
            "warning: llama-server not found in PATH, skipping service install\n"
            "  install llama.cpp and re-run sova-install to enable services"
        )

    manifest = {
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "home": str(home),
        "binary_path": str(binary_path),
        "services": service_paths,
    }
    _write_manifest(manifest_path, manifest)

    print(f"installed binary: {binary_path}")
    print(f"sova home: {home}")
    print(f"data: {data_dir}")
    if str(binary_path.parent) not in os.environ.get("PATH", "").split(os.pathsep):
        print(f"note: add to PATH -> {binary_path.parent}")


def remove_main() -> None:
    parser = argparse.ArgumentParser(description="Remove global sova binary install")
    parser.add_argument(
        "--purge-data",
        action="store_true",
        help="Also remove regular DB file under SOVA_HOME/data if present",
    )
    args = parser.parse_args()

    home = _default_home()
    manifest_path = _manifest_path(home)
    manifest = _read_manifest(manifest_path)

    _remove_services()

    binary_path = Path(manifest.get("binary_path", _default_binary_path())).expanduser()

    if binary_path.exists() or binary_path.is_symlink():
        _remove_path(binary_path)
        print(f"removed binary: {binary_path}")

    if args.purge_data:
        data_dir = home / "data"
        if data_dir.exists():
            shutil.rmtree(data_dir)
            print(f"removed data: {data_dir}")

    if manifest_path.exists():
        manifest_path.unlink()
        print(f"removed manifest: {manifest_path}")

    logs_dir = _logs_dir()
    if logs_dir.exists():
        shutil.rmtree(logs_dir)
        print(f"removed logs: {logs_dir}")

    data_dir = home / "data"
    if data_dir.exists() and not any(data_dir.iterdir()):
        data_dir.rmdir()
    if home.exists() and not any(home.iterdir()):
        home.rmdir()
