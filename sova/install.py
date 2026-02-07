"""Install and remove helpers for global Sova binary usage."""

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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

    docs_source = repo_root / "docs"
    docs_target = home / "docs"
    if docs_source.exists():
        _link_path(docs_source, docs_target, allow_replace, "docs link")
    else:
        print(f"warning: docs source not found, skipping link: {docs_source}")

    db_source = repo_root / "data" / "indexed.db"
    db_target = data_dir / "indexed.db"
    if db_source.exists():
        _link_path(db_source, db_target, allow_replace, "db link")
    else:
        print(f"warning: db source not found, skipping db link: {db_source}")

    manifest = {
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "home": str(home),
        "binary_path": str(binary_path),
        "binary_mode": "copy",
        "docs_path": str(docs_target),
        "docs_source": str(docs_source),
        "docs_mode": "symlink",
        "db_path": str(db_target),
        "db_source": str(db_source),
        "db_mode": "symlink",
    }
    _write_manifest(manifest_path, manifest)

    print(f"installed binary: {binary_path}")
    print(f"sova home: {home}")
    print(f"docs: {docs_target}")
    print(f"db: {db_target}")
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

    binary_path = Path(manifest.get("binary_path", _default_binary_path())).expanduser()
    docs_path = Path(manifest.get("docs_path", home / "docs")).expanduser()
    db_path = Path(manifest.get("db_path", home / "data" / "indexed.db")).expanduser()

    if binary_path.exists() or binary_path.is_symlink():
        _remove_path(binary_path)
        print(f"removed binary: {binary_path}")

    if docs_path.is_symlink():
        _remove_path(docs_path)
        print(f"removed docs link: {docs_path}")

    if db_path.is_symlink():
        _remove_path(db_path)
        print(f"removed db link: {db_path}")
    elif args.purge_data and db_path.exists():
        _remove_path(db_path)
        print(f"removed db file: {db_path}")

    if manifest_path.exists():
        manifest_path.unlink()
        print(f"removed manifest: {manifest_path}")

    data_dir = home / "data"
    if data_dir.exists() and not any(data_dir.iterdir()):
        data_dir.rmdir()
    if home.exists() and not any(home.iterdir()):
        home.rmdir()
