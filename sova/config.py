"""Configuration constants and paths."""

import os
import platform
import sys
from importlib.resources import files
from pathlib import Path

import sqlite_vector


def _is_frozen_binary() -> bool:
    return bool(getattr(sys, "frozen", False))


def _project_root() -> Path:
    if _is_frozen_binary():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _default_home_dir() -> Path:
    if _is_frozen_binary():
        return Path.home() / ".sova"
    return _project_root()


def _path_from_env(env_name: str, default: Path) -> Path:
    value = os.environ.get(env_name)
    return Path(value).expanduser() if value else default


PROJECT_ROOT = _project_root()
SOVA_HOME = _path_from_env("SOVA_HOME", _default_home_dir())
DATA_DIR = _path_from_env("SOVA_DATA_DIR", SOVA_HOME / "data")
DOCS_DIR = _path_from_env("SOVA_DOCS_DIR", SOVA_HOME / "docs")
DB_PATH = _path_from_env("SOVA_DB_PATH", DATA_DIR / "indexed.db")
_db_path_override: Path | None = None


def set_db_path(path: str | Path | None) -> None:
    """Override database path for the current process."""
    global _db_path_override
    _db_path_override = Path(path).expanduser() if path else None


def get_db_path() -> Path:
    """Return effective database path (CLI override or configured default)."""
    return _db_path_override or DB_PATH


def _resolve_vector_extension() -> Path:
    ext_by_platform = {
        "Darwin": "vector.dylib",
        "Linux": "vector.so",
        "Windows": "vector.dll",
    }
    platform_name = platform.system()
    if platform_name not in ext_by_platform:
        print(f"unsupported platform: {platform_name}", file=sys.stderr)
        sys.exit(1)
    filename = ext_by_platform[platform_name]

    # First try a regular site-packages layout.
    if sqlite_vector.__file__:
        candidate = Path(sqlite_vector.__file__).parent / "binaries" / filename
        if candidate.exists():
            return candidate

    # Then try package resources, which also works in frozen apps.
    try:
        candidate = Path(str(files("sqlite_vector").joinpath("binaries", filename)))
        if candidate.exists():
            return candidate
    except Exception:
        pass

    # Last fallback for frozen bootloaders that unpack under _MEIPASS.
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidate = Path(meipass) / "sqlite_vector" / "binaries" / filename
        if candidate.exists():
            return candidate

    print("sqlite_vector extension library not found", file=sys.stderr)
    sys.exit(1)


VECTOR_EXT = _resolve_vector_extension()

EMBEDDING_MODEL = "qwen3-embedding:4b"
CONTEXT_MODEL = "gemma3:12b"
EMBEDDING_DIM = 2560

SEARCH_RRF_K = 20
SEARCH_RRF_WEIGHT = 30.0
SEARCH_EXACT_PHRASE_BONUS = 0.3
SEARCH_EXACT_TERM_BONUS = 0.15
SEARCH_INDEX_PENALTY = -0.5
SEARCH_DIVERSITY_DECAY = 0.95

# Number of documents to embed per batch.
BATCH_SIZE = 10
# Target words per chunk. 512 balances embedding quality (models degrade on
# very long inputs) against preserving enough context per chunk.
CHUNK_SIZE = 512
