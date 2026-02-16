"""Configuration constants and paths."""

import json
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


def _path_from_env(env_name: str, default: Path) -> Path:
    value = os.environ.get(env_name)
    return Path(value).expanduser() if value else default


PROJECT_ROOT = _project_root()
SOVA_HOME = _path_from_env("SOVA_HOME", Path.home() / ".sova")
DATA_DIR = _path_from_env("SOVA_DATA_DIR", SOVA_HOME / "data")
DB_PATH = _path_from_env("SOVA_DB_PATH", DATA_DIR / "indexed.db")
_CONFIG_PATH = SOVA_HOME / "config.json"


def _read_config() -> dict:
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _write_config(data: dict) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def get_docs_dir() -> Path | None:
    """Return configured PDF source directory, or None if not set."""
    env = os.environ.get("SOVA_DOCS_DIR")
    if env:
        return Path(env).expanduser()
    cfg = _read_config()
    if "docs_dir" in cfg:
        return Path(cfg["docs_dir"]).expanduser()
    return None


def set_docs_dir(path: str | Path) -> None:
    """Persist the PDF source directory in config.json."""
    resolved = Path(path).expanduser().resolve()
    cfg = _read_config()
    cfg["docs_dir"] = str(resolved)
    _write_config(cfg)


def get_db_path() -> Path:
    """Return database path."""
    return DB_PATH


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

# llama-server endpoints (one instance per model)
EMBEDDING_SERVER_URL = "http://localhost:8081"
RERANKER_SERVER_URL = "http://localhost:8082"
CONTEXT_SERVER_URL = "http://localhost:8083"
# Embedding model used for indexing and query vectors
EMBEDDING_MODEL = "qwen3-embedding-4b"
# LLM model used for context answers and analysis
CONTEXT_MODEL = "ministral-3-14b-instruct-2512"
# Reranker model for second-stage ranking
RERANKER_MODEL = "qwen3-reranker-0.6b"
# Vector dimension expected from embedding model
EMBEDDING_DIM = 2560
# Number of top RRF results sent to reranker
# Reranker sees limit * RERANK_FACTOR candidates
RERANK_FACTOR = 2
# Reranker timeout in seconds; graceful fallback if exceeded
RERANK_TIMEOUT = 10.0
# RRF base rank constant controlling how strongly top results are favored
SEARCH_RRF_K = 20
# Weight multiplier applied to RRF score in final ranking
SEARCH_RRF_WEIGHT = 30.0
# Bonus added when full query phrase appears in chunk text
SEARCH_EXACT_PHRASE_BONUS = 0.3
# Bonus multiplier for fraction of query terms found in chunk text
SEARCH_EXACT_TERM_BONUS = 0.15
# Penalty applied to index-like chunks to demote boilerplate hits
SEARCH_INDEX_PENALTY = -0.5
# Decay factor for diversity reranking across similar documents
SEARCH_DIVERSITY_DECAY = 0.95
# Number of documents to embed per batch
BATCH_SIZE = 10
# Target words per chunk
CHUNK_SIZE = 512
