"""Configuration constants and paths."""

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
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
PROJECTS_DIR = SOVA_HOME / "projects"

_ACTIVE_PROJECT_ID: str | None = None
_ACTIVE_PROJECT_NAME: str | None = None
_ACTIVE_PROJECT_DOCS_DIR: Path | None = None
_ACTIVE_PROJECT_ROOT_DIR: Path | None = None
_ACTIVE_PROJECT_DATA_DIR: Path | None = None
_ACTIVE_PROJECT_DB_PATH: Path | None = None

# Memory policy defaults.
_DEFAULT_RESERVE_INDEX_GIB = 4.0
_DEFAULT_RESERVE_SEARCH_GIB = 10.0
_DEFAULT_METAL_HEADROOM_GIB = 1.0
_DEFAULT_SWAP_WEIGHT = 0.25
_DEFAULT_SWAP_BOOST_CAP_GIB = 2.0
_DEFAULT_METAL_FALLBACK_RATIO = 0.8
_METAL_PROBE_CACHE_TTL_S = 30.0

# Fixed model set in Sova; estimates are used for admission checks only.
_MODEL_MEMORY_ESTIMATE_GIB = {
    "embed": 9.5,
    "reranker": 1.5,
    "chat": 16.0,
}

_METAL_PROBE_CACHE: dict[str, float | None] = {
    "value": None,
    "ts": 0.0,
}


def _read_config() -> dict:
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _write_config(data: dict) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _CONFIG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.rename(_CONFIG_PATH)


def get_docs_dir() -> Path | None:
    """Return active project docs directory, or None when not selected."""
    if _ACTIVE_PROJECT_DOCS_DIR is None:
        return None
    return _ACTIVE_PROJECT_DOCS_DIR


def activate_project(
    *,
    project_id: str,
    project_name: str,
    docs_dir: Path,
    root_dir: Path,
    data_dir: Path,
    db_path: Path,
) -> None:
    """Activate per-project paths for this process."""
    global _ACTIVE_PROJECT_ID
    global _ACTIVE_PROJECT_NAME
    global _ACTIVE_PROJECT_DOCS_DIR
    global _ACTIVE_PROJECT_ROOT_DIR
    global _ACTIVE_PROJECT_DATA_DIR
    global _ACTIVE_PROJECT_DB_PATH
    _ACTIVE_PROJECT_ID = project_id
    _ACTIVE_PROJECT_NAME = project_name
    _ACTIVE_PROJECT_DOCS_DIR = docs_dir.expanduser().resolve()
    _ACTIVE_PROJECT_ROOT_DIR = root_dir.expanduser().resolve()
    _ACTIVE_PROJECT_DATA_DIR = data_dir.expanduser().resolve()
    _ACTIVE_PROJECT_DB_PATH = db_path.expanduser().resolve()


def clear_active_project() -> None:
    """Clear active project override."""
    global _ACTIVE_PROJECT_ID
    global _ACTIVE_PROJECT_NAME
    global _ACTIVE_PROJECT_DOCS_DIR
    global _ACTIVE_PROJECT_ROOT_DIR
    global _ACTIVE_PROJECT_DATA_DIR
    global _ACTIVE_PROJECT_DB_PATH
    _ACTIVE_PROJECT_ID = None
    _ACTIVE_PROJECT_NAME = None
    _ACTIVE_PROJECT_DOCS_DIR = None
    _ACTIVE_PROJECT_ROOT_DIR = None
    _ACTIVE_PROJECT_DATA_DIR = None
    _ACTIVE_PROJECT_DB_PATH = None


def get_active_project_id() -> str | None:
    return _ACTIVE_PROJECT_ID


def get_active_project_name() -> str | None:
    return _ACTIVE_PROJECT_NAME


def get_active_project_root_dir() -> Path | None:
    return _ACTIVE_PROJECT_ROOT_DIR


def get_data_dir() -> Path:
    """Return active project data directory (or fallback DATA_DIR)."""
    if _ACTIVE_PROJECT_DATA_DIR is not None:
        return _ACTIVE_PROJECT_DATA_DIR
    return DATA_DIR


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _probe_total_ram_gib() -> float:
    """Best-effort system RAM detection in GiB."""
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                mem_bytes = int(result.stdout.strip())
                return mem_bytes / (1024**3)
        except Exception:
            pass
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return float(page_size * pages) / (1024**3)
    except Exception:
        return 0.0


def _probe_available_ram_gib() -> float:
    """Best-effort available memory in GiB (macOS vm_stat, else total RAM)."""
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["vm_stat"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                text = result.stdout
                m = re.search(r"page size of (\d+) bytes", text)
                page_size = int(m.group(1)) if m else 4096
                page_counts: dict[str, int] = {}
                for line in text.splitlines():
                    if ":" not in line:
                        continue
                    key, raw_value = line.split(":", 1)
                    clean = raw_value.strip().rstrip(".").replace(".", "").replace(",", "")
                    if not clean.isdigit():
                        continue
                    page_counts[key.strip().lower()] = int(clean)
                available_pages = (
                    page_counts.get("pages free", 0)
                    + page_counts.get("pages inactive", 0)
                    + page_counts.get("pages speculative", 0)
                    + page_counts.get("pages purgeable", 0)
                )
                if available_pages > 0:
                    return float(available_pages * page_size) / (1024**3)
        except Exception:
            pass
    return _probe_total_ram_gib()


def _probe_free_swap_gib() -> float:
    """Best-effort free swap detection in GiB."""
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "vm.swapusage"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                text = f"{result.stdout}\n{result.stderr}"
                m = re.search(r"free\s*=\s*([0-9.]+)\s*([KMGTP])", text)
                if m:
                    value = float(m.group(1))
                    unit = m.group(2)
                    scale = {
                        "K": 1.0 / (1024**2),
                        "M": 1.0 / 1024.0,
                        "G": 1.0,
                        "T": 1024.0,
                        "P": 1024.0 * 1024.0,
                    }
                    return value * scale.get(unit, 0.0)
        except Exception:
            pass
    return 0.0


def _probe_metal_ceiling_gib() -> float | None:
    """Detect recommended Metal working set size from llama-server startup logs."""
    if sys.platform != "darwin":
        return None
    llama_server = shutil.which("llama-server")
    if not llama_server:
        return None
    try:
        result = subprocess.run(
            [llama_server, "--help"],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception:
        return None
    combined = f"{result.stdout}\n{result.stderr}"
    m = re.search(r"recommendedMaxWorkingSetSize\s*=\s*([0-9.]+)\s*MB", combined)
    if not m:
        return None
    try:
        mb = float(m.group(1))
    except ValueError:
        return None
    if mb <= 0:
        return None
    return mb / 1024.0


def _runtime_metal_recommended_gib() -> float | None:
    """Return cached runtime Metal recommendation (best-effort)."""
    now = time.monotonic()
    cached_ts = float(_METAL_PROBE_CACHE["ts"] or 0.0)
    cached_value = _METAL_PROBE_CACHE["value"]
    if cached_value is not None and (now - cached_ts) < _METAL_PROBE_CACHE_TTL_S:
        return float(cached_value)

    detected = _probe_metal_ceiling_gib()
    if detected is not None:
        _METAL_PROBE_CACHE["value"] = float(detected)
        _METAL_PROBE_CACHE["ts"] = now
        return detected

    # If probing fails temporarily, keep using last known recommendation.
    if cached_value is not None:
        return float(cached_value)
    return None


def _mode_key(mode: str) -> str:
    return "search" if mode == "search" else "index"


def _memory_cfg() -> dict:
    """Return optional user-provided memory overrides from config.json."""
    cfg = _read_config()
    memory_cfg = cfg.get("memory")
    if isinstance(memory_cfg, dict):
        return memory_cfg
    return {}


def get_memory_settings() -> dict:
    """Return current runtime memory settings snapshot."""
    return {
        "total_ram_gib": round(_probe_total_ram_gib(), 2),
        "reserve_index_gib": get_memory_reserve_gib("index"),
        "reserve_search_gib": get_memory_reserve_gib("search"),
        "swap_weight": get_swap_weight(),
        "swap_boost_cap_gib": get_swap_boost_cap_gib(),
        "metal_ceiling_gib": get_metal_ceiling_gib(),
    }


def get_memory_reserve_gib(mode: str) -> float:
    """Return configured reserve memory in GiB for mode: index/search."""
    mode_key = _mode_key(mode)
    env_override = _env_float(f"SOVA_MEMORY_RESERVE_{mode_key.upper()}_GIB")
    if env_override is not None:
        return max(0.0, env_override)
    settings = _memory_cfg()
    if mode_key == "search":
        return max(
            0.0,
            _parse_float(settings.get("reserve_search_gib"), _DEFAULT_RESERVE_SEARCH_GIB),
        )
    return max(
        0.0,
        _parse_float(settings.get("reserve_index_gib"), _DEFAULT_RESERVE_INDEX_GIB),
    )


def get_metal_ceiling_gib() -> float:
    """Return runtime Metal ceiling in GiB.

    Priority:
      1) env override
      2) runtime probe from llama-server recommendedMaxWorkingSetSize (cached)
      3) config.json manual override
      4) runtime fallback derived from total RAM ratio
    """
    env_override = _env_float("SOVA_MEMORY_METAL_CEILING_GIB")
    if env_override is not None:
        return max(0.0, env_override)

    detected = _runtime_metal_recommended_gib()
    if detected is not None:
        headroom = _env_float("SOVA_MEMORY_METAL_HEADROOM_GIB")
        if headroom is None:
            headroom = max(
                0.0,
                _parse_float(
                    _memory_cfg().get("metal_headroom_gib"),
                    _DEFAULT_METAL_HEADROOM_GIB,
                ),
            )
        return max(0.0, round(detected - max(0.0, headroom), 2))

    cfg_override = _parse_float(_memory_cfg().get("metal_ceiling_gib"), -1.0)
    if cfg_override >= 0.0:
        return cfg_override
    ratio = _env_float("SOVA_MEMORY_METAL_FALLBACK_RATIO")
    if ratio is None:
        ratio = max(
            0.0,
            _parse_float(
                _memory_cfg().get("metal_fallback_ratio"),
                _DEFAULT_METAL_FALLBACK_RATIO,
            ),
        )
    return round(max(0.0, _probe_total_ram_gib() * max(0.0, ratio)), 2)


def get_available_ram_gib() -> float:
    """Return current available RAM in GiB."""
    env_override = _env_float("SOVA_MEMORY_AVAILABLE_GIB")
    if env_override is not None:
        return max(0.0, env_override)
    return max(0.0, _probe_available_ram_gib())


def get_free_swap_gib() -> float:
    """Return current free swap in GiB."""
    env_override = _env_float("SOVA_MEMORY_FREE_SWAP_GIB")
    if env_override is not None:
        return max(0.0, env_override)
    return max(0.0, _probe_free_swap_gib())


def get_swap_weight() -> float:
    """Return weight applied to free swap for effective availability."""
    env_override = _env_float("SOVA_MEMORY_SWAP_WEIGHT")
    if env_override is not None:
        return max(0.0, env_override)
    settings = _memory_cfg()
    return max(0.0, _parse_float(settings.get("swap_weight"), _DEFAULT_SWAP_WEIGHT))


def get_swap_boost_cap_gib() -> float:
    """Return max swap contribution to effective available memory."""
    env_override = _env_float("SOVA_MEMORY_SWAP_BOOST_CAP_GIB")
    if env_override is not None:
        return max(0.0, env_override)
    settings = _memory_cfg()
    return max(
        0.0,
        _parse_float(
            settings.get("swap_boost_cap_gib"),
            _DEFAULT_SWAP_BOOST_CAP_GIB,
        ),
    )


def get_effective_available_gib() -> float:
    """Return effective available memory with discounted swap contribution."""
    available_ram = get_available_ram_gib()
    free_swap = get_free_swap_gib()
    swap_boost = min(free_swap * get_swap_weight(), get_swap_boost_cap_gib())
    return round(available_ram + swap_boost, 2)


def get_memory_hard_cap_gib(mode: str) -> float:
    """Compute runtime hard cap: min(metal_ceiling, available_now - reserve(mode))."""
    env_override = _env_float("SOVA_MEMORY_HARD_CAP_GIB")
    if env_override is not None:
        return max(0.0, env_override)
    ceiling = get_metal_ceiling_gib()
    available = get_effective_available_gib()
    reserve = get_memory_reserve_gib(mode)
    cap = min(ceiling, available - reserve)
    return round(max(0.0, cap), 2)


def get_model_memory_estimate_gib(model_kind: str) -> float:
    """Return static model memory estimate in GiB used by admission checks."""
    return _MODEL_MEMORY_ESTIMATE_GIB.get(model_kind, 0.0)


def get_db_path() -> Path:
    """Return database path."""
    if _ACTIVE_PROJECT_DB_PATH is not None:
        return _ACTIVE_PROJECT_DB_PATH
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
