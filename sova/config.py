"""Configuration constants and paths."""

import platform
import sys
from pathlib import Path

import sqlite_vector

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
DOCS_DIR = SCRIPT_DIR / "docs"
DB_PATH = DATA_DIR / "refs.db"

assert sqlite_vector.__file__ is not None
_binaries = Path(sqlite_vector.__file__).parent / "binaries"
_ext = {"Darwin": "vector.dylib", "Linux": "vector.so", "Windows": "vector.dll"}
_platform = platform.system()
if _platform not in _ext:
    print(f"unsupported platform: {_platform}", file=sys.stderr)
    sys.exit(1)
VECTOR_EXT = _binaries / _ext[_platform]

EMBEDDING_MODEL = "qwen3-embedding:4b"
CONTEXT_MODEL = "gemma3:12b"
EMBEDDING_DIM = 2560

# Number of documents to embed per batch.
BATCH_SIZE = 10
# Target words per chunk. 512 balances embedding quality (models degrade on
# very long inputs) against preserving enough context per chunk.
CHUNK_SIZE = 512
