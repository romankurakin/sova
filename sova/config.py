"""Configuration constants and paths."""

from pathlib import Path

import sqlite_vector

# Models
EMBEDDING_MODEL = "qwen3-embedding:8b"
LLM_MODEL = "gemma3:4b"
EMBEDDING_DIM = 1024

# Paths
SCRIPT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
DOCS_DIR = SCRIPT_DIR / "docs"
DB_PATH = DATA_DIR / "refs.db"

# Vector extension
assert sqlite_vector.__file__ is not None
VECTOR_EXT = Path(sqlite_vector.__file__).parent / "binaries" / "vector.dylib"

# Processing
BATCH_SIZE = 10
CHUNK_SIZE = 512
