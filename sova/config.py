"""Configuration constants and paths."""

import platform
import sys
from pathlib import Path

import sqlite_vector

EMBEDDING_MODEL = "qwen3-embedding:4b"
EMBEDDING_DIM = 2560

# Asymmetric retrieval: queries get this instruction prefix, chunks don't.
# This tells the model to optimize for query-passage matching rather than
# generic similarity, which measurably improves recall on retrieval tasks.
QUERY_TASK = "Given a search query, retrieve relevant passages that answer the query"

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

BATCH_SIZE = 10
# Target words per chunk. 512 balances embedding quality (models degrade on
# very long inputs) against preserving enough context per chunk.
CHUNK_SIZE = 512

# Diversification prevents top results from being dominated by a single doc.
# These were tuned empirically: 2 per doc keeps variety without dropping
# highly relevant repeated hits, 1 per section avoids near-duplicate content.
MAX_PER_DOC = 2
MAX_PER_SECTION = 1
MIN_UNIQUE_DOCS = 3
