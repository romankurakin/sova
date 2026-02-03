"""Configuration constants and paths."""

from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
DOCS_DIR = SCRIPT_DIR / "docs"
DB_PATH = DATA_DIR / "index.db"

EMBEDDING_MODEL = "qwen3-embedding:4b"
CONTEXT_MODEL = "gemma3:12b"
EMBEDDING_DIM = 2560

# Number of documents to embed per batch.
BATCH_SIZE = 10
# Target words per chunk. 512 balances embedding quality (models degrade on
# very long inputs) against preserving enough context per chunk.
CHUNK_SIZE = 512
