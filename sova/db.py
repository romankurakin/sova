"""Database initialization and connection management."""

import math
import sqlite3
import struct
from collections.abc import Generator
from contextlib import contextmanager

from sova import config
from sova.config import EMBEDDING_DIM, VECTOR_EXT

SCHEMA_VERSION = 7
VECTOR_INDEX_STATE_KEY = "index.vector.state"
_COMPACTION_PENDING_KEY = "storage.compaction.pending"


def _check_schema_version(conn: sqlite3.Connection) -> int:
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"database schema {version} is newer than supported {SCHEMA_VERSION}"
        )
    return version


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_xinfo({table})")}


# SQLite trim() defaults to ASCII spaces; match Python str.strip() exactly.
# Keep this expression self-contained so ordinary SQLite connections can read it.
_STRIP_CODEPOINTS = (
    *range(9, 14),
    *range(28, 33),
    133,
    160,
    5760,
    *range(8192, 8203),
    8232,
    8233,
    8239,
    8287,
    12288,
)
_SEARCH_TEXT_SQL = (
    "CASE WHEN search_prefix IS NULL THEN text ELSE search_prefix || "
    f"trim(text, char({','.join(map(str, _STRIP_CODEPOINTS))})) END"
)


def _drop_fts_triggers(conn: sqlite3.Connection) -> None:
    for name in ("chunks_ai", "chunks_ad", "chunks_au"):
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")


def _deduplicate_search_text(conn: sqlite3.Connection) -> bool:
    columns = {row[1]: row for row in conn.execute("PRAGMA table_xinfo(chunks)")}
    if "search_text" in columns and columns["search_text"][6] == 2:
        return False
    if "search_prefix" not in columns:
        conn.execute("ALTER TABLE chunks ADD COLUMN search_prefix TEXT")
    if "search_text" in columns:
        # Preserve the actually indexed input, not a reconstruction from possibly
        # newer document names or contexts. Bound memory for large corpora.
        cursor = conn.execute("SELECT id, text, search_text FROM chunks ORDER BY id")
        while rows := cursor.fetchmany(256):
            prefixes = []
            for chunk_id, source, search in rows:
                if search == source:
                    prefix = None  # Legacy raw text must retain whitespace too.
                else:
                    body = source.strip()
                    if not isinstance(search, str) or not search.endswith(body):
                        raise RuntimeError(
                            f"cannot safely deduplicate search text for chunk {chunk_id}"
                        )
                    prefix = search[: -len(body)] if body else search
                prefixes.append((prefix, chunk_id))
            conn.executemany(
                "UPDATE chunks SET search_prefix = ? WHERE id = ?", prefixes
            )
        if conn.execute(
            f"SELECT 1 FROM chunks WHERE search_text IS NOT ({_SEARCH_TEXT_SQL}) LIMIT 1"
        ).fetchone():
            raise RuntimeError(
                "search text reconstruction differs; migration cancelled"
            )
        _drop_fts_triggers(conn)
        conn.execute("ALTER TABLE chunks DROP COLUMN search_text")
    conn.execute(
        "ALTER TABLE chunks ADD COLUMN search_text TEXT "
        f"GENERATED ALWAYS AS ({_SEARCH_TEXT_SQL}) VIRTUAL"
    )
    return "search_text" in columns


def _ensure_fts(conn: sqlite3.Connection) -> None:
    exists = bool(
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunks_fts'"
        ).fetchone()
    )
    rebuild = not exists
    _drop_fts_triggers(conn)
    if exists and "search_text" not in _column_names(conn, "chunks_fts"):
        conn.execute("DROP TABLE chunks_fts")
        rebuild = True
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            search_text, content='chunks', content_rowid='id',
            tokenize='porter unicode61'
        )
    """)
    conn.execute("""
        CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, search_text) VALUES (new.id, new.search_text);
        END
    """)
    conn.execute("""
        CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, search_text)
            VALUES('delete', old.id, old.search_text);
        END
    """)
    conn.execute("""
        CREATE TRIGGER chunks_au AFTER UPDATE OF id, text, search_prefix ON chunks
        WHEN old.id IS NOT new.id OR old.search_text IS NOT new.search_text
        BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, search_text)
            VALUES('delete', old.id, old.search_text);
            INSERT INTO chunks_fts(rowid, search_text) VALUES (new.id, new.search_text);
        END
    """)
    if rebuild:
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Apply small, transactional schema upgrades to existing project databases."""
    _check_schema_version(conn)
    if not conn.in_transaction:
        conn.execute("BEGIN IMMEDIATE")
    with conn:
        chunks_columns = _column_names(conn, "chunks")
        if "embedding_signature" not in chunks_columns:
            conn.execute("ALTER TABLE chunks ADD COLUMN embedding_signature TEXT")
        if "section_path" not in chunks_columns:
            conn.execute(
                "ALTER TABLE chunks ADD COLUMN section_path TEXT NOT NULL DEFAULT ''"
            )
        if _deduplicate_search_text(conn):
            conn.execute(
                "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?, '1')",
                (_COMPACTION_PENDING_KEY,),
            )

        context_columns = _column_names(conn, "chunk_contexts")
        if "pipeline_signature" not in context_columns:
            conn.execute(
                "ALTER TABLE chunk_contexts "
                "ADD COLUMN pipeline_signature TEXT NOT NULL DEFAULT ''"
            )

        document_columns = _column_names(conn, "documents")
        if document_columns and "source_signature" not in document_columns:
            conn.execute("ALTER TABLE documents ADD COLUMN source_signature TEXT")
        if document_columns and "chunk_signature" not in document_columns:
            conn.execute("ALTER TABLE documents ADD COLUMN chunk_signature TEXT")

        _ensure_fts(conn)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def init_db() -> sqlite3.Connection:
    """Initialize database with tables and indexes."""
    config.get_data_dir().mkdir(parents=True, exist_ok=True)
    db_path = config.get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(str(VECTOR_EXT))
    conn.enable_load_extension(False)
    _check_schema_version(conn)

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL,
            path TEXT NOT NULL, line_count INTEGER, expected_chunks INTEGER,
            source_signature TEXT, chunk_signature TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS sections (
            id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL, title TEXT NOT NULL,
            level INTEGER NOT NULL, start_line INTEGER NOT NULL, end_line INTEGER,
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL, section_id INTEGER,
            start_line INTEGER NOT NULL, end_line INTEGER NOT NULL,
            word_count INTEGER NOT NULL, text TEXT NOT NULL, embedding BLOB,
            embedding_signature TEXT,
            section_path TEXT NOT NULL DEFAULT '',
            search_prefix TEXT,
            is_index INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS query_cache (
            id INTEGER PRIMARY KEY,
            embedding BLOB NOT NULL,
            vector_results BLOB NOT NULL,
            created_at REAL NOT NULL,
            model TEXT NOT NULL,
            candidate_count INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunk_contexts (
            chunk_id INTEGER PRIMARY KEY,
            context TEXT NOT NULL,
            model TEXT NOT NULL,
            pipeline_signature TEXT NOT NULL DEFAULT '',
            FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS index_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
        CREATE INDEX IF NOT EXISTS idx_query_cache_created ON query_cache(created_at);
        PRAGMA foreign_keys = ON;
    """)
    try:
        _migrate_schema(conn)
        # DROP COLUMN releases payload but does not shrink the file. The durable
        # marker retries compaction after interruption without redoing models or
        # the migration. VACUUM itself is transactional.
        if conn.execute(
            "SELECT 1 FROM index_meta WHERE key = ?", (_COMPACTION_PENDING_KEY,)
        ).fetchone():
            conn.execute("VACUUM")
            conn.execute(
                "DELETE FROM index_meta WHERE key = ?", (_COMPACTION_PENDING_KEY,)
            )
            conn.commit()
    except BaseException:
        conn.close()
        raise

    # A durable marker survives interruption between saved embeddings and
    # quantization. Existing indexes rebuild once to adopt this checkpoint.
    conn.execute(
        "INSERT OR IGNORE INTO index_meta (key, value) VALUES (?, 'pending')",
        (VECTOR_INDEX_STATE_KEY,),
    )
    for suffix, event in (
        ("ai", "INSERT"),
        ("ad", "DELETE"),
        (
            "au",
            (
                "UPDATE OF id, embedding ON chunks "
                "WHEN old.id IS NOT new.id OR old.embedding IS NOT new.embedding"
            ),
        ),
    ):
        clause = event if suffix == "au" else f"{event} ON chunks"
        conn.execute(f"""
            CREATE TRIGGER IF NOT EXISTS chunks_vector_{suffix} AFTER {clause}
            BEGIN
                INSERT INTO index_meta (key, value)
                VALUES ('{VECTOR_INDEX_STATE_KEY}', 'pending')
                ON CONFLICT(key) DO UPDATE SET value = 'pending';
                DELETE FROM query_cache;
            END
        """)
    conn.commit()

    conn.execute(
        f"SELECT vector_init('chunks', 'embedding', 'type=FLOAT32,dimension={EMBEDDING_DIM},distance=COSINE')"
    )
    conn.commit()

    return conn


def connect_readonly() -> sqlite3.Connection:
    """Connect to database in read-only mode."""
    db_path = config.get_db_path()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.enable_load_extension(True)
    conn.load_extension(str(VECTOR_EXT))
    conn.enable_load_extension(False)
    return conn


def connect_writable() -> sqlite3.Connection:
    """Open an existing database for routine writes without schema changes."""
    conn = sqlite3.connect(
        f"{config.get_db_path().resolve().as_uri()}?mode=rw", uri=True
    )
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def quantize_vectors(conn: sqlite3.Connection) -> None:
    """Quantize vectors for fast native search."""
    expected_bytes = EMBEDDING_DIM * 4
    mismatched = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL AND LENGTH(embedding) != ?",
        (expected_bytes,),
    ).fetchone()[0]
    if mismatched:
        raise RuntimeError(
            "found embeddings with unexpected dimension: "
            f"{mismatched} row(s) are not {EMBEDDING_DIM}-dim float32 vectors"
        )
    try:
        conn.execute("SELECT vector_quantize('chunks', 'embedding')")
        conn.execute("DELETE FROM query_cache")
        set_meta(conn, VECTOR_INDEX_STATE_KEY, "ready")
        conn.commit()
    except sqlite3.OperationalError as e:
        raise RuntimeError(f"vector quantization failed: {e}") from e


def embedding_to_blob(emb: list[float]) -> bytes:
    """Convert embedding list to binary blob."""
    if len(emb) != EMBEDDING_DIM:
        raise ValueError(
            f"embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(emb)}"
        )
    for v in emb:
        if not math.isfinite(v):
            raise ValueError("embedding contains non-finite value")
    return struct.pack(f"{len(emb)}f", *emb)


def blob_to_embedding(blob: bytes) -> list[float]:
    """Convert binary blob to embedding list."""
    if len(blob) % 4 != 0:
        raise ValueError("invalid embedding blob length")
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    """Read metadata value by key."""
    row = conn.execute("SELECT value FROM index_meta WHERE key = ?", (key,)).fetchone()
    if not row:
        return None
    value = row[0]
    return value if isinstance(value, str) else str(value)


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Upsert metadata key/value."""
    conn.execute(
        """
        INSERT INTO index_meta (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            updated_at = CURRENT_TIMESTAMP
        """,
        (key, value),
    )


@contextmanager
def get_connection(readonly: bool = False) -> Generator[sqlite3.Connection]:
    """Context manager for database connections."""
    conn = connect_readonly() if readonly else connect_writable()
    try:
        yield conn
    finally:
        conn.close()


def get_doc_status(
    conn,
    name: str,
    *,
    context_signature: str | None = None,
    embedding_signature: str | None = None,
) -> dict:
    """Get indexing status, including durable progress from interrupted runs."""
    empty = {
        "extracted": False,
        "embedded": 0,
        "complete": False,
        "chunks": 0,
        "expected": None,
        "text_size": 0,
        "embed_size": 0,
        "contextualized": 0,
    }
    row = conn.execute(
        "SELECT id, expected_chunks FROM documents WHERE name = ?", (name,)
    ).fetchone()
    if not row:
        return empty

    doc_id, expected = row
    row = conn.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(LENGTH(text)), 0), COALESCE(SUM(LENGTH(embedding)), 0)
        FROM chunks WHERE doc_id = ?
    """,
        (doc_id,),
    ).fetchone()
    chunk_count, text_size, embed_size = row

    chunk_columns = _column_names(conn, "chunks")
    embed_sig = embedding_signature or get_meta(conn, "pipeline.embedding.signature")
    if "embedding_signature" in chunk_columns and embed_sig:
        embedded = conn.execute(
            """
            SELECT COUNT(*) FROM chunks
            WHERE doc_id = ? AND embedding IS NOT NULL AND embedding_signature = ?
            """,
            (doc_id, embed_sig),
        ).fetchone()[0]
    else:
        embedded = 0

    try:
        context_columns = _column_names(conn, "chunk_contexts")
        context_sig = context_signature or get_meta(conn, "pipeline.context.signature")
        if "pipeline_signature" in context_columns and context_sig:
            contextualized = conn.execute(
                """
                SELECT COUNT(*) FROM chunk_contexts cc
                JOIN chunks c ON cc.chunk_id = c.id
                WHERE c.doc_id = ? AND cc.pipeline_signature = ?
                """,
                (doc_id, context_sig),
            ).fetchone()[0]
        else:
            contextualized = 0
    except sqlite3.Error:
        contextualized = 0

    complete = (
        expected is not None
        and chunk_count == expected
        and contextualized == expected
        and embedded == expected
    )

    return {
        "extracted": True,
        "embedded": embedded,
        "complete": complete,
        "chunks": chunk_count,
        "expected": expected,
        "text_size": text_size,
        "embed_size": embed_size,
        "contextualized": contextualized,
    }
