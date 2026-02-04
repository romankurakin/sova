"""Database initialization and connection management."""

import libsql_experimental as libsql
from contextlib import contextmanager
from typing import Any, Generator

from sova.config import DATA_DIR, DB_PATH, EMBEDDING_DIM


def init_db() -> Any:
    """Initialize database with tables and indexes."""
    DATA_DIR.mkdir(exist_ok=True)
    conn = libsql.connect(str(DB_PATH))  # ty: ignore[unresolved-attribute]

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL,
            path TEXT NOT NULL, line_count INTEGER, expected_chunks INTEGER,
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
            word_count INTEGER NOT NULL, text TEXT NOT NULL,
            embedding F32_BLOB({dim}), is_index INTEGER NOT NULL DEFAULT 0,
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
            FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
        CREATE INDEX IF NOT EXISTS idx_query_cache_created ON query_cache(created_at);
        PRAGMA foreign_keys = ON;
    """.format(dim=EMBEDDING_DIM)
    )

    conn.executescript("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            content='chunks',
            content_rowid='id',
            tokenize='porter unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
            INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
        END;
    """)

    # FTS triggers only fire on new inserts. If chunks were added before FTS
    # was set up (e.g. schema migration), backfill the index from existing rows.
    fts_count = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if fts_count == 0 and chunk_count > 0:
        conn.execute("INSERT INTO chunks_fts(rowid, text) SELECT id, text FROM chunks")
        conn.commit()

    conn.execute(
        "CREATE INDEX IF NOT EXISTS chunks_vec_idx ON chunks("
        "libsql_vector_idx(embedding, 'metric=cosine', 'compress_neighbors=float8'))"
    )
    conn.commit()

    return conn


def connect_readonly() -> Any:
    """Connect to database in read-only mode."""
    return libsql.connect(str(DB_PATH))  # ty: ignore[unresolved-attribute]


@contextmanager
def get_connection(readonly: bool = False) -> Generator[Any, None, None]:
    """Context manager for database connections."""
    conn = connect_readonly() if readonly else init_db()
    try:
        yield conn
    finally:
        conn.close()


def get_doc_status(conn, name: str) -> dict:
    """Get indexing status for a document."""
    empty = {
        "extracted": False,
        "embedded": False,
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

    embedded = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE doc_id = ? AND embedding IS NOT NULL",
        (doc_id,),
    ).fetchone()[0]

    try:
        contextualized = conn.execute(
            """
            SELECT COUNT(*) FROM chunk_contexts cc
            JOIN chunks c ON cc.chunk_id = c.id
            WHERE c.doc_id = ?
        """,
            (doc_id,),
        ).fetchone()[0]
    except Exception:
        contextualized = 0

    complete = expected is not None and chunk_count >= expected

    return {
        "extracted": True,
        "embedded": embedded > 0,
        "complete": complete,
        "chunks": chunk_count,
        "expected": expected,
        "text_size": text_size,
        "embed_size": embed_size,
        "contextualized": contextualized,
    }
