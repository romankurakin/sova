"""Tests for db module."""

import sqlite3
import struct

import pytest

from sova.config import EMBEDDING_DIM
from sova.db import embedding_to_blob, get_doc_status, get_meta, init_db, set_meta


def test_init_db_migrates_raw_fts_to_contextual_search_text(monkeypatch, tmp_path):
    from sova import config

    db_path = tmp_path / "indexed.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, path TEXT NOT NULL,
            line_count INTEGER, expected_chunks INTEGER, source_signature TEXT
        );
        CREATE TABLE sections (
            id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL, title TEXT NOT NULL,
            level INTEGER NOT NULL, start_line INTEGER NOT NULL, end_line INTEGER
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL, section_id INTEGER,
            start_line INTEGER NOT NULL, end_line INTEGER NOT NULL,
            word_count INTEGER NOT NULL, text TEXT NOT NULL, embedding BLOB,
            embedding_signature TEXT, is_index INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE chunk_contexts (
            chunk_id INTEGER PRIMARY KEY, context TEXT NOT NULL, model TEXT NOT NULL,
            pipeline_signature TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE query_cache (
            id INTEGER PRIMARY KEY, embedding BLOB NOT NULL,
            vector_results BLOB NOT NULL, created_at REAL NOT NULL,
            model TEXT NOT NULL, candidate_count INTEGER NOT NULL
        );
        CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            text, content='chunks', content_rowid='id', tokenize='porter unicode61'
        );
        INSERT INTO documents VALUES (1, 'manual', '/tmp/manual.md', 1, 1, 'sig');
        INSERT INTO chunks
            (id, doc_id, start_line, end_line, word_count, text)
        VALUES (1, 1, 1, 1, 2, 'raw source');
        INSERT INTO chunks_fts(rowid, text) VALUES (1, 'raw source');
        PRAGMA user_version = 3;
        """
    )
    conn.close()

    monkeypatch.setattr(config, "get_data_dir", lambda: tmp_path)
    monkeypatch.setattr(config, "get_db_path", lambda: db_path)
    migrated = init_db()

    chunk_columns = {
        row[1] for row in migrated.execute("PRAGMA table_xinfo(chunks)").fetchall()
    }
    fts_columns = {
        row[1] for row in migrated.execute("PRAGMA table_info(chunks_fts)").fetchall()
    }
    assert {"section_path", "search_text"} <= chunk_columns
    assert "search_text" in fts_columns
    assert migrated.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'raw'"
    ).fetchone() == (1,)

    migrated.execute(
        "UPDATE chunks SET search_prefix = 'context-only sentinel ' WHERE id = 1"
    )
    assert migrated.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'sentinel'"
    ).fetchone() == (1,)
    migrated.close()


class TestGetDocStatus:
    @staticmethod
    def _make_db():
        conn = sqlite3.connect(":memory:")
        conn.executescript("""
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL,
                path TEXT NOT NULL, line_count INTEGER, expected_chunks INTEGER
            );
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL,
                section_id INTEGER, start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL, word_count INTEGER NOT NULL,
                text TEXT NOT NULL, embedding BLOB,
                embedding_signature TEXT DEFAULT 'embed-v1'
            );
            CREATE TABLE chunk_contexts (
                chunk_id INTEGER PRIMARY KEY,
                context TEXT NOT NULL,
                model TEXT NOT NULL,
                pipeline_signature TEXT NOT NULL DEFAULT 'context-v1'
            );
            CREATE TABLE index_meta (
                key TEXT PRIMARY KEY, value TEXT NOT NULL
            );
            INSERT INTO index_meta VALUES
                ('pipeline.context.signature', 'context-v1'),
                ('pipeline.embedding.signature', 'embed-v1');
        """)
        return conn

    def test_missing_document(self):
        conn = self._make_db()
        status = get_doc_status(conn, "nonexistent")
        assert status["extracted"] is False
        assert status["embedded"] == 0
        assert status["complete"] is False
        assert status["chunks"] == 0
        conn.close()

    def test_extracted_no_embeddings(self):
        conn = self._make_db()
        conn.execute(
            "INSERT INTO documents (name, path, expected_chunks) VALUES (?, ?, ?)",
            ("doc1", "/tmp/doc1.md", 2),
        )
        conn.execute(
            "INSERT INTO chunks (doc_id, start_line, end_line, word_count, text) VALUES (1, 1, 10, 50, 'hello world')",
        )
        conn.commit()

        status = get_doc_status(conn, "doc1")
        assert status["extracted"] is True
        assert status["embedded"] == 0
        assert status["complete"] is False
        assert status["chunks"] == 1
        assert status["expected"] == 2
        conn.close()

    def test_fully_embedded(self):
        conn = self._make_db()
        conn.execute(
            "INSERT INTO documents (name, path, expected_chunks) VALUES (?, ?, ?)",
            ("doc1", "/tmp/doc1.md", 1),
        )
        fake_emb = struct.pack("2f", 0.1, 0.2)
        conn.execute(
            "INSERT INTO chunks (doc_id, start_line, end_line, word_count, text, embedding) VALUES (1, 1, 10, 50, 'hello', ?)",
            (fake_emb,),
        )
        conn.execute(
            """
            INSERT INTO chunk_contexts (chunk_id, context, model)
            VALUES (1, 'Authentication rules.', 'qwen3.8-27b')
            """
        )
        conn.commit()

        status = get_doc_status(conn, "doc1")
        assert status["extracted"] is True
        assert status["embedded"] == 1
        assert status["complete"] is True
        assert status["chunks"] == 1
        assert status["text_size"] > 0
        assert status["embed_size"] > 0
        conn.close()

    def test_partial_embedding(self):
        conn = self._make_db()
        conn.execute(
            "INSERT INTO documents (name, path, expected_chunks) VALUES (?, ?, ?)",
            ("doc1", "/tmp/doc1.md", 3),
        )
        fake_emb = struct.pack("1f", 0.1)
        conn.execute(
            "INSERT INTO chunks (doc_id, start_line, end_line, word_count, text, embedding) VALUES (1, 1, 10, 50, 'a', ?)",
            (fake_emb,),
        )
        conn.execute(
            "INSERT INTO chunks (doc_id, start_line, end_line, word_count, text) VALUES (1, 11, 20, 40, 'b')",
        )
        conn.commit()

        status = get_doc_status(conn, "doc1")
        assert status["extracted"] is True
        assert status["embedded"] == 1  # one chunk has embedding.
        assert status["complete"] is False  # 2 chunks < 3 expected.
        assert status["chunks"] == 2
        conn.close()

    def test_explicit_signatures_report_saved_progress_before_finalization(self):
        conn = self._make_db()
        conn.execute("DELETE FROM index_meta")
        conn.execute(
            "INSERT INTO documents (name, path, expected_chunks) VALUES (?, ?, ?)",
            ("doc1", "/tmp/doc1.md", 2),
        )
        conn.executemany(
            """
            INSERT INTO chunks
                (doc_id, start_line, end_line, word_count, text, embedding, embedding_signature)
            VALUES (1, ?, ?, 20, ?, ?, ?)
            """,
            [
                (1, 10, "one", struct.pack("1f", 0.1), "current-embed"),
                (11, 20, "two", None, None),
            ],
        )
        conn.execute(
            """
            INSERT INTO chunk_contexts
                (chunk_id, context, model, pipeline_signature)
            VALUES (1, 'saved context', 'model', 'current-context')
            """
        )
        conn.commit()

        status = get_doc_status(
            conn,
            "doc1",
            context_signature="current-context",
            embedding_signature="current-embed",
        )

        assert status["contextualized"] == 1
        assert status["embedded"] == 1
        assert status["complete"] is False
        conn.close()


class TestChunkContextsTable:
    @staticmethod
    def _make_db():
        conn = sqlite3.connect(":memory:")
        conn.executescript("""
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL,
                path TEXT NOT NULL, line_count INTEGER, expected_chunks INTEGER
            );
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL,
                section_id INTEGER, start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL, word_count INTEGER NOT NULL,
                text TEXT NOT NULL, embedding BLOB,
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
            );
            CREATE TABLE chunk_contexts (
                chunk_id INTEGER PRIMARY KEY,
                context TEXT NOT NULL,
                model TEXT NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
            );
            PRAGMA foreign_keys = ON;
        """)
        return conn

    def test_insert_and_retrieve(self):
        conn = self._make_db()
        conn.execute(
            "INSERT INTO documents (name, path) VALUES ('doc1', '/tmp/doc1.md')"
        )
        conn.execute(
            "INSERT INTO chunks (doc_id, start_line, end_line, word_count, text) VALUES (1, 1, 10, 50, 'hello')"
        )
        conn.execute(
            "INSERT INTO chunk_contexts (chunk_id, context, model) VALUES (1, 'This covers auth.', 'gemma3:12b')"
        )
        conn.commit()

        row = conn.execute(
            "SELECT context, model FROM chunk_contexts WHERE chunk_id = 1"
        ).fetchone()
        assert row == ("This covers auth.", "gemma3:12b")
        conn.close()

    def test_one_context_per_chunk(self):
        conn = self._make_db()
        conn.execute(
            "INSERT INTO documents (name, path) VALUES ('doc1', '/tmp/doc1.md')"
        )
        conn.execute(
            "INSERT INTO chunks (doc_id, start_line, end_line, word_count, text) VALUES (1, 1, 10, 50, 'hello')"
        )
        conn.execute(
            "INSERT INTO chunk_contexts (chunk_id, context, model) VALUES (1, 'ctx', 'gemma3:12b')"
        )
        conn.commit()

        # Inserting a second context for the same chunk should fail (PK constraint).
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO chunk_contexts (chunk_id, context, model) VALUES (1, 'ctx2', 'gemma3:12b')"
            )
        conn.close()

    def test_missing_context_returns_none(self):
        conn = self._make_db()
        row = conn.execute(
            "SELECT context FROM chunk_contexts WHERE chunk_id = 999"
        ).fetchone()
        assert row is None
        conn.close()


class TestEmbeddingToBlob:
    def test_rejects_wrong_dimension(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            embedding_to_blob([0.1, 0.2])

    def test_accepts_expected_dimension(self):
        emb = [0.0] * EMBEDDING_DIM
        blob = embedding_to_blob(emb)
        assert isinstance(blob, bytes)
        assert len(blob) == EMBEDDING_DIM * 4


class TestIndexMeta:
    def test_round_trip_meta(self):
        conn = sqlite3.connect(":memory:")
        conn.execute(
            """
            CREATE TABLE index_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        set_meta(conn, "k1", "v1")
        conn.commit()
        assert get_meta(conn, "k1") == "v1"
        conn.close()


def _legacy_search_db(path):
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY, text TEXT NOT NULL, search_text TEXT NOT NULL,
            embedding BLOB, embedding_signature TEXT, section_path TEXT DEFAULT ''
        );
        CREATE TABLE chunk_contexts (
            chunk_id INTEGER PRIMARY KEY, context TEXT, pipeline_signature TEXT
        );
        CREATE TABLE documents (id INTEGER PRIMARY KEY, source_signature TEXT);
        CREATE TABLE query_cache (id INTEGER PRIMARY KEY, payload BLOB);
        CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO index_meta VALUES ('index.vector.state', 'ready');
        INSERT INTO query_cache VALUES (1, X'010203');
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            search_text, content='chunks', content_rowid='id',
            tokenize='porter unicode61'
        );
        CREATE TRIGGER chunks_au AFTER UPDATE OF search_text ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, search_text)
            VALUES('delete', old.id, old.search_text);
            INSERT INTO chunks_fts(rowid, search_text) VALUES(new.id, new.search_text);
        END;
        PRAGMA user_version = 6;
    """)
    return conn


def _fts_results(conn, term):
    return conn.execute(
        "SELECT rowid, bm25(chunks_fts) FROM chunks_fts "
        "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts), rowid",
        (term,),
    ).fetchall()


def test_dedup_preserves_inputs_scores_and_saved_work(tmp_path):
    from sova.db import SCHEMA_VERSION, _migrate_schema
    from sova.index_text import contextualized_text

    path = tmp_path / "legacy.db"
    conn = _legacy_search_db(path)
    # Every Python whitespace codepoint, including those SQLite trim omits.
    whitespace = "".join(chr(i) for i in range(0x110000) if chr(i).isspace())
    sources = [
        whitespace + "shared source" + whitespace,
        " raw shared ",
        "",
        whitespace,
        "shared\x00embedded null",
    ]
    inputs = [contextualized_text("manual", "Section", t, "sentinel") for t in sources]
    inputs[1] = sources[1]  # Legacy raw input must not be stripped.
    for i, (source, search) in enumerate(zip(sources, inputs), 1):
        conn.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
            (i, source, search, bytes([i]), "saved-embedding", "Section"),
        )
        conn.execute(
            "INSERT INTO chunk_contexts VALUES (?, ?, ?)",
            (i, "saved context", "saved-context"),
        )
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    conn.commit()
    before = _fts_results(conn, "shared OR sentinel")
    contexts = conn.execute("SELECT * FROM chunk_contexts").fetchall()
    _migrate_schema(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert conn.execute(
        "SELECT text, search_text FROM chunks ORDER BY id"
    ).fetchall() == list(zip(sources, inputs))
    assert _fts_results(conn, "shared OR sentinel") == before
    assert conn.execute("SELECT * FROM chunk_contexts").fetchall() == contexts
    assert conn.execute(
        "SELECT embedding, embedding_signature FROM chunks ORDER BY id"
    ).fetchall() == [(bytes([i]), "saved-embedding") for i in range(1, 6)]
    assert conn.execute("SELECT * FROM query_cache").fetchall() == [
        (1, b"\x01\x02\x03")
    ]
    assert conn.execute(
        "SELECT value FROM index_meta WHERE key = 'index.vector.state'"
    ).fetchone() == ("ready",)
    assert (
        next(
            r
            for r in conn.execute("PRAGMA table_xinfo(chunks)")
            if r[1] == "search_text"
        )[6]
        == 2
    )
    _migrate_schema(conn)  # Reopening is idempotent.
    conn.close()
    conn = sqlite3.connect(path)  # No application-defined SQL functions required.
    assert conn.execute("SELECT search_text FROM chunks ORDER BY id").fetchall() == [
        (s,) for s in inputs
    ]
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    assert _fts_results(conn, "shared OR sentinel") == before
    conn.execute(
        "INSERT INTO chunks_fts(chunks_fts, rank) VALUES('integrity-check', 1)"
    )
    conn.close()


def test_generated_search_fts_tracks_all_source_changes(tmp_path):
    from sova.db import _migrate_schema

    conn = _legacy_search_db(tmp_path / "legacy.db")
    _migrate_schema(conn)
    conn.execute(
        "INSERT INTO chunks (id, text, search_prefix) VALUES (1, ' original ', 'header ')"
    )
    assert _fts_results(conn, "original")
    conn.execute("UPDATE chunks SET search_prefix = 'contextual ' WHERE id = 1")
    assert not _fts_results(conn, "header")
    assert _fts_results(conn, "contextual")
    conn.execute("UPDATE chunks SET text = 'replacement' WHERE id = 1")
    assert not _fts_results(conn, "original")
    assert _fts_results(conn, "replacement")
    conn.execute("UPDATE chunks SET id = 2 WHERE id = 1")
    assert _fts_results(conn, "replacement")[0][0] == 2
    conn.execute("DELETE FROM chunks WHERE id = 2")
    assert not _fts_results(conn, "replacement")
    conn.execute(
        "INSERT INTO chunks_fts(chunks_fts, rank) VALUES('integrity-check', 1)"
    )
    conn.close()


def test_dedup_failure_rolls_back_schema_and_partial_work(tmp_path):
    from sova.db import _migrate_schema

    conn = _legacy_search_db(tmp_path / "legacy.db")
    # Exceed the migration batch size so some updates precede the rejected row.
    conn.executemany(
        "INSERT INTO chunks (id, text, search_text) VALUES (?, ?, ?)",
        [(i, "source", "header source") for i in range(1, 300)],
    )
    conn.execute(
        "INSERT INTO chunks (id, text, search_text) VALUES (300, 'source', 'unrelated')"
    )
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    conn.commit()
    schema = conn.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()
    before = _fts_results(conn, "source")
    with pytest.raises(RuntimeError, match="chunk 300"):
        _migrate_schema(conn)
    assert conn.execute("PRAGMA user_version").fetchone() == (6,)
    assert (
        conn.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall() == schema
    )
    assert _fts_results(conn, "source") == before
    assert not conn.in_transaction
    conn.close()
