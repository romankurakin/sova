"""Tests for db module."""

import libsql_experimental as libsql
import struct

from sova.db import get_doc_status


class TestGetDocStatus:
    @staticmethod
    def _make_db():
        conn = libsql.connect(":memory:")  # ty: ignore[unresolved-attribute]
        conn.executescript("""
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL,
                path TEXT NOT NULL, line_count INTEGER, expected_chunks INTEGER
            );
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL,
                section_id INTEGER, start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL, word_count INTEGER NOT NULL,
                text TEXT NOT NULL, embedding BLOB
            );
        """)
        return conn

    def test_missing_document(self):
        conn = self._make_db()
        status = get_doc_status(conn, "nonexistent")
        assert status["extracted"] is False
        assert status["embedded"] is False
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
        assert status["embedded"] is False
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
        conn.commit()

        status = get_doc_status(conn, "doc1")
        assert status["extracted"] is True
        assert status["embedded"] is True
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
        assert status["embedded"] is True  # at least one has embedding
        assert status["complete"] is False  # 2 chunks < 3 expected
        assert status["chunks"] == 2
        conn.close()


class TestChunkContextsTable:
    @staticmethod
    def _make_db():
        conn = libsql.connect(":memory:")  # ty: ignore[unresolved-attribute]
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
        try:
            conn.execute(
                "INSERT INTO chunk_contexts (chunk_id, context, model) VALUES (1, 'ctx2', 'gemma3:12b')"
            )
            assert False, "Should have raised IntegrityError"
        except Exception:
            pass
        conn.close()

    def test_missing_context_returns_none(self):
        conn = self._make_db()
        row = conn.execute(
            "SELECT context FROM chunk_contexts WHERE chunk_id = 999"
        ).fetchone()
        assert row is None
        conn.close()
