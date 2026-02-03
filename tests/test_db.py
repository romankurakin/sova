"""Tests for db module."""

import sqlite3
import struct

from sova.db import blob_to_embedding, embedding_to_blob, get_doc_status


class TestEmbeddingToBlob:
    def test_roundtrip(self):
        emb = [0.1, 0.2, 0.3, -0.5, 0.0]
        blob = embedding_to_blob(emb)
        result = blob_to_embedding(blob)
        assert len(result) == len(emb)
        for a, b in zip(emb, result):
            assert abs(a - b) < 1e-6

    def test_empty(self):
        blob = embedding_to_blob([])
        assert blob == b""
        assert blob_to_embedding(b"") == []

    def test_single_value(self):
        emb = [1.0]
        blob = embedding_to_blob(emb)
        assert len(blob) == 4  # one float32
        assert blob_to_embedding(blob) == [1.0]

    def test_blob_is_float32(self):
        emb = [1.0, 2.0]
        blob = embedding_to_blob(emb)
        assert len(blob) == 8  # two float32s
        assert blob == struct.pack("2f", 1.0, 2.0)


class TestGetDocStatus:
    @staticmethod
    def _make_db() -> sqlite3.Connection:
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
        fake_emb = embedding_to_blob([0.1, 0.2])
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
        fake_emb = embedding_to_blob([0.1])
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
