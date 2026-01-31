"""End-to-end tests for search functionality."""

import sqlite3
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_db():
    """Create a temporary database with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        conn = sqlite3.connect(db_path)

        conn.executescript("""
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                path TEXT NOT NULL,
                line_count INTEGER,
                expected_chunks INTEGER
            );
            CREATE TABLE sections (
                id INTEGER PRIMARY KEY,
                doc_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                level INTEGER NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER
            );
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,
                doc_id INTEGER NOT NULL,
                section_id INTEGER,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                word_count INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB
            );
            CREATE INDEX idx_chunks_doc ON chunks(doc_id);

            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                text,
                content='chunks',
                content_rowid='id',
                tokenize='porter unicode61'
            );
        """)

        # Insert test document
        conn.execute(
            "INSERT INTO documents (name, path, line_count) VALUES (?, ?, ?)",
            ("test_doc", "/tmp/test.md", 100),
        )

        # Insert test chunks
        test_chunks = [
            (1, 1, None, 1, 10, 50, "Memory management and virtual addressing in operating systems"),
            (2, 1, None, 11, 20, 45, "Process scheduling algorithms including round robin"),
            (3, 1, None, 21, 30, 55, "File system implementation and disk management"),
            (4, 1, None, 31, 40, 40, "Network protocols TCP IP and socket programming"),
            (5, 1, None, 41, 50, 35, "Table of contents 1.1 ... 5 1.2 ... 10 1.3 ... 15"),
        ]

        for chunk in test_chunks:
            conn.execute(
                "INSERT INTO chunks (id, doc_id, section_id, start_line, end_line, word_count, text) VALUES (?, ?, ?, ?, ?, ?, ?)",
                chunk,
            )
            conn.execute(
                "INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)",
                (chunk[0], chunk[6]),
            )

        conn.commit()
        yield conn, db_path
        conn.close()


class TestFTSSearch:
    def test_fts_exact_match(self, temp_db):
        from sova.search import search_fts

        conn, _ = temp_db
        results = search_fts(conn, "memory management", 5)
        assert len(results) > 0
        # First chunk should match
        chunk_ids = [r[0] for r in results]
        assert 1 in chunk_ids

    def test_fts_no_match(self, temp_db):
        from sova.search import search_fts

        conn, _ = temp_db
        results = search_fts(conn, "quantum computing blockchain", 5)
        assert len(results) == 0

    def test_fts_partial_match(self, temp_db):
        from sova.search import search_fts

        conn, _ = temp_db
        results = search_fts(conn, "scheduling", 5)
        assert len(results) > 0


class TestRRFIntegration:
    def test_fusion_combines_results(self):
        from sova.search import rrf_fusion

        vector_results = [(1, 0.9), (2, 0.8), (3, 0.7)]
        fts_results = [(2, 5.0), (4, 4.0), (1, 3.0)]

        scores = rrf_fusion([vector_results, fts_results])

        # Item 2 appears high in both lists
        assert scores[2] > scores[3]  # 3 only in vector
        assert scores[2] > scores[4]  # 4 only in fts


class TestIndexDetection:
    def test_toc_detected(self, temp_db):
        from sova.search import is_index_like

        conn, _ = temp_db
        toc_text = conn.execute(
            "SELECT text FROM chunks WHERE id = 5"
        ).fetchone()[0]
        assert is_index_like(toc_text) is True

    def test_content_not_flagged(self, temp_db):
        from sova.search import is_index_like

        conn, _ = temp_db
        content_text = conn.execute(
            "SELECT text FROM chunks WHERE id = 1"
        ).fetchone()[0]
        assert is_index_like(content_text) is False
