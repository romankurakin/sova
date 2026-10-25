"""Tests for cache module."""

import sqlite3
import time
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from sova.cache import VectorCache
from sova.search import compute_candidates


def _make_cache_db():
    """Create an in-memory DB with the query_cache table."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE query_cache (
            id INTEGER PRIMARY KEY,
            embedding BLOB NOT NULL,
            vector_results BLOB NOT NULL,
            created_at REAL NOT NULL,
            model TEXT NOT NULL,
            candidate_count INTEGER NOT NULL
        )
    """)
    conn.execute("CREATE INDEX idx_query_cache_created ON query_cache(created_at)")
    conn.commit()
    return conn


@pytest.fixture
def cache_db():
    """Provide a cache instance backed by an in-memory DB."""
    conn = _make_cache_db()
    cache = VectorCache(max_size=3, ttl=3600)

    @contextmanager
    def _fake_connection(readonly=False):
        yield conn

    with patch("sova.cache.get_connection", _fake_connection):
        yield cache, conn
    conn.close()


class TestVectorCache:
    def test_miss_on_empty_cache(self, cache_db):
        cache, _conn = cache_db
        result = cache.get([1.0, 0.0, 0.0])
        assert result is None

    def test_put_then_get_identical(self, cache_db):
        cache, _conn = cache_db
        emb = [1.0, 0.0, 0.0]
        expected = [(1, 0.9), (2, 0.8)]
        cache.put(emb, expected)
        result = cache.get(emb)
        assert expected == result

    def test_miss_on_dissimilar_query(self, cache_db):
        cache, _conn = cache_db
        cache.put([1.0, 0.0, 0.0], [(1, 0.9)])
        # A different vector cannot reuse the candidate list.
        result = cache.get([0.0, 1.0, 0.0])
        assert result is None

    def test_miss_on_similar_query(self, cache_db):
        cache, _conn = cache_db
        cache.put([1.0, 0.0, 0.0], [(1, 0.9)])
        # Similarity must not substitute candidates for another query.
        result = cache.get([0.99, 0.01, 0.0])
        assert result is None

    def test_clear(self, cache_db):
        cache, conn = cache_db
        cache.put([1.0, 0.0], [(1, 0.9)])
        conn.execute("DELETE FROM query_cache")
        conn.commit()
        result = cache.get([1.0, 0.0])
        assert result is None

    def test_lru_eviction(self, cache_db):
        cache, conn = cache_db
        # max_size=3, insert 4 entries.
        cache.put([1.0, 0.0], [(1, 0.9)])
        cache.put([0.0, 1.0], [(2, 0.8)])
        cache.put([0.5, 0.5], [(3, 0.7)])
        cache.put([0.7, 0.7], [(4, 0.6)])

        count = conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
        assert count <= 3

    def test_ttl_expiry(self, cache_db):
        cache, conn = cache_db
        emb = [1.0, 0.0, 0.0]
        cache.put(emb, [(1, 0.9)])
        # Backdate the entry beyond TTL.
        conn.execute("UPDATE query_cache SET created_at = ?", (time.time() - 7200,))
        conn.commit()

        result = cache.get(emb)
        assert result is None

    def test_candidate_depth_is_part_of_cache_identity(self, cache_db):
        cache, _conn = cache_db
        emb = [1.0, 0.0, 0.0]
        cache.put(emb, [(1, 0.9), (2, 0.8)])  # 2 candidates.

        # Only an equal depth may hit.
        assert cache.get(emb, candidate_count=2) is not None
        assert cache.get(emb, candidate_count=1) is None
        # A shallower entry cannot serve a deeper search either.
        assert cache.get(emb, candidate_count=5) is None

    def test_identical_query_hits_when_corpus_is_smaller_than_candidate_floor(
        self, cache_db
    ):
        cache, _conn = cache_db
        emb = [1.0, 0.0]
        results = [(i, 0.9) for i in range(20)]
        cache.put(emb, results)
        assert cache.get(emb, compute_candidates(20, 10)) == results


def test_cache_operations_preserve_database_schema(monkeypatch, tmp_path):
    from sova import config
    from sova.db import init_db

    db_path = tmp_path / "indexed.db"
    monkeypatch.setattr(config, "get_data_dir", lambda: tmp_path)
    monkeypatch.setattr(config, "get_db_path", lambda: db_path)
    conn = init_db()
    schema_version = conn.execute("PRAGMA schema_version").fetchone()[0]
    schema = conn.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()
    cache = VectorCache()
    cache.put([1.0, 0.0], [(1, 0.9)])
    assert cache.get([1.0, 0.0]) == [(1, 0.9)]
    cache.clear()
    assert conn.execute("PRAGMA schema_version").fetchone()[0] == schema_version
    assert (
        conn.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall() == schema
    )
    conn.close()


def test_cache_miss_for_different_dimension(cache_db):
    cache, _ = cache_db
    cache.put([1.0, 0.0], [(1, 0.9)])
    assert cache.get([1.0, 0.0, 0.0]) is None


def test_cache_miss_for_different_model(cache_db):
    cache, conn = cache_db
    cache.put([1.0, 0.0], [(1, 0.9)])
    conn.execute("UPDATE query_cache SET model = 'different-model'")
    conn.commit()
    assert cache.get([1.0, 0.0]) is None


def test_cache_hit_survives_readonly_storage(cache_db):
    cache, conn = cache_db
    cache.put([1.0, 0.0], [(1, 0.9)])
    conn.execute("PRAGMA query_only=ON")
    assert cache.get([1.0, 0.0]) == [(1, 0.9)]
