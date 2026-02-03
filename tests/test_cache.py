"""Tests for cache module."""

import sqlite3
import time
from unittest.mock import patch

import numpy as np
import pytest

from sova.cache import SemanticCache, _cosine_sim_batch


class TestCosineSimBatch:
    def test_identical_vectors(self):
        query = np.array([1.0, 0.0, 0.0])
        embeddings = np.array([[1.0, 0.0, 0.0]])
        sims = _cosine_sim_batch(query, embeddings)
        assert sims[0] == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        query = np.array([1.0, 0.0])
        embeddings = np.array([[0.0, 1.0]])
        sims = _cosine_sim_batch(query, embeddings)
        assert sims[0] == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        query = np.array([1.0, 0.0])
        embeddings = np.array([[-1.0, 0.0]])
        sims = _cosine_sim_batch(query, embeddings)
        assert sims[0] == pytest.approx(-1.0, abs=1e-6)

    def test_batch_ordering(self):
        query = np.array([1.0, 0.0])
        embeddings = np.array(
            [
                [0.0, 1.0],  # orthogonal
                [1.0, 0.0],  # identical
                [0.5, 0.5],  # partial
            ]
        )
        sims = _cosine_sim_batch(query, embeddings)
        assert sims[1] > sims[2] > sims[0]

    def test_unnormalized_vectors(self):
        query = np.array([3.0, 0.0])
        embeddings = np.array([[5.0, 0.0]])
        sims = _cosine_sim_batch(query, embeddings)
        assert sims[0] == pytest.approx(1.0, abs=1e-6)


def _make_cache_db() -> sqlite3.Connection:
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
    with patch("sova.cache.get_connection") as mock_gc:
        mock_gc.return_value.__enter__ = lambda s: conn
        mock_gc.return_value.__exit__ = lambda s, *a: None
        cache = SemanticCache(threshold=0.92, max_size=3, ttl=3600)
        yield cache, conn
    conn.close()


class TestSemanticCache:
    def test_miss_on_empty_cache(self, cache_db):
        cache, _ = cache_db
        result = cache.get([1.0, 0.0, 0.0])
        assert result is None

    def test_put_then_get_identical(self, cache_db):
        cache, _ = cache_db
        emb = [1.0, 0.0, 0.0]
        expected = [(1, 0.9), (2, 0.8)]
        cache.put(emb, expected)
        result = cache.get(emb)
        # msgpack deserializes tuples as lists
        assert [list(t) for t in expected] == result

    def test_miss_on_dissimilar_query(self, cache_db):
        cache, _ = cache_db
        cache.put([1.0, 0.0, 0.0], [(1, 0.9)])
        # Orthogonal vector — well below 0.92 threshold
        result = cache.get([0.0, 1.0, 0.0])
        assert result is None

    def test_hit_on_similar_query(self, cache_db):
        cache, _ = cache_db
        cache.put([1.0, 0.0, 0.0], [(1, 0.9)])
        # Very close vector — should exceed 0.92 threshold
        result = cache.get([0.99, 0.01, 0.0])
        assert result is not None

    def test_clear(self, cache_db):
        cache, _ = cache_db
        cache.put([1.0, 0.0], [(1, 0.9)])
        cache.clear()
        result = cache.get([1.0, 0.0])
        assert result is None

    def test_lru_eviction(self, cache_db):
        cache, conn = cache_db
        # max_size=3, insert 4 entries
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
        # Backdate the entry beyond TTL
        conn.execute("UPDATE query_cache SET created_at = ?", (time.time() - 7200,))
        conn.commit()

        result = cache.get(emb)
        assert result is None

    def test_min_candidates_filter(self, cache_db):
        cache, _ = cache_db
        emb = [1.0, 0.0, 0.0]
        cache.put(emb, [(1, 0.9), (2, 0.8)])  # 2 candidates

        # Requesting at least 2 — should hit
        assert cache.get(emb, min_candidates=2) is not None
        # Requesting at least 5 — should miss
        assert cache.get(emb, min_candidates=5) is None
