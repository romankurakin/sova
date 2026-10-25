"""Exact cache for vector search candidates."""

import sqlite3
import time

import msgpack
import numpy as np

from sova.config import EMBEDDING_MODEL
from sova.db import get_connection


class VectorCache:
    """Cache candidates for identical serialized query vectors and search depths."""

    def __init__(
        self,
        max_size: int = 500,
        ttl: int = 3600,
    ):
        self.max_size = max_size
        self.ttl = ttl

    def get(
        self, query_emb: list[float], candidate_count: int | None = None
    ) -> list[tuple[int, float]] | None:
        """Reuse only the exact native-search input; depth changes can alter fusion."""
        now = time.time()
        query_blob = np.array(query_emb, dtype=np.float32).tobytes()
        with get_connection(readonly=True) as conn:
            row = conn.execute(
                """SELECT id, vector_results FROM query_cache
                   WHERE created_at > ? AND model = ? AND embedding = ?
                     AND (? IS NULL OR candidate_count = ?)
                   ORDER BY created_at DESC, id DESC LIMIT 1""",
                (
                    now - self.ttl,
                    EMBEDDING_MODEL,
                    query_blob,
                    candidate_count,
                    candidate_count,
                ),
            ).fetchone()
        if row is None:
            return None
        row_id, results_blob = row
        # LRU bookkeeping is optional: a read-only index can still serve a hit.
        try:
            with get_connection() as conn:
                conn.execute(
                    "UPDATE query_cache SET created_at = ? WHERE id = ?", (now, row_id)
                )
                conn.commit()
        except sqlite3.OperationalError:
            pass
        return [tuple(result) for result in msgpack.unpackb(results_blob)]

    def put(
        self, query_emb: list[float], vector_results: list[tuple[int, float]]
    ) -> None:
        """Cache vector results for query embedding."""
        with get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
            if count >= self.max_size:
                conn.execute(
                    """DELETE FROM query_cache WHERE id = (
                        SELECT id FROM query_cache ORDER BY created_at ASC LIMIT 1
                    )"""
                )

            conn.execute(
                """INSERT INTO query_cache
                   (embedding, vector_results, created_at, model, candidate_count)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    np.array(query_emb, dtype=np.float32).tobytes(),
                    msgpack.packb(vector_results),
                    time.time(),
                    EMBEDDING_MODEL,
                    len(vector_results),
                ),
            )
            conn.commit()

    def clear(self) -> None:
        """Clear all cached entries."""
        with get_connection() as conn:
            conn.execute("DELETE FROM query_cache")
            conn.commit()


_cache: VectorCache | None = None


def get_cache() -> VectorCache:
    """Get or create global cache."""
    global _cache
    if _cache is None:
        _cache = VectorCache()
    return _cache
