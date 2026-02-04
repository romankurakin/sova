"""Semantic cache for vector search candidates."""

import time

import msgpack
import numpy as np

from sova.config import EMBEDDING_MODEL
from sova.db import get_connection


def _cosine_sim_batch(query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarities between query and embedding matrix."""
    # 1e-9 avoids division by zero for degenerate zero-norm embeddings.
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query) + 1e-9
    return embeddings @ query / norms


class SemanticCache:
    """Cache vector search candidates by query embedding similarity."""

    def __init__(
        self,
        # 0.92 cosine sim means queries must be nearly identical in meaning.
        # Lower values give more cache hits but risk returning stale results
        # for semantically different queries.
        threshold: float = 0.92,
        max_size: int = 500,
        ttl: int = 3600,
    ):
        self.threshold = threshold
        self.max_size = max_size
        self.ttl = ttl

    def get(
        self, query_emb: list[float], min_candidates: int = 0
    ) -> list[tuple[int, float]] | None:
        """Get cached vector results if similar query exists."""
        now = time.time()
        cutoff = now - self.ttl
        query_arr = np.array(query_emb, dtype=np.float32)

        with get_connection(readonly=True) as conn:
            rows = conn.execute(
                """SELECT id, embedding, vector_results
                   FROM query_cache
                   WHERE created_at > ? AND model = ? AND candidate_count >= ?""",
                (cutoff, EMBEDDING_MODEL, min_candidates),
            ).fetchall()

        if not rows:
            return None

        row_ids = []
        embeddings = []
        results_blobs = []
        for row_id, emb_blob, results_blob in rows:
            row_ids.append(row_id)
            embeddings.append(np.frombuffer(emb_blob, dtype=np.float32))
            results_blobs.append(results_blob)

        if not embeddings:
            return None

        similarities = _cosine_sim_batch(query_arr, np.array(embeddings))

        best_idx = int(np.argmax(similarities))
        if similarities[best_idx] >= self.threshold:
            row_id = row_ids[best_idx]
            # Touch created_at so this entry survives LRU eviction longer.
            with get_connection() as conn:
                conn.execute(
                    "UPDATE query_cache SET created_at = ? WHERE id = ?", (now, row_id)
                )
                conn.commit()
            return msgpack.unpackb(results_blobs[best_idx])

        return None

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


_cache: SemanticCache | None = None


def get_cache() -> SemanticCache:
    """Get or create global cache."""
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache
