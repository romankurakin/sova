"""Search interface for benchmarks.

UPDATE THIS FILE when refactoring sova.
"""

from dataclasses import dataclass


@dataclass
class SearchResult:
    chunk_id: int
    doc: str
    text: str
    score: float
    section_id: int | None = None


class SovaBackend:
    """Adapter for sova. Update when refactoring."""

    def __init__(self):
        from sova.config import DB_PATH

        if not DB_PATH.exists():
            raise FileNotFoundError("No database. Run sova indexing first.")

        from sova.db import connect_readonly

        self.conn = connect_readonly()

    def close(self):
        self.conn.close()

    def _embed_query(self, text: str) -> list[float]:
        from sova.ollama_client import get_query_embedding

        return get_query_embedding(text)

    def search(
        self, query: str, limit: int = 10, embedding: list[float] | None = None
    ) -> list[SearchResult]:
        from sova.search import hybrid_search

        if embedding is None:
            embedding = self._embed_query(query)
        hits, _, _ = hybrid_search(self.conn, embedding, query, limit)

        return [
            SearchResult(
                chunk_id=h["chunk_id"],
                doc=h["doc"],
                text=h["text"],
                score=h.get("final_score", h.get("embed_score", 0)),
                section_id=h.get("section_id"),
            )
            for h in hits
        ]


_backend: SovaBackend | None = None


def get_backend() -> SovaBackend:
    global _backend
    if _backend is None:
        _backend = SovaBackend()
    return _backend


def close_backend():
    global _backend
    if _backend is not None:
        _backend.close()
        _backend = None


def measure_latency(queries: list[str]) -> dict:
    """Returns embed_times, search_times, total_times (ms)."""
    import time

    backend = get_backend()
    embed_times, search_times, total_times = [], [], []

    for q in queries:
        t0 = time.perf_counter()
        emb = backend._embed_query(q)
        t1 = time.perf_counter()
        backend.search(q, limit=10, embedding=emb)
        t2 = time.perf_counter()

        embed_times.append((t1 - t0) * 1000)
        search_times.append((t2 - t1) * 1000)
        total_times.append((t2 - t0) * 1000)

    return {
        "embed_times": embed_times,
        "search_times": search_times,
        "total_times": total_times,
    }


def clear_cache():
    from sova.cache import get_cache

    get_cache().clear()
