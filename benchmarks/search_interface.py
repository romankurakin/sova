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
        from sova.llama_client import get_query_embedding

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

    def search_fts_only(self, query: str, limit: int = 20) -> list[SearchResult]:
        """BM25-only search via FTS5."""
        from sova.search import search_fts

        fts_results = search_fts(self.conn, query, limit)
        return self._hydrate_chunks(fts_results)

    def search_vector_only(self, query: str, limit: int = 20) -> list[SearchResult]:
        """Vector-only search."""
        from sova.search import get_vector_candidates

        embedding = self._embed_query(query)
        vector_results = get_vector_candidates(self.conn, embedding, limit)
        return self._hydrate_chunks(vector_results[:limit])

    def _hydrate_chunks(
        self, chunk_id_score_pairs: list[tuple[int, float]]
    ) -> list[SearchResult]:
        """Fetch text + doc for a list of (chunk_id, score) pairs."""
        if not chunk_id_score_pairs:
            return []

        ids = [cid for cid, _ in chunk_id_score_pairs]
        score_map = {cid: score for cid, score in chunk_id_score_pairs}
        placeholders = ",".join("?" * len(ids))

        rows = self.conn.execute(
            f"SELECT c.id, d.name, c.text, c.section_id"
            f" FROM chunks c JOIN documents d ON c.doc_id = d.id"
            f" WHERE c.id IN ({placeholders})",
            tuple(ids),
        ).fetchall()

        row_map = {r[0]: r for r in rows}
        results = []
        for cid in ids:
            row = row_map.get(cid)
            if row:
                results.append(
                    SearchResult(
                        chunk_id=row[0],
                        doc=row[1],
                        text=row[2],
                        score=score_map.get(cid, 0.0),
                        section_id=row[3],
                    )
                )
        return results

    def get_chunk_text(self, chunk_id: int) -> tuple[str, str] | None:
        """Fetch (doc, text) for a single chunk. Returns None if not found."""
        row = self.conn.execute(
            "SELECT d.name, c.text FROM chunks c"
            " JOIN documents d ON c.doc_id = d.id"
            " WHERE c.id = ?",
            (chunk_id,),
        ).fetchone()
        return (row[0], row[1]) if row else None


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
