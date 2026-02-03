"""Run benchmark against ground truth."""

from .search_interface import get_backend, SearchResult


def results_to_dicts(results: list[SearchResult]) -> list[dict]:
    """Convert SearchResult objects to dicts."""
    return [
        {
            "chunk_id": r.chunk_id,
            "doc": r.doc,
            "text": r.text,
            "score": r.score,
            "section_id": r.section_id,
        }
        for r in results
    ]


def run_search(query: str, limit: int = 10) -> list[dict]:
    """Run search and return results."""
    backend = get_backend()
    results = backend.search(query, limit=limit)
    return results_to_dicts(results)
