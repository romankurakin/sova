"""Score-decay diversity for search results.

Penalizes repeated documents with exponential decay so highly relevant
same-doc chunks survive while mediocre ones yield to other sources.
"""


def score_decay_diversify(
    results: list[dict],
    limit: int = 10,
    decay: float = 0.7,
) -> list[dict]:
    """Score-decay diversity: penalize repeated docs with diminishing scores.

    Each additional chunk from the same doc is penalized by decay^count,
    so highly relevant same-doc chunks survive while mediocre ones yield
    to other sources.
    """
    doc_counts: dict[str, int] = {}

    # Compute adjusted scores
    scored = []
    for r in results:
        doc = r["doc"]
        count = doc_counts.get(doc, 0)
        raw = r.get("rerank_score", r.get("final_score", 0))
        adjusted = raw * (decay**count)
        scored.append((adjusted, count, r))
        doc_counts[doc] = count + 1

    # Re-sort by adjusted score and pick top-k
    scored.sort(key=lambda x: x[0], reverse=True)

    out = []
    for adjusted, _, r in scored[:limit]:
        r["diversity_score"] = adjusted
        out.append(r)
    return out
