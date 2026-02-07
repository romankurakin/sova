"""Search functionality: vector search, FTS, and hybrid fusion."""

import heapq
import math
import re
import sqlite3
from array import array

from sova.config import EMBEDDING_DIM
from sova.db import embedding_to_blob
from sova.diversity import score_decay_diversify


def search_vector(
    conn: sqlite3.Connection, query_blob: bytes, candidates: int
) -> list[tuple[int, float]]:
    """Vector similarity search. Returns list of (chunk_id, similarity_score)."""
    try:
        conn.execute(
            f"SELECT vector_init('chunks', 'embedding', 'type=FLOAT32,dimension={EMBEDDING_DIM},distance=COSINE')"
        )
    except sqlite3.OperationalError:
        pass

    # Preload quantized vectors into memory for faster search. Not all DBs
    # have quantized data yet (first run, interrupted indexing), so we ignore
    # the error and fall through to brute-force if needed.
    try:
        conn.execute("SELECT vector_quantize_preload('chunks', 'embedding')")
    except sqlite3.OperationalError:
        pass

    try:
        rows = conn.execute(
            """
            SELECT c.id, v.distance
            FROM chunks c
            JOIN vector_quantize_scan('chunks', 'embedding', ?, ?) AS v
            ON c.id = v.rowid
        """,
            (query_blob, candidates),
        ).fetchall()
        # sqlite-vector returns cosine distance (0 = identical), convert to
        # similarity (1 = identical) for consistent scoring downstream.
        return [(row[0], 1.0 - row[1]) for row in rows]
    except sqlite3.OperationalError:
        return []


def fallback_vector_scan(
    conn: sqlite3.Connection, query_emb: list[float], candidates: int
) -> list[tuple[int, float]]:
    """Fallback brute-force vector search when quantized index unavailable."""
    query_norm = math.sqrt(sum(v * v for v in query_emb))
    if query_norm == 0:
        return []

    q_len = len(query_emb)
    rows = conn.execute(
        """
        SELECT c.id, c.embedding
        FROM chunks c
        WHERE c.embedding IS NOT NULL
    """
    ).fetchall()

    # Use a min-heap to track top-K without sorting all results. O(n log k)
    # vs O(n log n) for sorted(), which matters when scanning thousands of chunks.
    top: list[tuple[float, int]] = []
    for chunk_id, emb_blob in rows:
        emb = array("f")
        emb.frombytes(emb_blob)
        if len(emb) != q_len:
            continue

        dot = sum(qv * ev for qv, ev in zip(query_emb, emb))
        norm = math.sqrt(sum(ev * ev for ev in emb))
        if norm == 0.0:
            continue

        sim = dot / (query_norm * norm)
        if len(top) < candidates:
            heapq.heappush(top, (sim, chunk_id))
        elif sim > top[0][0]:
            heapq.heapreplace(top, (sim, chunk_id))

    return [(chunk_id, sim) for sim, chunk_id in sorted(top, reverse=True)]


def search_fts(
    conn: sqlite3.Connection, query: str, limit: int
) -> list[tuple[int, float]]:
    """FTS5 BM25 search. Returns list of (chunk_id, bm25_score)."""
    try:
        # Quote each term for exact matching in FTS5 syntax. Single-char
        # tokens are dropped because they're almost always noise and can
        # cause FTS to return too many low-quality matches.
        fts_query = " ".join(
            f'"{term}"'
            for term in re.findall(r"[a-zA-Z0-9_-]+", query)
            if len(term) >= 2
        )
        if not fts_query:
            return []

        rows = conn.execute(
            """
            SELECT rowid, bm25(chunks_fts) as score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
        return [(row[0], abs(row[1])) for row in rows]
    except Exception:
        return []


def rrf_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    # k=60 is the standard RRF constant from Cormack et al. 2009.
    # Higher k reduces the influence of top ranks, making fusion more
    # uniform. 60 is widely used and works well in practice.
    k: int = 60,
) -> dict[int, float]:
    """Reciprocal Rank Fusion to combine multiple ranked lists."""
    scores: dict[int, float] = {}
    for ranked_list in ranked_lists:
        for rank, (item_id, _) in enumerate(ranked_list, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores


_ALPHA = re.compile(r"[^\W\d_]")


def text_density(text: str) -> float:
    """Calculate letter density (letters / total chars). Unicode-aware."""
    if not text:
        return 0.0
    return len(_ALPHA.findall(text)) / len(text)


def is_index_like(text: str) -> bool:
    """Detect ToC/index pages using text density."""
    if "table of contents" in text[:600].lower():
        return True
    # ToC pages have lots of dots, dashes, and numbers (page refs) which
    # push letter density below ~55%. Only check first 1000 chars because
    # the pattern is strongest at the start and full-text scan is wasteful.
    return text_density(text[:1000]) < 0.55


def compute_candidates(total_chunks: int, limit: int) -> int:
    """Compute number of vector candidates needed for a given limit."""
    # We need more candidates than the final limit because RRF fusion,
    # diversification, and index-page penalties all filter results down.
    # The adaptive sizing scales with corpus size (5% of chunks, min 150)
    # but caps at 1500 to keep search latency bounded.
    base_candidates = max(limit * 4, 50)
    adaptive = min(total_chunks, max(150, int(total_chunks * 0.05), base_candidates))
    return min(max(base_candidates, adaptive), 1500)


def get_vector_candidates(
    conn: sqlite3.Connection,
    query_emb: list[float],
    limit: int,
) -> list[tuple[int, float]]:
    """Get vector search candidates (cacheable). Returns list of (chunk_id, score)."""
    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    candidates = compute_candidates(total_chunks, limit)

    query_blob = embedding_to_blob(query_emb)

    vector_results = search_vector(conn, query_blob, candidates)
    if not vector_results:
        vector_results = fallback_vector_scan(conn, query_emb, candidates)

    return vector_results


def _exact_match_bonuses(
    conn: sqlite3.Connection,
    chunk_ids: list[int],
    query: str,
) -> dict[int, float]:
    """Score bonus for chunks containing query text verbatim (case-insensitive).

    Checks two signals:
    - Full phrase match (query appears as-is in chunk): +0.3
    - Per-term match (fraction of query terms found):   +0.15 * fraction
    """
    if not chunk_ids or not query.strip():
        return {}

    terms = [t for t in re.findall(r"[a-zA-Z0-9_-]+", query) if len(t) >= 2]
    if not terms:
        return {}

    query_lower = query.strip().lower()
    placeholders = ",".join("?" * len(chunk_ids))

    # Single SQL query: check full phrase + each individual term via INSTR.
    cols = ["CASE WHEN INSTR(LOWER(text), ?) > 0 THEN 1 ELSE 0 END"]
    params: list = [query_lower]
    for term in terms:
        cols.append("CASE WHEN INSTR(LOWER(text), ?) > 0 THEN 1 ELSE 0 END")
        params.append(term.lower())

    select = ", ".join(cols)
    rows = conn.execute(
        f"SELECT id, {select} FROM chunks WHERE id IN ({placeholders})",
        params + list(chunk_ids),
    ).fetchall()

    bonuses: dict[int, float] = {}
    for row in rows:
        phrase_hit = row[1]
        term_frac = sum(row[2:]) / len(terms)

        bonus = 0.3 * phrase_hit + 0.15 * term_frac
        if bonus > 0:
            bonuses[row[0]] = bonus

    return bonuses


def fuse_and_rank(
    conn,
    vector_results: list[tuple[int, float]],
    query_text: str,
    limit: int,
) -> tuple[list[dict], int, int]:
    """Fuse vector results with FTS and rank. Returns (results, n_vector, n_fts)."""
    if not vector_results:
        return [], 0, 0

    candidates = len(vector_results)

    fts_terms = [
        term for term in re.findall(r"[a-zA-Z0-9_-]+", query_text) if len(term) >= 2
    ]
    fts_results = search_fts(conn, query_text, candidates)

    if fts_results:
        rrf_scores = rrf_fusion([vector_results, fts_results])
        fused_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    else:
        fused_ids = [r[0] for r in vector_results]
        rrf_scores = {r[0]: r[1] for r in vector_results}

    top_ids = fused_ids[:candidates]
    if not top_ids:
        return [], 0, 0

    placeholders = ",".join("?" * len(top_ids))
    vector_score_map = {r[0]: r[1] for r in vector_results}
    fts_id_set = {r[0] for r in fts_results}

    # Exact match bonus: boost chunks containing the query verbatim.
    exact_bonuses = _exact_match_bonuses(conn, top_ids, query_text)

    # Fetch only doc name and is_index for ranking and diversification,
    # deferring the heavier text column to after we know the final top-k.
    meta = {
        r[0]: (r[1], r[2])
        for r in conn.execute(
            f"SELECT c.id, d.name, c.is_index FROM chunks c"
            f" JOIN documents d ON c.doc_id = d.id"
            f" WHERE c.id IN ({placeholders})",
            tuple(top_ids),
        ).fetchall()
    }

    scored = []
    for chunk_id in top_ids:
        if chunk_id not in meta:
            continue
        doc, is_idx = meta[chunk_id]
        rrf_score = rrf_scores.get(chunk_id, 0.0)
        embed_score = vector_score_map.get(chunk_id, 0.0)
        index_penalty = -0.5 if is_idx else 0.0
        exact_bonus = exact_bonuses.get(chunk_id, 0.0)
        scored.append(
            {
                "chunk_id": chunk_id,
                "doc": doc,
                "final_score": rrf_score * 30 + index_penalty + exact_bonus,
                "embed_score": embed_score,
                "rrf_score": rrf_score,
                "fts_hit": chunk_id in fts_id_set,
                "is_idx": is_idx,
                "fts_terms": fts_terms,
            }
        )

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    filtered = score_decay_diversify(scored, limit=limit, decay=0.95)

    if filtered:
        hi, lo = filtered[0]["diversity_score"], filtered[-1]["diversity_score"]
        span = hi - lo
        for r in filtered:
            r["display_score"] = (r["diversity_score"] - lo) / span if span else 1.0

    # Now that we know the final top-k, fetch the text and line ranges
    # only for results we will actually display.
    final_ids = [r["chunk_id"] for r in filtered]
    if final_ids:
        ph = ",".join("?" * len(final_ids))
        text_data = {
            r[0]: r
            for r in conn.execute(
                f"SELECT id, section_id, start_line, end_line, text"
                f" FROM chunks WHERE id IN ({ph})",
                tuple(final_ids),
            ).fetchall()
        }
        for r in filtered:
            row = text_data.get(r["chunk_id"])
            if row:
                r["section_id"] = row[1]
                r["start"] = row[2]
                r["end"] = row[3]
                r["text"] = row[4]

    return filtered, len(vector_results), len(fts_results)


def hybrid_search(
    conn,
    query_emb: list[float],
    query_text: str,
    limit: int,
) -> tuple[list[dict], int, int]:
    """Perform hybrid vector + FTS search with RRF fusion."""
    vector_results = get_vector_candidates(conn, query_emb, limit)
    return fuse_and_rank(conn, vector_results, query_text, limit)
