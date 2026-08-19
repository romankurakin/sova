"""Build and validate a frozen search benchmark suite from blinded qrel shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, action="append", required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen suite: {args.output}; choose a new output"
        )

    specs = json.loads(args.queries.read_text(encoding="utf-8"))["queries"]
    spec_by_id = {item["id"]: item for item in specs}
    if len(spec_by_id) != len(specs):
        raise ValueError("duplicate query IDs")

    judged_by_id: dict[str, dict] = {}
    for shard_path in args.judgments:
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        for item in shard["queries"]:
            query_id = item["id"]
            if query_id in judged_by_id:
                raise ValueError(f"duplicate judged query: {query_id}")
            judged_by_id[query_id] = item
    if judged_by_id.keys() != spec_by_id.keys():
        missing = sorted(spec_by_id.keys() - judged_by_id.keys())
        extra = sorted(judged_by_id.keys() - spec_by_id.keys())
        raise ValueError(f"query mismatch; missing={missing}, extra={extra}")

    all_chunk_ids: set[int] = set()
    score_counts: Counter[int] = Counter()
    for spec in specs:
        item = judged_by_id[spec["id"]]
        for field in ("query", "category", "language"):
            if item.get(field) != spec[field]:
                raise ValueError(f"{spec['id']} has mismatched {field}")
        seen: set[int] = set()
        for judgment in item["judgments"]:
            chunk_id = int(judgment["chunk_id"])
            score = int(judgment["score"])
            if chunk_id in seen:
                raise ValueError(f"{spec['id']} repeats chunk {chunk_id}")
            if score not in {0, 1, 2, 3}:
                raise ValueError(f"{spec['id']} has invalid score {score}")
            seen.add(chunk_id)
            all_chunk_ids.add(chunk_id)
            score_counts[score] += 1
        if spec["category"] != "negative" and not any(
            judgment["score"] >= 2 for judgment in item["judgments"]
        ):
            raise ValueError(f"{spec['id']} has no relevant judgments")

    uri = f"file:{args.database.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows: dict[int, str] = {}
        ids = sorted(all_chunk_ids)
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows.update(
                connection.execute(
                    f"SELECT c.id, d.name FROM chunks c"
                    f" JOIN documents d ON d.id = c.doc_id"
                    f" WHERE c.id IN ({placeholders})",
                    batch,
                ).fetchall()
            )
        if rows.keys() != all_chunk_ids:
            raise ValueError("suite references missing chunks")
        for item in judged_by_id.values():
            for judgment in item["judgments"]:
                if rows[judgment["chunk_id"]] != judgment["doc"]:
                    raise ValueError(
                        f"chunk {judgment['chunk_id']} has mismatched document"
                    )
        documents = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        contexts = connection.execute("SELECT COUNT(*) FROM chunk_contexts").fetchone()[0]

    queries = [judged_by_id[spec["id"]] for spec in specs]
    payload = {
        "suite": {
            "id": "search",
            "schema_version": 2,
            "created": "2026-08-20",
            "description": "Frozen regression benchmark for local document search.",
            "corpus": {
                "project": "operating-system-documents",
                "database_sha256": _sha256(args.database),
                "documents": documents,
                "chunks": chunks,
                "contextualized_chunks": contexts,
            },
            "query_count": len(queries),
            "language_counts": dict(Counter(item["language"] for item in queries)),
            "category_counts": dict(Counter(item["category"] for item in queries)),
            "judgment_count": sum(len(item["judgments"]) for item in queries),
            "score_counts": {str(score): score_counts[score] for score in range(4)},
            "relevance_threshold": 2,
            "unjudged_policy": "error",
            "judging": "Blinded pooled judgments; candidate-system provenance is not stored.",
            "acceptance": {
                "minimum_mean_ndcg_delta_at_10": 0.03,
                "ndcg_bootstrap_ci_lower_must_exceed": 0.0,
                "maximum_latency_p50_ms": 500,
                "maximum_latency_p95_ms": 750,
            },
            "comparison_rule": "Suite SHA-256, initial database SHA-256, k, and query IDs must match.",
        },
        "queries": queries,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
