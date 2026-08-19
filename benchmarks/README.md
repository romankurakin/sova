# Search Benchmark

`suite.json` is the single active, frozen search benchmark. It contains 25
queries and 506 blinded pooled judgments for the
`operating-system-documents` corpus.

## Recorded results

- `results/baseline.json` — the production hybrid search: vector retrieval,
  FTS, reciprocal-rank fusion, exact-match bonuses, index penalties, and
  diversity selection.
- `results/multilingual-reranker-candidate.json` — an evaluated GTE
  multilingual reranker candidate. It met the latency budget but was rejected
  because it did not improve quality over the baseline.

Both files use the same suite hash, corpus hash, `k`, and query IDs. The
candidate includes paired deltas and a bootstrap confidence interval against
the baseline.

## Run

```bash
uv run python -m benchmarks run operating-system-documents <result-name> \
  --description "What changed and why this run exists"

# Compare a future candidate with the recorded baseline.
uv run python -m benchmarks run operating-system-documents <candidate-name> \
  --baseline baseline \
  --description "Candidate algorithm and its important settings"
```

The runner performs one deterministic quality pass. It warms the embedding
model with two untimed requests, then measures latency over every suite query.
An unjudged top-10 chunk is an error. Existing result files are never
overwritten.

Each result stores `experiment` as one plain-text sentence describing what was
tested. The frozen suite hash is the only technical identity needed for a fair
comparison.

## What is frozen

The suite records:

- exact, conceptual, cross-document, natural-language, and negative queries;
- English and Russian slices;
- blinded 0–3 relevance judgments;
- the exact corpus database SHA-256 and counts;
- acceptance criteria and a suite SHA-256 copied into every result.

Judgment candidates were pooled from multiple retrieval methods and evaluated
without their system provenance. Pooling only makes the qrels sufficiently
complete; the benchmark evaluates each submitted top 10 solely against the
frozen judgments.

## Metrics and acceptance

Primary metric: nDCG@10. Secondary metrics: MRR@10, MAP@10, Precision@10, and
pooled Recall@10. Results also contain category/language slices and per-query
ranks for paired comparison.

A candidate is accepted only if:

- mean nDCG@10 improves by at least `+0.03`;
- the lower bound of the paired 95% bootstrap CI is above `0`;
- P50 latency is no more than `500 ms`;
- P95 latency is no more than `750 ms`.

## Maintaining the suite

Never edit a frozen suite or an existing result to accommodate an algorithm.
If a future system returns an unjudged chunk, create and review a new suite,
then rerun every system being compared. Results are comparable only when suite
hash, initial database hash, `k`, and query IDs all match.

`build_suite.py` validates blinded judgment shards against SQLite in read-only
mode and refuses to overwrite its output. `judge` writes only
`draft-suite.json`; it cannot replace the active frozen suite.
