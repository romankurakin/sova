# Sova Benchmark

Measure search quality against LLM-judged ground truth.

## Usage

```bash
uv run python -m sova                               # Index docs first
uv run python -m benchmarks judge                   # Generate ground truth
uv run python -m benchmarks judge --debias          # Use debiasing
uv run python -m benchmarks run my-test             # Run benchmark
uv run python -m benchmarks show                    # View results
uv run python -m benchmarks --help                  # Full CLI help
```

## Methodology

1. **Ground Truth** - LLM (gemma3:27b) scores relevance 0-3 for top chunks per query
2. **Benchmark Run** - Search retrieves results, compared against ground truth
3. **Metrics** - Standard IR metrics computed, saved to `results/{name}.json`

Compare runs: refactor sova → run benchmark → compare to previous run.

## Query Categories

20 queries across 5 categories testing different retrieval capabilities:

| Category | Tests | Example |
|----------|-------|---------|
| **Exact lookup** | BM25, specific terms | "ELR_EL1 register" |
| **Conceptual** | Semantic understanding | "how the OS reclaims memory from a terminated process" |
| **Cross-doc** | Multi-document retrieval | "how ARM and RISC-V differ in exception handling" |
| **Natural** | Vague/real user phrasing | "my kernel crashes right after enabling the MMU" |
| **Negative** | False positive rate | "Python asyncio event loop" (should return nothing) |

## Metrics

| Metric | Question | How it works | Target |
|--------|----------|--------------|--------|
| **Latency P50** | Typical speed? | Median query time. User is waiting. | < 400ms |
| **Latency P95** | Worst case? | 95th percentile - slowest 5% | < 800ms |
| **nDCG** | Best at top? | Rewards highly-relevant results ranked higher | > 0.7 |
| **MRR** | First good result? | 1 / rank of first relevant result | > 0.7 |
| **Precision** | How much junk? | (relevant in top-k) / k | > 0.6 |
| **MAP** | Consistent ranking? | Avg precision at each relevant hit | > 0.5 |
| **Recall** | Found them all? | (relevant in top-k) / total relevant | Higher |
| **Hit Rate** | At least one? | 1 if any relevant in top-k, else 0 | > 0.9 |
| **Doc-Cov** | Multiple sources? | Unique documents in top-k. Shows breadth. | Higher |
| **S-Recall** | Diverse topics? | Results cover different aspects | Higher |
| **α-nDCG** | Novel results? | nDCG with redundancy penalty | Higher |
| **FP Rate** | False positives? | Precision on negative queries (should be 0) | 0.0 |

## Relevance Scale

| Score | Meaning |
|-------|---------|
| 3 | Directly answers the query |
| 2 | Contains useful related info |
| 1 | Mentions topic but no detail |
| 0 | Unrelated |

## Files

```text
benchmarks/
├── search_interface.py   # <== Update this when refactoring sova
├── judge.py              # LLM judge
├── evaluate.py           # Metrics computation
├── run_benchmark.py      # Search runner
└── results/              # Output JSONs and reports
```
