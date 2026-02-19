# Sova Benchmark

Measure search quality against LLM-judged ground truth.

## Usage

```bash
sova index /path/to/pdfs                                  # Index docs first
sova projects                                             # Find project id
uv run python -m benchmarks judge <project-id>            # Generate ground truth
uv run python -m benchmarks run <project-id> my-test      # Run benchmark
uv run python -m benchmarks show <project-id>             # View results
uv run python -m benchmarks --help                        # Full CLI help
```

## Methodology

1. **Ground Truth** - LLM (`gemma-3-12b-it` via llama-server) scores relevance 0-3 for top chunks per query
2. **Benchmark Run** - Search retrieves results, compared against ground truth
3. **Metrics** - Standard IR metrics computed, saved to `results/{name}.json`

Compare runs: refactor sova -> run benchmark -> compare to previous run.

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

| Metric | Question | How it works | Baseline |
|--------|----------|--------------|----------|
| **Latency P50** | Typical speed? | Median query time. User is waiting. | 310ms |
| **Latency P95** | Worst case? | 95th percentile - slowest 5% | 2125ms |
| **nDCG@10** | Best at top? | Rewards highly-relevant results ranked higher | 0.705 |
| **MRR@10** | First good result? | 1 / rank of first relevant result | 0.708 |
| **Precision@10** | How much junk? | (relevant in top-k) / k | 0.538 |
| **MAP@10** | Consistent ranking? | Avg precision at each relevant hit | 0.455 |
| **Recall@10** | Found them all? | (relevant in top-k) / total relevant | 0.539 |
| **Hit Rate@10** | At least one? | 1 if any relevant in top-k, else 0 | 1.000 |
| **Doc-Cov@10** | Multiple sources? | Unique documents in top-k. Shows breadth. | 0.356 |
| **S-Recall@10** | Diverse topics? | Results cover different aspects | 0.546 |
| **α-nDCG@10** | Novel results? | nDCG with redundancy penalty | 0.991 |
| **FP Rate@10** | False positives? | Precision on negative queries (should be 0) | 0.000 |

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
├── judge.py              # LLM judge (gemma-3-12b-it via llama-server)
├── evaluate.py           # Metrics computation
├── run_benchmark.py      # Search runner
└── results/              # Output JSONs and reports
```
