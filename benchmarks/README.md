# Search benchmark

## Summary

- The benchmark measures Sova search with 207 queries on 3,473 passages.
- The passages come from five technical manuals: RISC-V, LoongArch, xv6 and an
  operating-systems textbook.
- Two LLM judges give each pooled passage a grade from 0 to 3. The reference grade is
  the mean of the two grades.
- The two judges agree well: the quadratic-weighted κ is 0.88.
- Each experiment gets its own scores.
  [EXPERIMENTS.md](results/os-architectures-v1/EXPERIMENTS.md) shows all scores.

## Test collection

- **Corpus.** Five pinned PDFs with 1,835 pages. The Sova production chunker cuts them
  into passages of approximately 768 tokens. [sources.lock.json](sources.lock.json)
  identifies the files, and [NOTICE.md](results/NOTICE.md) gives their licenses.
- **Queries.** An editor wrote 187 English queries against the sources, in 99 groups
  of related queries. The query types are lookups, concepts, comparisons, procedures
  and problem descriptions. Ten Russian and ten Chinese translations of English
  queries test search across languages. [questions.md](queries/questions.md) shows all
  queries.
- **Pool.** For each query, the pool contains the top 10 results of BM25, dense and
  hybrid search. It also contains the top 10 results of each experiment run. The pool
  has 3,470 pairs, approximately 17 for each query. A new run can only add passages,
  and the judges grade only the new pairs.
- **Judges.** `gpt-6.1-sol` and `claude-sonnet-5-5` with `xhigh` reasoning effort, as
  set in [judges.json](judges.json). Each judge works alone and does not see the
  other grades. A judge gets the query, its intent and four passages with their source
  and section.
- **Grades.** 3: the passage answers the query. 2: the passage is a useful part of the
  answer. 1: the passage gives only background. 0: the passage does not help, or it
  is about the wrong architecture. A grade above 0 must quote the passage exactly.
- **Reference.** The reference grade of a pair is the mean of the two grades. Before
  it grades the pool, each judge must pass a control with a known answer.
- **Agreement.** The two grades are the same for 74% of the pairs and differ by one or
  less for 99%. For the decision "grade 2 or more", the κ is 0.78. The Claude grades
  are 0.23 lower on average. 25% of the pairs are useful, and each query has at least
  one useful passage.

## Metrics

- nDCG@1, @3, @5 and @10 use the gain 2^grade − 1. nDCG@10 is the primary metric.
- MRR@10, P@10, R@10 and Hit@10 count a passage as relevant if its grade is 2 or
  more. R@10 is relative to the pool.
- The 95% interval comes from a cluster bootstrap over the groups of related queries,
  with 10,000 samples.
- If the top 10 of a query contains a passage without a grade, the query is
  `unjudged`. An unjudged query does not count as zero.
- The report also gives the scores for each language, query type, topic, architecture
  and source document.

## Limitations

- No person checked the grades. When two models agree, this does not prove that the
  grades are correct.
- The pool contains the results of the measured systems. A very different system can
  find passages that have no grade. The `unjudged` count shows this.
- The queries that tune Sova are also the queries that evaluate it.
- The Russian and Chinese sets are small, so their intervals are wide.
- The latency comes from one Mac and includes the start of the CLI.

## How to run

```bash
uv sync --locked
uv run bench prepare              # Make the corpus, passages, index and pool.
uv run bench run [experiment]     # Measure one experiment. The default is baseline.
uv run bench prepare              # Add the passages that the run found to the pool.
uv run bench judge codex          # Grade the new pairs. The judges can run in parallel.
uv run bench judge claude
uv run bench report               # Score all runs and write EXPERIMENTS.md.
```

An experiment is a JSON file in [experiments/](experiments/). It has search settings
and a one-sentence `description`. It can change the search method, the candidate
depth, the ranking constants or the embedding model. A different embedder runs in a
private `llama-server` on the same passages, so the existing grades stay valid. To
start, copy [baseline.json](experiments/baseline.json).

All data is in [results/os-architectures-v1/](results/README.md): the grades, the
judge answers and the runs.
