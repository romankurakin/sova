# sova — Local Document Semantic Search

```text
   ___
  (o o)
 (  V  )
/|  |  |\
  "   "
```

*sova* — owl in Slavic languages.

## Quick Start

```bash
ln -s /path/to/your/docs docs         # Symlink your documents
uv run sova                           # Index all PDFs
```

## Search

```bash
uv run sova -s "your query"           # Semantic search (with LLM expansion)
uv run sova -s "query" -n 10          # More results
uv run sova -s "query" --no-expand    # Disable query expansion
```

## Commands

```bash
uv run sova                        # Index all PDFs
uv run sova [doc...]               # Index specific docs
uv run sova --list                 # List docs and status
uv run sova --reset                # Delete DB and extracted files
```

## How It Works

Sova converts PDFs to Markdown via pymupdf4llm, then indexes using two methods:

**Embeddings** (`qwen3-embedding:8b`): Dense vector representations that capture
semantic meaning. Chosen for strong multilingual support and 1024-dim vectors
that balance quality and storage. Enables finding conceptually related content
even when exact terms differ.

**BM25 full-text** (FTS5): Traditional keyword search with Porter stemming.
Excels at exact matches and specific technical terms that embeddings might miss.

**Hybrid search** combines both via Reciprocal Rank Fusion (RRF). This
outperforms either method alone by leveraging BM25's precision for exact terms
and embeddings' recall for semantic variations.

**Query expansion** (enabled by default) uses `gemma3:4b` to generate synonyms,
bridging vocabulary gaps when documents use different terminology than queries.
Use `--no-expand` to disable.

**Text density heuristics** detect and down-rank table-of-contents pages that
would otherwise match many queries due to keyword density.

Everything is stored in SQLite with sqlite-vec for fast similarity search.

## References

- **RRF (Reciprocal Rank Fusion)**: Cormack, Clarke, Büttcher. [Reciprocal Rank
  Fusion outperforms Condorcet and Individual Rank Learning Methods](https://dl.acm.org/doi/10.1145/1571941.1572114).
  SIGIR 2009. Used to combine BM25 and vector search results (k=60).

- **Text Density**: Vogels, Ganea, Eickhoff. [Web2Text: Deep Structured
  Boilerplate Removal](https://arxiv.org/abs/1801.02607). ECIR 2018. Inspired
  the text density heuristic for detecting index/ToC pages.

- **LLM Query Expansion**: Jagerman, Zhuang, et al. [Query Expansion by Prompting
  Large Language Models](https://arxiv.org/abs/2305.03653). 2023. Foundation for
  using LLMs to generate query synonyms and related terms.

## Requirements

You'll need [uv](https://docs.astral.sh/uv/) as the Python package manager and
[Ollama](https://ollama.ai) running locally. Models are pulled automatically on first run.

## License

MIT. Note: sqlite-vec uses Elastic License 2.0.
