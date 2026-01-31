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

PDFs -> Markdown (pymupdf4llm) -> chunked and indexed two ways:

- **Contextual Embeddings**: LLM detects document domain from section titles,
  prepends context to each chunk before embedding (Anthropic's Contextual Retrieval)
- **BM25 full-text**: Keyword search with Porter stemming for exact matches

**Hybrid search** fuses both via RRF. Query expansion generates synonyms to
bridge vocabulary gaps. Text density heuristics down-rank ToC pages.

Models: `qwen3-embedding:8b` (embeddings), `gemma3:4b` (domain detection, query expansion)

## References

[1] G. V. Cormack, C. L. A. Clarke, and S. Büttcher, "[Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods](https://doi.org/10.1145/1571941.1572114)," in *Proc. SIGIR*, Boston, MA, USA, 2009, pp. 758–759.

[2] T. Vogels, O. E. Ganea, and C. Eickhoff, "[Web2Text: Deep Structured Boilerplate Removal](https://doi.org/10.1007/978-3-319-76941-7_13)," in *Proc. ECIR*, Grenoble, France, 2018, pp. 167–179.

[3] R. Jagerman, H. Zhuang, Z. Qin, X. Wang, and M. Bendersky, "[Query Expansion by Prompting Large Language Models](https://doi.org/10.48550/arXiv.2305.03653)," in *Gen-IR@SIGIR*, 2023.

[4] Anthropic, "[Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)," Anthropic Blog, Sep. 2024.

## Requirements

- [uv](https://docs.astral.sh/uv/) — Python package manager
- [Ollama](https://ollama.ai) — running locally (models pulled automatically)

## License

MIT. Note: sqlite-vec uses Elastic License 2.0.
