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
ollama pull qwen3-embedding:8b        # Required for embeddings
ln -s /path/to/your/docs docs         # Symlink your documents
uv run sova.py                        # Index and search
```

## Search

```bash
uv run sova.py -s "your query"        # Semantic search
uv run sova.py -s "query" -n 10       # More results
```

## Commands

```bash
uv run sova.py                     # Index all PDFs
uv run sova.py [doc...]            # Index specific docs
uv run sova.py --list              # List docs and status
uv run sova.py --skip-topics       # Skip topic extraction (faster)
uv run sova.py --reset             # Delete DB and .md files
```

## How It Works

- PDF → Markdown extraction (pymupdf4llm)
- Local vector embeddings (Ollama)
- Optional LLM topic extraction
- SQLite storage (sqlite-vec)

## Requirements

- [uv](https://docs.astral.sh/uv/) - Python package manager
- [Ollama](https://ollama.ai) running locally
  - `qwen3-embedding:8b` for embeddings
  - `gemma3:12b` for topic extraction (optional)

## License

MIT. Note: sqlite-vec uses Elastic License 2.0.
