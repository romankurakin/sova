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

Sova converts your PDFs to Markdown using pymupdf4llm, then generates vector
embeddings locally through Ollama. Topics can be optionally extracted via LLM
for richer metadata. Everything is stored in SQLite with sqlite-vec for fast
similarity search.

On first run, sova generates a topic extraction prompt based on your document
names. Use descriptive file names (e.g.,
`operating_systems_three_easy_pieces.pdf` instead of `os.pdf`) for better topic
extraction. Use `--reset-prompt` to regenerate the prompt after adding new
documents.

## Requirements

You'll need [uv](https://docs.astral.sh/uv/) as the Python package manager and
[Ollama](https://ollama.ai) running locally. Models (`qwen3-embedding:8b` for
embeddings, `gemma3:12b` for topics) are pulled automatically on first run.

## License

MIT. Note: sqlite-vec uses Elastic License 2.0.
