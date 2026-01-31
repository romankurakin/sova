#!/usr/bin/env python3
"""sova - Local document semantic search.

Index:
    uv run sova.py                         # Index all PDFs
    uv run sova.py mybook manual           # Index specific docs
    uv run sova.py --skip-topics           # Skip topic extraction
    uv run sova.py --list                  # Show index status
    uv run sova.py --reset                 # Delete index
    uv run sova.py --reset-topics          # Clear topics, keep embeddings

Search:
    uv run sova.py -s "query"              # Semantic search (shows preview)
    rg -i "keyword" data/*.md              # Text search (use rg)

Results show file:line-range. Use Read tool for full context.
Place documents in ./docs/ directory (or symlink).
"""

import argparse
import json
import hashlib
import heapq
import math
import re
import sqlite3
import struct
import subprocess
import sys
import time
import unicodedata
from array import array
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import snowballstemmer
import sqlite_vector
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)

console = Console()

SOVA_ASCII = """\
   ___
  (o o)
 (  V  )
/|  |  |\\
  "   " """

EMBEDDING_MODEL = "qwen3-embedding:8b"
EMBEDDING_DIM = 1024
TOPIC_MODEL = "gemma3:12b"
PROMPT_MODEL = "qwen3:30b"
OLLAMA_URL = "http://localhost:11434"

_topic_prompt_template: str | None = None

TOPIC_PROMPT_SUFFIX = (
    "Prefer precise, domain-specific technical terms (proper nouns, protocols, "
    "algorithms, components). Avoid generic words and document metadata."
)
TOPIC_OUTPUT_INSTRUCTIONS = (
    "Output a JSON array of 3-5 unique terms. No explanations."
)
TOPIC_MIN_LEN = 3
TOPIC_MAX_LEN = 60

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
DOCS_DIR = SCRIPT_DIR / "docs"
DB_PATH = DATA_DIR / "refs.db"
assert sqlite_vector.__file__ is not None
VECTOR_EXT = Path(sqlite_vector.__file__).parent / "binaries" / "vector.dylib"
BATCH_SIZE = 10


def _label(name: str) -> str:
    padded = f"{name}:".ljust(8)
    return f"[dim]{padded}[/dim]"


def report(name: str, msg: str):
    console.print(f"{_label(name)} {msg}")


def report_progress(name: str):
    return Progress(
        TextColumn(_label(name)),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


def fmt_size(size_bytes: int, dim_zero: bool = False) -> str:
    if size_bytes == 0:
        return "[dim]-[/dim]" if dim_zero else "-"
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:>6.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:>6.1f}m"
    return f"{seconds / 3600:>6.1f}h"


def pull_model(model: str) -> bool:
    try:
        result = subprocess.run(["ollama", "pull", model])
        return result.returncode == 0
    except Exception:
        return False


def check_ollama() -> tuple[bool, str]:
    try:
        req = Request(
            f"{OLLAMA_URL}/api/tags", headers={"Content-Type": "application/json"}
        )
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            missing = []
            if not any(EMBEDDING_MODEL in m for m in models):
                missing.append(EMBEDDING_MODEL)
            if not any(TOPIC_MODEL in m for m in models):
                missing.append(TOPIC_MODEL)
            if not any(PROMPT_MODEL in m for m in models):
                missing.append(PROMPT_MODEL)
            if missing:
                for model in missing:
                    report("pull", model)
                    if not pull_model(model):
                        return False, f"failed to pull {model}"
            return True, "ready"
    except URLError:
        return False, "not running (ollama serve)"
    except Exception as e:
        return False, str(e)


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    data = json.dumps({"model": EMBEDDING_MODEL, "input": texts}).encode()
    req = Request(
        f"{OLLAMA_URL}/api/embed",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())["embeddings"]
    except HTTPError as e:
        body = e.read().decode()
        raise Exception(f"{e.code}: {body}") from e


def generate_topic_prompt(docs: list[dict]) -> tuple[str, str | None]:
    """Ask LLM to generate optimal topic extraction prompt based on documents.

    docs: list of {"name": str, "size": int} dicts
    """
    def fmt(d):
        size_mb = d["size"] / 1024 / 1024
        if size_mb >= 1:
            return f"{d['name']} ({size_mb:.0f}MB)"
        return d["name"]

    if len(docs) <= 20:
        sample = docs
    else:
        step = len(docs) // 20
        sample = docs[::step][:20]

    docs_str = ", ".join(fmt(d) for d in sample)
    meta_prompt = f"""I have a document search system indexing these documents: {docs_str}

Generate a prompt for extracting 3-5 key topics from text chunks. Output comma-separated.

Rules:
- Extract searchable domain terms only
- NO document metadata or boilerplate
- NO generic section names
- Lowercase, acronyms uppercase

Format:
PROMPT: <prompt>
EXAMPLES: <examples>"""

    data = json.dumps(
        {"model": PROMPT_MODEL, "prompt": meta_prompt, "stream": False}
    ).encode()
    req = Request(
        f"{OLLAMA_URL}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=300) as resp:
        output = json.loads(resp.read()).get("response", "").strip()
        if not output or len(output) <= 20:
            raise ValueError(f"empty response from {PROMPT_MODEL}")

        prompt_match = re.search(
            r"PROMPT:\s*(.+?)(?=EXAMPLES:|$)", output, re.DOTALL | re.IGNORECASE
        )
        examples_match = re.search(
            r"EXAMPLES:\s*(.+)", output, re.DOTALL | re.IGNORECASE
        )

        if not prompt_match:
            raise ValueError(f"no PROMPT: found in response: {output[:100]}")

        prompt = prompt_match.group(1).strip()
        if len(prompt) < 20:
            raise ValueError(f"prompt too short: {prompt}")

        examples = examples_match.group(1).strip() if examples_match else None
        return prompt, examples


def get_topic_prompt(conn: sqlite3.Connection) -> str:
    """Get cached topic prompt or generate new one."""
    global _topic_prompt_template
    if _topic_prompt_template:
        return _topic_prompt_template

    try:
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = 'topic_prompt'"
        ).fetchone()
        if row:
            _topic_prompt_template = row[0]
            console.print(
                f"\n[bold]Topic prompt:[/bold]\n[cyan]{_topic_prompt_template}[/cyan]\n"
            )
            return _topic_prompt_template
    except sqlite3.OperationalError:
        pass

    docs = find_docs()
    report("prompt", f"generating from {len(docs)} docs...")
    _topic_prompt_template, examples = generate_topic_prompt(docs)
    console.print(
        f"\n[bold]Generated topic prompt:[/bold]\n[cyan]{_topic_prompt_template}[/cyan]"
    )
    if examples:
        console.print(f"\n[bold]Example topics:[/bold]\n[dim]{examples}[/dim]\n")
    else:
        console.print()

    conn.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)"
    )
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES ('topic_prompt', ?)",
        (_topic_prompt_template,),
    )
    conn.commit()

    return _topic_prompt_template


def _ascii_fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _clean_topic_label(label: str) -> str:
    cleaned = _ascii_fold(label).strip().strip(",;:.")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.lower()
    return cleaned


def _is_generic_topic(label: str, blocklist: set[str]) -> bool:
    if label in blocklist:
        return True
    words = label.split()
    if not words:
        return True
    return False


def _parse_topic_output(output: str) -> list[str]:
    output = output.strip()
    json_match = re.search(r"\[[\s\S]*\]", output)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            if isinstance(data, list):
                return [str(x) for x in data]
        except json.JSONDecodeError:
            pass
    if "\n" in output and "," not in output:
        return [line.strip("- ").strip() for line in output.splitlines() if line.strip()]
    return [p.strip() for p in output.split(",")]


def _is_valid_topic(label: str) -> bool:
    if not (TOPIC_MIN_LEN <= len(label) <= TOPIC_MAX_LEN):
        return False
    if not re.search(r"[a-z]", label):
        return False
    return True


def _dedupe_topics(labels: list[str]) -> list[str]:
    by_norm: dict[str, list[str]] = {}
    for label in labels:
        norm = normalize_topic(label)
        by_norm.setdefault(norm, []).append(label)
    deduped = []
    for variants in by_norm.values():
        variants.sort(key=lambda v: (len(v.split()), len(v)), reverse=True)
        deduped.append(variants[0])
    return deduped


def extract_topics(text: str, prompt: str, blocklist: set[str], retries: int = 3) -> list[str]:
    full_prompt = f"{prompt}\n\n{TOPIC_PROMPT_SUFFIX}\n{TOPIC_OUTPUT_INSTRUCTIONS}"
    data = json.dumps({"model": TOPIC_MODEL, "prompt": f"{full_prompt}\n\n{text[:2000]}", "stream": False}).encode()
    req = Request(f"{OLLAMA_URL}/api/generate", data=data, headers={"Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=120) as resp:
                output = json.loads(resp.read()).get("response", "")
            topics = _parse_topic_output(output)
            cleaned = []
            for topic in topics:
                label = _clean_topic_label(topic)
                if not label or not _is_valid_topic(label):
                    continue
                if _is_generic_topic(label, blocklist):
                    continue
                cleaned.append(label)
            return _dedupe_topics(cleaned)
        except (URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return []


def init_db() -> sqlite3.Connection:
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    conn.load_extension(str(VECTOR_EXT))
    conn.enable_load_extension(False)

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL,
            path TEXT NOT NULL, line_count INTEGER, expected_chunks INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS sections (
            id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL, title TEXT NOT NULL,
            level INTEGER NOT NULL, start_line INTEGER NOT NULL, end_line INTEGER,
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL, section_id INTEGER,
            start_line INTEGER NOT NULL, end_line INTEGER NOT NULL,
            word_count INTEGER NOT NULL, text TEXT NOT NULL, embedding BLOB,
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY, label TEXT UNIQUE NOT NULL, normalized_label TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunk_topics (
            chunk_id INTEGER NOT NULL, topic_id INTEGER NOT NULL,
            PRIMARY KEY (chunk_id, topic_id)
        );
        CREATE TABLE IF NOT EXISTS topic_cache (
            hash TEXT PRIMARY KEY,
            topics TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
        CREATE INDEX IF NOT EXISTS idx_topics_normalized ON topics(normalized_label);
        PRAGMA foreign_keys = ON;
    """)

    # FTS5 full-text search index
    conn.executescript("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            content='chunks',
            content_rowid='id',
            tokenize='porter unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
            INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
        END;
    """)

    # Populate FTS index if empty but chunks exist
    fts_count = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if fts_count == 0 and chunk_count > 0:
        conn.execute("INSERT INTO chunks_fts(rowid, text) SELECT id, text FROM chunks")
        conn.commit()

    try:
        conn.execute(
            f"SELECT vector_init('chunks', 'embedding', 'type=FLOAT32,dimension={EMBEDDING_DIM},distance=COSINE')"
        )
        conn.commit()
    except sqlite3.OperationalError:
        pass

    try:
        conn.execute("ALTER TABLE documents ADD COLUMN expected_chunks INTEGER")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    return conn


def quantize_vectors(conn: sqlite3.Connection):
    """Quantize vectors for fast native search."""
    try:
        conn.execute("SELECT vector_quantize('chunks', 'embedding')")
        conn.commit()
    except sqlite3.OperationalError:
        pass


def embedding_to_blob(emb: list[float]) -> bytes:
    return struct.pack(f"{len(emb)}f", *emb)


def get_doc_status(conn: sqlite3.Connection, name: str) -> dict:
    empty = {
        "extracted": False,
        "embedded": False,
        "complete": False,
        "topics": False,
        "topics_complete": False,
        "chunks": 0,
        "expected": None,
        "text_size": 0,
        "embed_size": 0,
        "topic_count": 0,
        "chunks_with_topics": 0,
    }
    row = conn.execute(
        "SELECT id, expected_chunks FROM documents WHERE name = ?", (name,)
    ).fetchone()
    if not row:
        return empty

    doc_id, expected = row
    row = conn.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(LENGTH(text)), 0), COALESCE(SUM(LENGTH(embedding)), 0)
        FROM chunks WHERE doc_id = ?
    """,
        (doc_id,),
    ).fetchone()
    chunk_count, text_size, embed_size = row

    embedded = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE doc_id = ? AND embedding IS NOT NULL",
        (doc_id,),
    ).fetchone()[0]

    topic_count = conn.execute(
        """
        SELECT COUNT(DISTINCT ct.topic_id)
        FROM chunk_topics ct JOIN chunks c ON ct.chunk_id = c.id WHERE c.doc_id = ?
    """,
        (doc_id,),
    ).fetchone()[0]

    chunks_with_topics = conn.execute(
        """
        SELECT COUNT(DISTINCT c.id)
        FROM chunks c
        WHERE c.doc_id = ? AND EXISTS (SELECT 1 FROM chunk_topics ct WHERE ct.chunk_id = c.id)
    """,
        (doc_id,),
    ).fetchone()[0]

    complete = expected is not None and chunk_count >= expected
    topics_complete = chunk_count > 0 and chunks_with_topics >= chunk_count

    return {
        "extracted": True,
        "embedded": embedded > 0,
        "complete": complete,
        "topics": topic_count > 0,
        "topics_complete": topics_complete,
        "chunks": chunk_count,
        "expected": expected,
        "text_size": text_size,
        "embed_size": embed_size,
        "topic_count": topic_count,
        "chunks_with_topics": chunks_with_topics,
    }


def parse_sections(lines: list[str]) -> list[dict]:
    sections = []
    for i, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if match:
            sections.append(
                {
                    "title": match.group(2).strip()[:200],
                    "level": len(match.group(1)),
                    "start_line": i + 1,
                    "end_line": None,
                }
            )
    for i, s in enumerate(sections):
        s["end_line"] = (
            sections[i + 1]["start_line"] - 1 if i + 1 < len(sections) else len(lines)
        )
    return sections


def chunk_text(lines: list[str], target_words: int = 500) -> list[dict]:
    chunks = []
    current_lines, current_words, chunk_start = [], 0, 1

    for i, line in enumerate(lines):
        line_words = len(line.split())
        current_lines.append(line)
        current_words += line_words

        is_break = line.strip() == "" or line.startswith("#")
        if (current_words >= target_words and is_break) or (
            line.startswith("#") and current_words > 50
        ):
            text = "\n".join(current_lines).strip()
            if text and current_words > 10:
                chunks.append(
                    {
                        "start_line": chunk_start,
                        "end_line": i + 1,
                        "word_count": current_words,
                        "text": text,
                    }
                )
            current_lines, current_words, chunk_start = [], 0, i + 2

    if current_lines:
        text = "\n".join(current_lines).strip()
        if text and current_words > 10:
            chunks.append(
                {
                    "start_line": chunk_start,
                    "end_line": len(lines),
                    "word_count": current_words,
                    "text": text,
                }
            )
    return chunks


def find_section(sections: list[dict], line: int) -> int | None:
    for i, s in enumerate(sections):
        if s["start_line"] <= line <= (s["end_line"] or float("inf")):
            return i
    return None


_stemmer = snowballstemmer.stemmer('english')


def normalize_topic(label: str) -> str:
    """Normalize topic for deduplication using Snowball stemmer."""
    n = _ascii_fold(label)
    n = re.sub(r"\s+", " ", n.lower().strip())
    n = re.sub(r"[^a-z0-9\s-]", "", n)
    words = n.split()
    stemmed = _stemmer.stemWords(words)
    return " ".join(stemmed)


def topic_cache_key(prompt: str, text: str) -> str:
    payload = f"{prompt}\n\n{text}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def find_docs() -> list[dict]:
    pdfs = list(DOCS_DIR.glob("*.pdf")) if DOCS_DIR.exists() else []
    mds = sorted(
        [m for m in DATA_DIR.glob("*.md")] if DATA_DIR.exists() else [],
        key=lambda p: p.name.lower(),
    )
    pdf_names = {p.stem for p in pdfs}

    docs = []
    for pdf in pdfs:
        md = DATA_DIR / f"{pdf.stem}.md"
        docs.append(
            {
                "name": pdf.stem,
                "pdf": pdf,
                "md": md if md.exists() else None,
                "size": pdf.stat().st_size,
            }
        )
    for md in mds:
        if md.stem not in pdf_names:
            docs.append(
                {"name": md.stem, "pdf": None, "md": md, "size": md.stat().st_size}
            )
    return sorted(docs, key=lambda d: (d["size"], d["name"].lower()))


def process_doc(
    name: str,
    pdf_path: Path | None,
    md_path: Path | None,
    conn: sqlite3.Connection,
    skip_topics: bool,
):
    console.print(f"\n[bold]{name}[/bold]")

    extracted_now = False
    if not md_path or not md_path.exists():
        if not pdf_path:
            report("error", "no PDF found")
            return
        try:
            import pymupdf4llm

            start = time.time()
            markdown = pymupdf4llm.to_markdown(str(pdf_path))
            DATA_DIR.mkdir(exist_ok=True)
            md_path = DATA_DIR / f"{name}.md"
            md_path.write_text(markdown, encoding="utf-8")
            lines = len(markdown.splitlines())
            report("extract", f"{lines:>9,} lines  {fmt_duration(time.time() - start)}")
            extracted_now = True
        except Exception as e:
            report("extract", f"[red]{e}[/red]")
            return

    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    if not extracted_now:
        report("extract", f"{len(lines):>9,} lines")
    sections = parse_sections(lines)
    chunks = chunk_text(lines)

    row = conn.execute("SELECT id FROM documents WHERE name = ?", (name,)).fetchone()
    if row:
        doc_id = row[0]
        conn.execute(
            "UPDATE documents SET expected_chunks = ? WHERE id = ? AND expected_chunks IS NULL",
            (len(chunks), doc_id),
        )
        conn.commit()
    else:
        cursor = conn.execute(
            "INSERT INTO documents (name, path, line_count, expected_chunks) VALUES (?, ?, ?, ?)",
            (name, str(md_path), len(lines), len(chunks)),
        )
        doc_id = cursor.lastrowid
        for s in sections:
            conn.execute(
                "INSERT INTO sections (doc_id, title, level, start_line, end_line) VALUES (?, ?, ?, ?, ?)",
                (doc_id, s["title"], s["level"], s["start_line"], s["end_line"]),
            )
        conn.commit()

    section_rows = conn.execute(
        "SELECT id, start_line FROM sections WHERE doc_id = ?", (doc_id,)
    ).fetchall()
    section_ids = {r[1]: r[0] for r in section_rows}

    existing = set(
        r[0]
        for r in conn.execute(
            "SELECT start_line FROM chunks WHERE doc_id = ? AND embedding IS NOT NULL",
            (doc_id,),
        ).fetchall()
    )
    pending = [(i, c) for i, c in enumerate(chunks) if c["start_line"] not in existing]

    if not pending:
        count = len(chunks)
        report("embed", f"{count:>9,} chunks")
    else:
        try:
            start = time.time()
            total = len(pending)

            with report_progress("embed") as progress:
                task = progress.add_task("", total=total)
                for batch_start in range(0, len(pending), BATCH_SIZE):
                    batch = pending[batch_start : batch_start + BATCH_SIZE]
                    texts = [chunks[i]["text"] for i, _ in batch]
                    embeddings = get_embeddings_batch(texts)

                    for (idx, chunk), emb in zip(batch, embeddings):
                        sec_idx = find_section(sections, chunk["start_line"])
                        sec_line = (
                            sections[sec_idx]["start_line"]
                            if sec_idx is not None
                            else None
                        )
                        sec_id = section_ids.get(sec_line)
                        conn.execute(
                            "INSERT INTO chunks (doc_id, section_id, start_line, end_line, word_count, text, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (
                                doc_id,
                                sec_id,
                                chunk["start_line"],
                                chunk["end_line"],
                                chunk["word_count"],
                                chunk["text"],
                                embedding_to_blob(emb),
                            ),
                        )
                    conn.commit()
                    progress.update(task, advance=len(batch))

            report("embed", f"{total:>9,} chunks {fmt_duration(time.time() - start)}")

        except Exception as e:
            conn.rollback()
            report("error", f"[red]{e}[/red]")
            return

    if skip_topics:
        report("topics", "skipped")
    else:
        rows = conn.execute(
            """
            SELECT c.id, c.text FROM chunks c
            WHERE c.doc_id = ? AND NOT EXISTS (
                SELECT 1 FROM chunk_topics ct WHERE ct.chunk_id = c.id
            )
            ORDER BY c.start_line ASC
        """,
            (doc_id,),
        ).fetchall()

        if not rows:
            count = conn.execute(
                """
                SELECT COUNT(DISTINCT ct.topic_id)
                FROM chunk_topics ct JOIN chunks c ON ct.chunk_id = c.id WHERE c.doc_id = ?
            """,
                (doc_id,),
            ).fetchone()[0]
            report("topics", f"{count:>9,} topics")
        else:
            try:
                total = len(rows)
                start = time.time()
                topic_prompt = get_topic_prompt(conn)
                blocklist = set()

                with report_progress("topics") as progress:
                    task = progress.add_task("", total=total)
                    for chunk_id, text in rows:
                        cache_key = topic_cache_key(topic_prompt, text)
                        cached = conn.execute(
                            "SELECT topics FROM topic_cache WHERE hash = ?",
                            (cache_key,),
                        ).fetchone()
                        if cached:
                            try:
                                topics = json.loads(cached[0])
                            except json.JSONDecodeError:
                                topics = []
                        else:
                            topics = extract_topics(text, topic_prompt, blocklist)
                            conn.execute(
                                "INSERT OR REPLACE INTO topic_cache (hash, topics) VALUES (?, ?)",
                                (cache_key, json.dumps(topics)),
                            )
                        for topic_label in topics:
                            normalized = normalize_topic(topic_label)
                            row = conn.execute(
                                "SELECT id FROM topics WHERE normalized_label = ?", (normalized,)
                            ).fetchone()
                            if row:
                                tid = row[0]
                            else:
                                conn.execute(
                                    "INSERT INTO topics (label, normalized_label) VALUES (?, ?)",
                                    (topic_label.lower(), normalized),
                                )
                                tid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                            conn.execute(
                                "INSERT OR IGNORE INTO chunk_topics (chunk_id, topic_id) VALUES (?, ?)",
                                (chunk_id, tid),
                            )
                        conn.commit()
                        progress.update(task, advance=1)

                topic_count = conn.execute(
                    "SELECT COUNT(DISTINCT topic_id) FROM chunk_topics ct JOIN chunks c ON ct.chunk_id = c.id WHERE c.doc_id = ?",
                    (doc_id,),
                ).fetchone()[0]
                report(
                    "topics",
                    f"{topic_count:>9,} topics {fmt_duration(time.time() - start)}",
                )

            except Exception as e:
                conn.rollback()
                report("error", f"[red]{e}[/red]")


def list_docs():
    docs = find_docs()

    conn = None
    if DB_PATH.exists():
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        conn.enable_load_extension(True)
        conn.load_extension(str(VECTOR_EXT))
        conn.enable_load_extension(False)

    table = Table(show_header=True, header_style="dim")
    table.add_column("Name")
    table.add_column("PDF", justify="right", style="dim")
    table.add_column("Chunks", justify="right")
    table.add_column("Text", justify="right")
    table.add_column("Vectors", justify="right")
    table.add_column("Topics", justify="right")
    table.add_column("Top", justify="left", style="dim")

    total_chunks, total_text, total_embed, total_topics = 0, 0, 0, 0

    for d in docs:
        status = get_doc_status(conn, d["name"]) if conn else {}
        chunks = status.get("chunks", 0)
        text_size = status.get("text_size", 0)
        embed_size = status.get("embed_size", 0)
        topic_count = status.get("topic_count", 0)

        total_chunks += chunks
        total_text += text_size
        total_embed += embed_size
        total_topics += topic_count

        pdf_col = fmt_size(d["size"])
        if chunks:
            expected = status.get("expected")
            if expected and chunks < expected:
                chunks_col = f"[yellow]{chunks}/{expected}[/yellow]"
            else:
                chunks_col = str(chunks)
        else:
            chunks_col = "[dim]-[/dim]"
        text_col = fmt_size(text_size, dim_zero=True)
        embed_col = fmt_size(embed_size, dim_zero=True)

        if topic_count:
            chunks_with_topics = status.get("chunks_with_topics", 0)
            if chunks and chunks_with_topics < chunks:
                topics_col = f"[yellow]{chunks_with_topics}/{chunks}[/yellow]"
            else:
                topics_col = str(topic_count)
        else:
            topics_col = "[dim]-[/dim]"

        top_topics = "-"
        if conn and topic_count:
            rows = conn.execute(
                """
                SELECT t.label, COUNT(*) AS uses
                FROM topics t
                JOIN chunk_topics ct ON ct.topic_id = t.id
                JOIN chunks c ON c.id = ct.chunk_id
                JOIN documents d ON d.id = c.doc_id
                WHERE d.name = ?
                GROUP BY t.id
                ORDER BY uses DESC, t.label ASC
                LIMIT 3
            """,
                (d["name"],),
            ).fetchall()
            if rows:
                top_topics = ", ".join(r[0] for r in rows)

        table.add_row(
            d["name"],
            pdf_col,
            chunks_col,
            text_col,
            embed_col,
            topics_col,
            top_topics,
        )

    if total_chunks > 0:
        table.add_section()
        table.add_row(
            "[bold]Total[/bold]",
            "",
            f"[bold]{total_chunks}[/bold]",
            f"[bold]{fmt_size(total_text)}[/bold]",
            f"[bold]{fmt_size(total_embed)}[/bold]",
            f"[bold]{total_topics}[/bold]",
            "",
        )

    if conn:
        conn.close()
    console.print(table)


def show_stats():
    if DB_PATH.exists():
        console.print(
            f"\n[dim]Database:[/dim] {DB_PATH.name} ({fmt_size(DB_PATH.stat().st_size)})"
        )


def search_semantic(query: str, limit: int = 5):
    if not DB_PATH.exists():
        console.print("[red]error:[/red] no database, run indexing first")
        return

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.enable_load_extension(True)
    conn.load_extension(str(VECTOR_EXT))
    conn.enable_load_extension(False)

    try:
        embeddings = get_embeddings_batch([query])
        query_emb = embeddings[0]
        query_blob = embedding_to_blob(query_emb)
    except Exception as e:
        console.print(f"[red]error:[/red] {e}")
        conn.close()
        return

    try:
        row = conn.execute("SELECT value FROM metadata WHERE key = 'topic_prompt'").fetchone()
        topic_prompt = row[0] if row else None
    except sqlite3.OperationalError:
        topic_prompt = None

    try:
        conn.execute(
            f"SELECT vector_init('chunks', 'embedding', 'type=FLOAT32,dimension={EMBEDDING_DIM},distance=COSINE')"
        )
    except sqlite3.OperationalError:
        pass

    query_topics = extract_topics(query, topic_prompt, set()) if topic_prompt else []
    query_topics_normalized = {normalize_topic(t) for t in query_topics}
    if query_topics:
        console.print(f"[dim]query topics:[/dim] {', '.join(query_topics)}")

    try:
        conn.execute("SELECT vector_quantize_preload('chunks', 'embedding')")
    except sqlite3.OperationalError:
        pass

    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    base_candidates = max(limit * 4, 50)
    adaptive = min(total_chunks, max(150, int(total_chunks * 0.05), base_candidates))
    candidates = min(max(base_candidates, adaptive), 1500)

    # Vector search
    try:
        vector_rows = conn.execute(
            """
            SELECT c.id, v.distance
            FROM chunks c
            JOIN vector_quantize_scan('chunks', 'embedding', ?, ?) AS v
            ON c.id = v.rowid
        """,
            (query_blob, candidates),
        ).fetchall()
        vector_results = [(row[0], 1.0 - row[1]) for row in vector_rows]
    except sqlite3.OperationalError:
        fallback = fallback_vector_scan(conn, query_emb, candidates)
        vector_results = [(r[0], 1.0 - r[6]) for r in fallback]

    # FTS5 BM25 search
    fts_results = search_fts(conn, query, candidates)

    # RRF fusion of vector and FTS results
    if fts_results:
        rrf_scores = rrf_fusion([vector_results, fts_results])
        fused_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        console.print(f"[dim]hybrid:[/dim] {len(vector_results)} vector + {len(fts_results)} fts")
    else:
        fused_ids = [r[0] for r in vector_results]
        rrf_scores = {r[0]: r[1] for r in vector_results}

    top_ids = fused_ids[:candidates]
    if not top_ids:
        conn.close()
        console.print("[dim]no results[/dim]")
        return

    # Fetch chunk details for fused results
    placeholders = ",".join("?" * len(top_ids))
    rows = conn.execute(
        f"""
        SELECT c.id, d.name, c.section_id, c.start_line, c.end_line, c.text
        FROM chunks c
        JOIN documents d ON c.doc_id = d.id
        WHERE c.id IN ({placeholders})
    """,
        top_ids,
    ).fetchall()
    chunk_data = {r[0]: r for r in rows}
    vector_score_map = {r[0]: r[1] for r in vector_results}

    # Fetch topics for results
    chunk_ids = top_ids
    chunk_topics_map = {}
    if query_topics_normalized and chunk_ids:
        topic_rows = conn.execute(
            f"""
            SELECT ct.chunk_id, t.normalized_label
            FROM chunk_topics ct
            JOIN topics t ON ct.topic_id = t.id
            WHERE ct.chunk_id IN ({placeholders})
        """,
            chunk_ids,
        ).fetchall()
        for chunk_id, norm_label in topic_rows:
            chunk_topics_map.setdefault(chunk_id, set()).add(norm_label)

    conn.close()

    scored = []
    for chunk_id in top_ids:
        if chunk_id not in chunk_data:
            continue
        _, doc, section_id, start, end, text = chunk_data[chunk_id]

        rrf_score = rrf_scores.get(chunk_id, 0.0)
        embed_score = vector_score_map.get(chunk_id, 0.0)

        topic_boost = 0.0
        matched_topics = []
        if query_topics_normalized and chunk_id in chunk_topics_map:
            matched = query_topics_normalized & chunk_topics_map[chunk_id]
            base_boost = min(len(matched) * 0.1, 0.3)
            similarity_scale = max(0.0, min(1.0, embed_score))
            topic_boost = base_boost * (0.5 + 0.5 * similarity_scale)
            matched_topics = list(matched)

        index_penalty = -0.5 if _is_index_like(text) else 0.0
        final_score = rrf_score * 30 + topic_boost + index_penalty
        scored.append(
            (
                final_score,
                embed_score,
                topic_boost + index_penalty,
                matched_topics,
                doc,
                section_id,
                start,
                end,
                text,
            )
        )

    scored.sort(reverse=True, key=lambda x: x[0])
    max_per_doc = 2
    filtered = []
    per_doc = {}
    per_section = {}
    for row in scored:
        doc = row[4]
        section_id = row[5]
        if per_doc.get(doc, 0) >= max_per_doc:
            continue
        if section_id is not None and per_section.get(section_id, 0) >= 1:
            continue
        per_doc[doc] = per_doc.get(doc, 0) + 1
        if section_id is not None:
            per_section[section_id] = per_section.get(section_id, 0) + 1
        filtered.append(row)
        if len(filtered) >= limit:
            break
    if len(filtered) < limit:
        filtered = scored[:limit]

    for final, embed, boost, topics, doc, section_id, start, end, text in filtered:
        boost_str = f"+{boost:.2f}" if boost > 0 else ""
        topic_str = f" [cyan]{','.join(topics)}[/cyan]" if topics else ""
        console.print(
            f"\n[bold]{doc}.md[/bold]:{start}-{end} [dim]({embed:.2f}{boost_str})[/dim]{topic_str}"
        )
        preview = text[:200].replace("\n", " ")
        if len(text) > 200:
            preview += "..."
        console.print(f"  [dim]{preview}[/dim]")


def _text_density(text: str) -> float:
    """Calculate letter density (letters / total chars)."""
    if not text:
        return 0.0
    letters = sum(c.isalpha() for c in text)
    return letters / len(text)


def _is_index_like(text: str) -> bool:
    """Detect ToC/index pages using text density."""
    if "table of contents" in text[:600].lower():
        return True
    return _text_density(text[:1000]) < 0.55


def search_fts(conn: sqlite3.Connection, query: str, limit: int) -> list[tuple[int, float]]:
    """Search using FTS5 BM25. Returns list of (chunk_id, bm25_score)."""
    try:
        fts_query = " ".join(
            f'"{term}"' for term in re.findall(r"[a-zA-Z0-9_-]+", query) if len(term) >= 2
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
    except sqlite3.OperationalError:
        return []


def rrf_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60,
) -> dict[int, float]:
    """Reciprocal Rank Fusion to combine multiple ranked lists."""
    scores: dict[int, float] = {}
    for ranked_list in ranked_lists:
        for rank, (item_id, _) in enumerate(ranked_list, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores


def fallback_vector_scan(
    conn: sqlite3.Connection, query_emb: list[float], candidates: int
) -> list[tuple[int, str, int | None, int, int, str, float]]:
    query_norm = math.sqrt(sum(v * v for v in query_emb))
    if query_norm == 0:
        return []

    q_len = len(query_emb)
    rows = conn.execute(
        """
        SELECT c.id, d.name, c.section_id, c.start_line, c.end_line, c.text, c.embedding
        FROM chunks c
        JOIN documents d ON c.doc_id = d.id
        WHERE c.embedding IS NOT NULL
    """
    ).fetchall()

    top: list[tuple[float, tuple[int, str, int | None, int, int, str, float]]] = []
    for chunk_id, doc, section_id, start, end, text, emb_blob in rows:
        emb = array("f")
        emb.frombytes(emb_blob)
        if len(emb) != q_len:
            continue

        dot = 0.0
        norm = 0.0
        for qv, ev in zip(query_emb, emb):
            dot += qv * ev
            norm += ev * ev
        if norm == 0.0:
            continue

        sim = dot / (query_norm * math.sqrt(norm))
        distance = 1.0 - sim
        entry = (sim, (chunk_id, doc, section_id, start, end, text, distance))
        if len(top) < candidates:
            heapq.heappush(top, entry)
        elif sim > top[0][0]:
            heapq.heapreplace(top, entry)

    top.sort(reverse=True, key=lambda x: x[0])
    return [item[1] for item in top]


def main():
    parser = argparse.ArgumentParser(
        description="sova - Local document semantic search"
    )
    parser.add_argument("docs", nargs="*", help="Specific documents to index")
    parser.add_argument(
        "-l", "--list", action="store_true", help="List documents and status"
    )
    parser.add_argument("-s", "--search", metavar="QUERY", help="Semantic search")
    parser.add_argument(
        "-n", "--limit", type=int, default=5, help="Max results (default: 5)"
    )
    parser.add_argument(
        "--skip-topics", action="store_true", help="Skip topic extraction"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Delete DB and extracted files"
    )
    parser.add_argument(
        "--reset-topics", action="store_true", help="Clear topics and prompt (keeps embeddings)"
    )
    parser.add_argument(
        "--show-prompt", action="store_true", help="Show current topic prompt"
    )
    args = parser.parse_args()

    console.print(SOVA_ASCII, style="dim")

    if args.search:
        search_semantic(args.search, args.limit)
        return

    if args.reset:
        if DB_PATH.exists():
            DB_PATH.unlink()
            console.print(f"[dim]deleted:[/dim] {DB_PATH.name}")
        if DATA_DIR.exists():
            for md in DATA_DIR.glob("*.md"):
                md.unlink()
                console.print(f"[dim]deleted:[/dim] {md.name}")
        console.print("reset complete")
        return

    if args.list:
        list_docs()
        show_stats()
        return

    if args.show_prompt:
        if not DB_PATH.exists():
            console.print("[dim]no database yet[/dim]")
            return
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT value FROM metadata WHERE key = 'topic_prompt'"
            ).fetchone()
            if row:
                console.print(f"[bold]Topic prompt:[/bold]\n[cyan]{row[0]}[/cyan]")
            else:
                console.print("[dim]no prompt generated yet[/dim]")
        except sqlite3.OperationalError:
            console.print("[dim]no prompt generated yet[/dim]")
        conn.close()
        return

    if args.reset_topics:
        if not DB_PATH.exists():
            console.print("[dim]no database yet[/dim]")
            return
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM chunk_topics")
        conn.execute("DELETE FROM topics")
        conn.execute("DELETE FROM topic_cache")
        conn.execute("DELETE FROM metadata WHERE key IN ('topic_prompt')")
        conn.commit()
        conn.close()
        console.print("topics cleared (embeddings preserved)")
        return

    ok, msg = check_ollama()
    if not ok:
        report("ollama", f"[red]{msg}[/red]")
        sys.exit(1)
    report("ollama", msg)

    try:
        conn = init_db()
    except Exception as e:
        report("database", f"[red]{e}[/red]")
        sys.exit(1)
    report("database", "ready")

    docs = find_docs()

    if args.docs:
        docs = [d for d in docs if d["name"] in args.docs]
        if not docs:
            console.print(f"[red]error:[/red] no matching documents: {args.docs}")
            sys.exit(1)

    if not args.skip_topics:
        get_topic_prompt(conn)

    console.print(f"processing {len(docs)} documents")

    start_time = time.time()
    interrupted = False
    try:
        for doc in docs:
            process_doc(doc["name"], doc["pdf"], doc["md"], conn, args.skip_topics)
    except KeyboardInterrupt:
        interrupted = True
        console.print("\n[dim]interrupted[/dim]")

    if not interrupted:
        report("quantize", "building index...")
        quantize_vectors(conn)

    conn.close()
    console.print()
    if interrupted:
        console.print(
            f"[dim]interrupted after {fmt_duration(time.time() - start_time)}[/dim]"
        )
    else:
        console.print(f"[dim]done in {fmt_duration(time.time() - start_time)}[/dim]")
    show_stats()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted[/dim]")
        sys.exit(130)
