#!/usr/bin/env python3
"""sova - Local document semantic search.

Index:
    uv run sova.py                         # Index all PDFs
    uv run sova.py mybook manual           # Index specific docs
    uv run sova.py --skip-topics           # Skip topic extraction
    uv run sova.py --list                  # Show index status
    uv run sova.py --reset                 # Delete index

Search:
    uv run sova.py -s "query"              # Semantic search (shows preview)
    rg -i "keyword" *.md                   # Text search (use rg)

Results show file:line-range. Use Read tool for full context.
Place documents in ./docs/ directory (or symlink).
"""

import argparse
import json
import re
import sqlite3
import struct
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

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
EMBEDDING_DIM = 4096
TOPIC_MODEL = "gemma3:12b"
OLLAMA_URL = "http://localhost:11434"

SCRIPT_DIR = Path(__file__).parent.resolve()
DOCS_DIR = SCRIPT_DIR / "docs"
DB_PATH = SCRIPT_DIR / "refs.db"
assert sqlite_vector.__file__ is not None
VECTOR_EXT = Path(sqlite_vector.__file__).parent / "binaries" / "vector.dylib"
BATCH_SIZE = 10


def _label(name: str) -> str:
    padded = f"{name}:".ljust(8)
    return f"  [dim]{padded}[/dim]"


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


def extract_topics(text: str) -> list[str]:
    prompt = f"""Extract 3-5 key technical topics from this text. Return ONLY a JSON array like ["topic1", "topic2"].

Text:
{text[:2000]}"""
    data = json.dumps(
        {"model": TOPIC_MODEL, "prompt": prompt, "stream": False}
    ).encode()
    req = Request(
        f"{OLLAMA_URL}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=120) as resp:
            output = json.loads(resp.read()).get("response", "")
            match = re.search(r"\[.*?\]", output, re.DOTALL)
            if match:
                topics = json.loads(match.group())
                return [t.strip() for t in topics if isinstance(t, str) and len(t) > 1]
    except Exception:
        pass
    return []


def init_db() -> sqlite3.Connection:
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
        CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
        PRAGMA foreign_keys = ON;
    """)

    try:
        conn.execute(
            f"SELECT vector_init('chunks', 'embedding', 'type=FLOAT32,dimension={EMBEDDING_DIM}')"
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


def normalize_topic(label: str) -> str:
    n = re.sub(r"\s+", " ", label.lower().strip())
    return n[:-1] if n.endswith("s") and len(n) > 3 else n


def find_docs() -> list[dict]:
    pdfs = (
        sorted(DOCS_DIR.glob("*.pdf"), key=lambda p: p.stat().st_size)
        if DOCS_DIR.exists()
        else []
    )
    mds = sorted(
        [m for m in SCRIPT_DIR.glob("*.md") if m.name != "README.md"],
        key=lambda p: p.stat().st_size,
    )
    pdf_names = {p.stem for p in pdfs}

    docs = []
    for pdf in pdfs:
        md = SCRIPT_DIR / f"{pdf.stem}.md"
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
    return docs


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
            md_path = SCRIPT_DIR / f"{name}.md"
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
                topic_count = 0

                with report_progress("topics") as progress:
                    task = progress.add_task("", total=total)
                    for chunk_id, text in rows:
                        topics = extract_topics(text)
                        for topic_label in topics:
                            normalized = normalize_topic(topic_label)
                            conn.execute(
                                "INSERT OR IGNORE INTO topics (label, normalized_label) VALUES (?, ?)",
                                (topic_label, normalized),
                            )
                            tid = conn.execute(
                                "SELECT id FROM topics WHERE label = ?", (topic_label,)
                            ).fetchone()[0]
                            conn.execute(
                                "INSERT OR IGNORE INTO chunk_topics (chunk_id, topic_id) VALUES (?, ?)",
                                (chunk_id, tid),
                            )
                            topic_count += 1
                        conn.commit()
                        progress.update(task, advance=1)

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

        table.add_row(d["name"], pdf_col, chunks_col, text_col, embed_col, topics_col)

    if total_chunks > 0:
        table.add_section()
        table.add_row(
            "[bold]Total[/bold]",
            "",
            f"[bold]{total_chunks}[/bold]",
            f"[bold]{fmt_size(total_text)}[/bold]",
            f"[bold]{fmt_size(total_embed)}[/bold]",
            f"[bold]{total_topics}[/bold]",
        )

    if conn:
        conn.close()
    console.print(table)


def show_stats():
    if DB_PATH.exists():
        console.print(
            f"\n[dim]Database:[/dim] {DB_PATH.name} ({fmt_size(DB_PATH.stat().st_size)})"
        )


def blob_to_embedding(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


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
    except Exception as e:
        console.print(f"[red]error:[/red] {e}")
        conn.close()
        return

    rows = conn.execute("""
        SELECT c.id, d.name, c.start_line, c.end_line, c.text, c.embedding
        FROM chunks c
        JOIN documents d ON c.doc_id = d.id
        WHERE c.embedding IS NOT NULL
    """).fetchall()

    conn.close()

    if not rows:
        console.print("[dim]no results[/dim]")
        return

    scored = []
    for _, doc, start, end, text, emb_blob in rows:
        emb = blob_to_embedding(emb_blob)
        score = cosine_similarity(query_emb, emb)
        scored.append((score, doc, start, end, text))

    scored.sort(reverse=True, key=lambda x: x[0])

    for score, doc, start, end, text in scored[:limit]:
        console.print(f"\n[bold]{doc}.md[/bold]:{start}-{end} [dim]({score:.2f})[/dim]")
        preview = text[:200].replace("\n", " ")
        if len(text) > 200:
            preview += "..."
        console.print(f"  [dim]{preview}[/dim]")


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
    args = parser.parse_args()

    console.print(SOVA_ASCII, style="dim")

    if args.search:
        search_semantic(args.search, args.limit)
        return

    if args.reset:
        if DB_PATH.exists():
            DB_PATH.unlink()
            console.print(f"[dim]deleted:[/dim] {DB_PATH.name}")
        for md in SCRIPT_DIR.glob("*.md"):
            if md.name != "README.md":
                md.unlink()
                console.print(f"[dim]deleted:[/dim] {md.name}")
        console.print("reset complete")
        return

    if args.list:
        list_docs()
        show_stats()
        return

    ok, msg = check_ollama()
    if not ok:
        console.print(f"[red]ollama:[/red] {msg}")
        sys.exit(1)
    console.print(f"[dim]ollama:[/dim] {msg}")

    try:
        conn = init_db()
    except Exception as e:
        console.print(f"[red]database:[/red] {e}")
        sys.exit(1)
    console.print("[dim]database:[/dim] ready")

    docs = find_docs()

    if args.docs:
        docs = [d for d in docs if d["name"] in args.docs]
        if not docs:
            console.print(f"[red]error:[/red] no matching documents: {args.docs}")
            sys.exit(1)

    console.print(f"processing {len(docs)} documents")

    start_time = time.time()
    interrupted = False
    try:
        for doc in docs:
            process_doc(doc["name"], doc["pdf"], doc["md"], conn, args.skip_topics)
    except KeyboardInterrupt:
        interrupted = True
        console.print("\n[dim]interrupted[/dim]")

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
