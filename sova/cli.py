"""Command-line interface and Rich UI."""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from sova.config import BATCH_SIZE, DATA_DIR, DB_PATH
from sova.db import (
    connect_readonly,
    embedding_to_blob,
    get_doc_status,
    init_db,
    quantize_vectors,
)
from sova.extract import (
    chunk_text,
    extract_pdf,
    find_docs,
    find_section,
    parse_sections,
)
from sova.ollama_client import check_ollama, expand_query, get_embeddings_batch
from sova.search import hybrid_search

console = Console()

SOVA_ASCII = """\
   ___
  (o o)
 (  V  )
/|  |  |\\
  "   " """


def _label(name: str) -> str:
    padded = f"{name}:".ljust(8)
    return f"[dim]{padded}[/dim]"


def report(name: str, msg: str) -> None:
    console.print(f"{_label(name)} {msg}")


def report_progress(name: str) -> Progress:
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


def process_doc(
    name: str,
    pdf_path: Path | None,
    md_path: Path | None,
    conn: sqlite3.Connection,
) -> None:
    """Process a single document: extract, chunk, and embed."""
    console.print(f"\n[bold]{name}[/bold]")

    extracted_now = False
    if not md_path or not md_path.exists():
        if not pdf_path:
            report("error", "no PDF found")
            return
        try:
            start = time.time()
            markdown = extract_pdf(pdf_path)
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


def list_docs() -> None:
    """List all documents and their indexing status."""
    docs = find_docs()

    conn = None
    if DB_PATH.exists():
        conn = connect_readonly()

    table = Table(show_header=True, header_style="dim")
    table.add_column("Name")
    table.add_column("PDF", justify="right", style="dim")
    table.add_column("Chunks", justify="right")
    table.add_column("Text", justify="right")
    table.add_column("Vectors", justify="right")

    total_chunks, total_text, total_embed = 0, 0, 0

    for d in docs:
        status = get_doc_status(conn, d["name"]) if conn else {}
        chunks = status.get("chunks", 0)
        text_size = status.get("text_size", 0)
        embed_size = status.get("embed_size", 0)

        total_chunks += chunks
        total_text += text_size
        total_embed += embed_size

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

        table.add_row(d["name"], pdf_col, chunks_col, text_col, embed_col)

    if total_chunks > 0:
        table.add_section()
        table.add_row(
            "[bold]Total[/bold]",
            "",
            f"[bold]{total_chunks}[/bold]",
            f"[bold]{fmt_size(total_text)}[/bold]",
            f"[bold]{fmt_size(total_embed)}[/bold]",
        )

    if conn:
        conn.close()
    console.print(table)


def show_stats() -> None:
    """Show database statistics."""
    if DB_PATH.exists():
        console.print(
            f"\n[dim]Database:[/dim] {DB_PATH.name} ({fmt_size(DB_PATH.stat().st_size)})"
        )


def search_semantic(query: str, limit: int = 5, expand: bool = False) -> None:
    """Perform semantic search and display results."""
    if not DB_PATH.exists():
        console.print("[red]error:[/red] no database, run indexing first")
        return

    conn = connect_readonly()

    # Query expansion
    expanded_terms: list[str] = []
    if expand:
        expanded_terms = expand_query(query)
        if expanded_terms:
            console.print(f"[dim]expand:[/dim] {', '.join(expanded_terms)}")

    try:
        embeddings = get_embeddings_batch([query])
        query_emb = embeddings[0]
    except Exception as e:
        console.print(f"[red]error:[/red] {e}")
        conn.close()
        return

    results, n_vector, n_fts = hybrid_search(
        conn, query_emb, query, expanded_terms, limit
    )
    conn.close()

    if n_fts:
        console.print(f"[dim]hybrid:[/dim] {n_vector} vector + {n_fts} fts")

    if not results:
        console.print("[dim]no results[/dim]")
        return

    for r in results:
        console.print(
            f"\n[bold]{r['doc']}.md[/bold]:{r['start']}-{r['end']} [dim]({r['embed_score']:.2f})[/dim]"
        )
        preview = r["text"][:200].replace("\n", " ")
        if len(r["text"]) > 200:
            preview += "..."
        console.print(f"  [dim]{preview}[/dim]")


def main() -> None:
    """Main entry point."""
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
        "--no-expand", action="store_true", help="Disable LLM query expansion"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Delete DB and extracted files"
    )
    args = parser.parse_args()

    console.print(SOVA_ASCII, style="dim")

    if args.search:
        search_semantic(args.search, args.limit, expand=not args.no_expand)
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

    console.print(f"processing {len(docs)} documents")

    start_time = time.time()
    interrupted = False
    try:
        for doc in docs:
            process_doc(doc["name"], doc["pdf"], doc["md"], conn)
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
