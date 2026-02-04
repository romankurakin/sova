"""Command-line interface and Rich UI."""

import argparse
import sys
import time
from pathlib import Path

from rich.table import Table

from sova.cache import get_cache
from sova.config import BATCH_SIZE, CONTEXT_MODEL, DATA_DIR, DB_PATH
from sova.db import (
    connect_readonly,
    get_doc_status,
    init_db,
)
from sova.extract import (
    chunk_text,
    extract_pdf,
    find_docs,
    find_section,
    parse_sections,
)
from sova.ollama_client import (
    check_ollama,
    generate_context,
    get_embeddings_batch,
    get_query_embedding,
)
from sova.search import (
    compute_candidates,
    fuse_and_rank,
    get_vector_candidates,
    is_index_like,
)
from sova.ui import console, fmt_duration, report, report_progress


def _snippet(text: str, terms: list[str], width: int = 200) -> str:
    """Extract a preview snippet, centering on the first FTS term match."""
    flat = " ".join(text.split())
    if len(flat) <= width:
        return flat
    flat_lower = flat.lower()
    best_pos = -1
    for term in terms:
        pos = flat_lower.find(term.lower())
        if pos != -1:
            best_pos = pos
            break
    if best_pos == -1:
        return flat[:width] + "..."
    # Center the window around the match.
    start = max(0, best_pos - width // 2)
    end = start + width
    if end > len(flat):
        end = len(flat)
        start = max(0, end - width)
    snippet = flat[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(flat):
        snippet = snippet + "..."
    return snippet


SOVA_ASCII = """\
   ___
  (o o)
 (  V  )
/|  |  |\\
  "   " """


def fmt_size(size_bytes: int, dim_zero: bool = False) -> str:
    if size_bytes == 0:
        return "-" if dim_zero else "-"
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def process_doc(
    name: str,
    pdf_path: Path | None,
    md_path: Path | None,
    conn,
) -> None:
    """Process a single document: extract, chunk, generate context, and embed."""
    console.print(f"\n{name}")

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
        doc_id = None

    if doc_id is None:
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

    # Insert any new chunks that don't exist in DB yet (text extraction).
    existing_starts = set(
        r[0]
        for r in conn.execute(
            "SELECT start_line FROM chunks WHERE doc_id = ?", (doc_id,)
        ).fetchall()
    )
    new_chunks = []
    for chunk in chunks:
        if chunk["start_line"] not in existing_starts:
            sec_idx = find_section(sections, chunk["start_line"])
            sec_line = sections[sec_idx]["start_line"] if sec_idx is not None else None
            sec_id = section_ids.get(sec_line)
            new_chunks.append(
                (
                    doc_id,
                    sec_id,
                    chunk["start_line"],
                    chunk["end_line"],
                    chunk["word_count"],
                    chunk["text"],
                    1 if is_index_like(chunk["text"]) else 0,
                )
            )
    if new_chunks:
        conn.executemany(
            "INSERT INTO chunks (doc_id, section_id, start_line, end_line, word_count, text, is_index) VALUES (?, ?, ?, ?, ?, ?, ?)",
            new_chunks,
        )
    conn.commit()

    # Build a map from start_line to chunk_id for this document.
    chunk_rows = conn.execute(
        "SELECT id, start_line FROM chunks WHERE doc_id = ?", (doc_id,)
    ).fetchall()
    chunk_id_by_start = {r[1]: r[0] for r in chunk_rows}

    # Generate context for chunks without chunk_contexts rows.
    existing_contexts = set(
        r[0]
        for r in conn.execute(
            "SELECT chunk_id FROM chunk_contexts WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?)",
            (doc_id,),
        ).fetchall()
    )
    chunks_needing_context = []
    for i, chunk in enumerate(chunks):
        chunk_id = chunk_id_by_start.get(chunk["start_line"])
        if chunk_id is None:
            continue
        if chunk_id not in existing_contexts:
            chunks_needing_context.append((i, chunk, chunk_id))

    if chunks_needing_context:
        try:
            start = time.time()
            total = len(chunks_needing_context)

            with report_progress("context") as progress:
                task = progress.add_task("", total=total)
                for i, chunk, chunk_id in chunks_needing_context:
                    sec_idx = find_section(sections, chunk["start_line"])
                    sec_title = (
                        sections[sec_idx]["title"] if sec_idx is not None else None
                    )
                    prev_text = chunks[i - 1]["text"] if i > 0 else ""
                    next_text = chunks[i + 1]["text"] if i + 1 < len(chunks) else ""

                    try:
                        ctx = generate_context(
                            name, sec_title, chunk["text"], prev_text, next_text
                        )
                    except Exception:
                        # Graceful fallback: skip context, embedding will use
                        # the basic [doc | section] prefix instead.
                        progress.update(task, advance=1)
                        continue

                    conn.execute(
                        "INSERT INTO chunk_contexts (chunk_id, context, model) VALUES (?, ?, ?)",
                        (chunk_id, ctx, CONTEXT_MODEL),
                    )
                    # NULL the embedding so Pass 2 re-embeds with context.
                    conn.execute(
                        "UPDATE chunks SET embedding = NULL WHERE id = ?",
                        (chunk_id,),
                    )
                    conn.commit()
                    progress.update(task, advance=1)

            report("context", f"{total:>9,} chunks {fmt_duration(time.time() - start)}")

        except Exception as e:
            conn.rollback()
            report("error", f"[red]{e}[/red]")
            return
    else:
        count = len(chunks)
        report("context", f"{count:>9,} chunks")

    # Embed chunks with embedding IS NULL, prepending context.
    context_map: dict[int, str] = {
        r[0]: r[1]
        for r in conn.execute(
            "SELECT cc.chunk_id, cc.context FROM chunk_contexts cc JOIN chunks c ON cc.chunk_id = c.id WHERE c.doc_id = ?",
            (doc_id,),
        ).fetchall()
    }

    # Find chunks missing embeddings in one query.
    embedded_ids = set(
        r[0]
        for r in conn.execute(
            "SELECT id FROM chunks WHERE doc_id = ? AND embedding IS NOT NULL",
            (doc_id,),
        ).fetchall()
    )
    pending_embed = []
    for i, chunk in enumerate(chunks):
        chunk_id = chunk_id_by_start.get(chunk["start_line"])
        if chunk_id is not None and chunk_id not in embedded_ids:
            pending_embed.append((i, chunk, chunk_id))

    if not pending_embed:
        count = len(chunks)
        report("embed", f"{count:>9,} chunks")
    else:
        try:
            start = time.time()
            total = len(pending_embed)

            with report_progress("embed") as progress:
                task = progress.add_task("", total=total)
                for batch_start in range(0, len(pending_embed), BATCH_SIZE):
                    batch = pending_embed[batch_start : batch_start + BATCH_SIZE]

                    contextual_texts = []
                    for i, chunk, chunk_id in batch:
                        sec_idx = find_section(sections, chunk["start_line"])
                        sec_title = (
                            sections[sec_idx]["title"] if sec_idx is not None else None
                        )
                        prefix = f"[{name}"
                        if sec_title:
                            prefix += f" | {sec_title}"
                        prefix += "]\n\n"

                        llm_ctx = context_map.get(chunk_id)
                        if llm_ctx:
                            prefix += llm_ctx + "\n\n"

                        contextual_texts.append(prefix + chunk["text"])

                    emb_blobs = get_embeddings_batch(contextual_texts)

                    for (idx, chunk, chunk_id), blob in zip(batch, emb_blobs):
                        conn.execute(
                            "UPDATE chunks SET embedding = ? WHERE id = ?",
                            (blob, chunk_id),
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
    table.add_column("Context", justify="right")
    table.add_column("Text", justify="right")
    table.add_column("Vectors", justify="right")

    total_chunks, total_text, total_embed, total_ctx = 0, 0, 0, 0

    for d in docs:
        status = get_doc_status(conn, d["name"]) if conn else {}
        chunks = status.get("chunks", 0)
        text_size = status.get("text_size", 0)
        embed_size = status.get("embed_size", 0)
        ctx_count = status.get("contextualized", 0)

        total_chunks += chunks
        total_text += text_size
        total_embed += embed_size
        total_ctx += ctx_count

        pdf_col = fmt_size(d["size"])
        if chunks:
            expected = status.get("expected")
            if expected and chunks < expected:
                chunks_col = f"[yellow]{chunks}/{expected}[/yellow]"
            else:
                chunks_col = str(chunks)
        else:
            chunks_col = "-"
        if ctx_count:
            if chunks and ctx_count < chunks:
                ctx_col = f"[yellow]{ctx_count}/{chunks}[/yellow]"
            else:
                ctx_col = str(ctx_count)
        else:
            ctx_col = "-"
        text_col = fmt_size(text_size, dim_zero=True)
        embed_col = fmt_size(embed_size, dim_zero=True)

        table.add_row(d["name"], pdf_col, chunks_col, ctx_col, text_col, embed_col)

    if total_chunks > 0:
        table.add_section()
        table.add_row(
            "Total",
            "",
            f"{total_chunks}",
            f"{total_ctx}",
            f"{fmt_size(total_text)}",
            f"{fmt_size(total_embed)}",
        )

    if conn:
        conn.close()
    console.print(table)


def show_stats() -> None:
    """Show database statistics."""
    if DB_PATH.exists():
        console.print(
            f"\nDatabase: {DB_PATH.name} ({fmt_size(DB_PATH.stat().st_size)})"
        )
        cache = get_cache()
        conn = connect_readonly()
        count = conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
        conn.close()
        console.print(f"Cache: {count} entries (max {cache.max_size})")


def search_semantic(query: str, limit: int = 10, verbose: bool = False) -> None:
    """Perform semantic search and display results."""
    if not DB_PATH.exists():
        console.print("[red]error:[/red] no database, run indexing first")
        return

    conn = connect_readonly()
    cache = get_cache()

    try:
        query_emb = get_query_embedding(query)
    except Exception as e:
        console.print(f"[red]error:[/red] {e}")
        conn.close()
        return

    # Compute min candidates before checking cache so we only accept cached
    # results that searched at least as broadly as we need.
    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    min_candidates = compute_candidates(total_chunks, limit)
    cached_vectors = cache.get(conn, query_emb, min_candidates)
    if cached_vectors:
        console.print("cache: hit")
        results, n_vector, n_fts = fuse_and_rank(conn, cached_vectors, query, limit)
    else:
        vector_results = get_vector_candidates(conn, query_emb, limit)
        cache.put(conn, query_emb, vector_results)
        results, n_vector, n_fts = fuse_and_rank(conn, vector_results, query, limit)

    conn.close()

    if verbose and n_fts:
        console.print(f"[dim]hybrid: {n_vector} vector + {n_fts} fts[/dim]")

    if not results:
        console.print("[dim]no results[/dim]")
        return

    console.print()
    for i, r in enumerate(results, 1):
        location = f"{r['doc']}.md:{r['start']}-{r['end']}"
        if verbose:
            console.print(
                f"[bold]{location}[/bold]  [dim]{r['display_score']:.2f}[/dim]"
            )
        else:
            console.print(f"[bold]{location}[/bold]")
        if verbose:
            tags = []
            if r.get("fts_hit"):
                tags.append("fts")
            if r.get("is_idx"):
                tags.append("idx")
            tag_str = "  ".join(tags)
            console.print(
                f"  [dim]vec[/dim] {r['embed_score']:.2f}"
                f". [dim]rrf[/dim] {r['rrf_score']:.4f}"
                f"  {tag_str}"
            )
        console.print(f"  [dim]{_snippet(r['text'], r.get('fts_terms', []))}[/dim]")
        console.print()


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
        "-n", "--limit", type=int, default=10, help="Max results (default: 10)"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Delete DB and extracted files"
    )
    parser.add_argument(
        "--clear-cache", action="store_true", help="Clear semantic search cache"
    )
    parser.add_argument(
        "--reset-context",
        action="store_true",
        help="Delete generated contexts; next run regenerates them",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show pipeline scores"
    )
    args = parser.parse_args()

    console.print(SOVA_ASCII)

    if args.search:
        search_semantic(args.search, args.limit, verbose=args.verbose)
        return

    if args.reset:
        if DB_PATH.exists():
            DB_PATH.unlink()
            console.print(f"deleted: {DB_PATH.name}")
        if DATA_DIR.exists():
            for md in DATA_DIR.glob("*.md"):
                md.unlink()
                console.print(f"deleted: {md.name}")
        console.print("reset complete")
        return

    if args.clear_cache:
        cache = get_cache()
        cache.clear()
        console.print("cache cleared")
        return

    if args.reset_context:
        if not DB_PATH.exists():
            console.print("no database")
            return
        conn = init_db()
        deleted = conn.execute("SELECT COUNT(*) FROM chunk_contexts").fetchone()[0]
        conn.execute("DELETE FROM chunk_contexts")
        conn.commit()
        conn.close()
        console.print(f"deleted {deleted} contexts; embeddings kept")
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
        console.print()  # newline after progress bar

    # Invalidate cached search results since new chunks were indexed.
    get_cache().clear(conn)

    conn.close()
    console.print()
    if interrupted:
        console.print(f"interrupted after {fmt_duration(time.time() - start_time)}")
    else:
        console.print(f"done in {fmt_duration(time.time() - start_time)}")
    show_stats()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\ninterrupted")
        sys.exit(130)
