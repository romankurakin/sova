"""Command-line interface."""

import hashlib
import re
import signal
import sqlite3
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from typer._click import exceptions as click_exceptions
from typer.core import TyperGroup

from sova import config, projects
from sova.audit import Finding, audit_database
from sova.cache import get_cache
from sova.db import (
    SCHEMA_VERSION,
    connect_readonly,
    embedding_to_blob,
    get_doc_status,
    get_meta,
    init_db,
    quantize_vectors,
    set_meta,
)
from sova.extract import (
    add_section_paths,
    chunk_text,
    extract_pdf,
    find_docs,
    find_section,
    parse_sections,
)
from sova.index_text import contextualized_text
from sova.llama_client import (
    CONTEXT_SYSTEM_PROMPT,
    CONTEXT_USER_PROMPT,
    QUERY_TASK,
    check_servers,
    generate_context,
    get_embeddings_batch,
    get_model_status,
    get_query_embedding,
    get_service_diagnostics,
    get_token_counts_batch,
    is_model_cached,
    is_service_installed,
    is_service_running,
    run_embedding_canary,
    start_service,
    stop_server,
)
from sova.search import (
    compute_candidates,
    fuse_and_rank,
    get_vector_candidates,
    is_index_like,
)
from sova.ui import (
    close_output,
    configure_output,
    emit,
    fmt_duration,
    is_json_output,
    make_table,
    progress,
    render_table,
    report_error,
    status,
)
from sova.ui import (
    result as report_result,
)
from sova.ui import (
    runtime as report_runtime,
)
from sova.ui import (
    scope as report_scope,
)


def _display_path(path: Path) -> str:
    """Render path with ~ for home-relative locations."""
    home = Path.home()
    try:
        rel = path.relative_to(home)
        return "~" if str(rel) == "." else f"~/{rel}"
    except ValueError:
        return str(path)


def _file_signature(path: Path) -> str:
    """Return a stable source fingerprint without loading the file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


# Keep embedding work bounded so Python memory and llama-server lifetime stay.
# controlled on large indexes.
_EMBED_WINDOW_CHUNKS = 256
_EMBED_RECYCLE_CHUNKS = 1800
_EMBED_RECYCLE_PAUSE_S = 2.0
_EMBED_RECYCLE_CANARY_REQUESTS = 4
_EMBED_PREFIX_VERSION = "chunk-prefix.v2"
_CONTEXT_RETRY_ATTEMPTS = 2
_CONTEXT_RECYCLE_PAUSE_S = 2.0
_RUNTIME_REFRESH_S = 20.0

_META_CONTEXT_SIG = "pipeline.context.signature"
_META_EMBED_SIG = "pipeline.embedding.signature"
_META_CHUNK_SIG = "pipeline.chunk.signature"
_META_SOURCE_SIG_PREFIX = "source.extract.signature."


@dataclass(frozen=True)
class _IndexSignatureState:
    force_rebuild_context: bool
    force_rebuild_embed: bool
    context_sig: str
    embed_sig: str
    chunk_sig: str


@dataclass(frozen=True)
class _PreparedSource:
    name: str
    markdown_path: Path
    source_signature: str


def _source_checkpoint_key(name: str) -> str:
    return f"{_META_SOURCE_SIG_PREFIX}{name}"


def _write_text_atomic(path: Path, text: str) -> None:
    """Replace a generated text artifact only after its full write succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _preview(text: str, max_chars: int = 48) -> str:
    """Short single-line preview for headers."""
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 1].rstrip() + "…"


def fmt_size(size_bytes: int) -> str:
    if size_bytes == 0:
        return "-"
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def _progress_pct(done: int, total: int) -> int:
    """Single percent calculator for status labels."""
    if total <= 0:
        return 0
    if done >= total:
        return 100
    return min(99, round((done / total) * 100))


def _format_error_chain(exc: BaseException) -> str:
    """Render exception and cause chain in one concise line."""
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = str(current).strip() or current.__class__.__name__
        if text not in parts:
            parts.append(text)
        if current.__cause__ is not None:
            current = current.__cause__
            continue
        if current.__suppress_context__:
            break
        current = current.__context__
    return " | ".join(parts)


def _report_error_block(
    summary: str,
    *,
    cause: str | None = None,
    action: str | None = None,
    detail: str | None = None,
) -> None:
    report_error(summary, cause=cause, action=action, detail=detail)


def _infer_service_name(message: str) -> str | None:
    low = message.lower()
    if ":8081" in low or "embedding" in low:
        return "embedding"
    if ":8083" in low or "context" in low or "chat" in low:
        return "chat"
    return None


def _classify_error(message: str) -> tuple[str, str | None, str | None]:
    low = message.lower()
    if "index database not found" in low:
        return (
            "database not ready",
            message,
            "run: sova index <project>",
        )
    if "memory hard-cap exceeded" in low:
        return (
            "model does not fit current memory budget",
            message,
            "close extra apps and retry, or reduce reserve for this mode",
        )
    if "server not reachable at" in low:
        svc = _infer_service_name(message)
        summary = f"{svc} server unavailable" if svc else "model server unavailable"
        return (
            summary,
            message,
            "ensure services are installed and loaded (run sova-install), then retry",
        )
    if "server timeout" in low:
        return (
            "model server timed out",
            message,
            "retry; if this repeats, lower concurrent system load",
        )
    if _is_likely_oom(message):
        return (
            "model ran out of memory",
            message,
            "close extra apps or increase reserve for this mode",
        )
    return ("operation failed", message, None)


def _is_likely_oom(message: str) -> bool:
    m = message.lower()
    markers = (
        "out of memory",
        "outofmemory",
        "kiogpucommandbuffercallbackerroroutofmemory",
        "oom",
        "failed to allocate",
        "insufficient memory",
    )
    return any(marker in m for marker in markers)


def _report_error(exc: BaseException) -> None:
    """Print a structured error block with actionable guidance."""
    text = re.sub(r"\s+", " ", _format_error_chain(exc)).strip()
    summary, cause, action = _classify_error(text)
    _report_error_block(summary, cause=cause, action=action)


def _report_service_diag(url: str) -> None:
    name = (
        "embedding"
        if url == config.EMBEDDING_SERVER_URL
        else "chat"
        if url == config.CONTEXT_SERVER_URL
        else "service"
    )
    diag = get_service_diagnostics(url)
    if diag:
        emit(
            "service_diagnostic",
            f"{name}: {diag}",
            level="warning",
            data={"service": name, "diagnostic": diag},
        )


def _report_relevant_service_diags(
    exc: BaseException,
    mode: str,
) -> None:
    text = _format_error_chain(exc).lower()
    urls: list[str] = []
    if "8081" in text or "embedding" in text:
        urls.append(config.EMBEDDING_SERVER_URL)
    if "8083" in text or "context" in text or "chat" in text:
        urls.append(config.CONTEXT_SERVER_URL)
    if not urls:
        if mode == "search":
            urls = [config.EMBEDDING_SERVER_URL]
        elif mode == "index_context":
            urls = [config.CONTEXT_SERVER_URL]
        elif mode == "index_embed":
            urls = [config.EMBEDDING_SERVER_URL]
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        _report_service_diag(url)


def _report_phase_runtime(phase: str, service_name: str, mode: str = "index") -> None:
    """Update runtime telemetry without adding permanent terminal lines."""
    del phase, service_name
    try:
        effective = config.get_effective_available_gib()
        reserve = config.get_memory_reserve_gib(mode)
        budget_now = max(0.0, round(effective - reserve, 2))
        report_runtime(memory_headroom_gib=budget_now)
    except OSError, RuntimeError, TypeError, ValueError:
        pass


def _make_runtime_reporter(
    phase: str, service_name: str, mode: str = "index"
) -> Callable[[bool], None]:
    """Return throttled runtime reporter for long-running loops."""
    last_report = 0.0

    def tick(force: bool = False) -> None:
        nonlocal last_report
        now = time.monotonic()
        if force or (now - last_report) >= _RUNTIME_REFRESH_S:
            _report_phase_runtime(phase, service_name, mode=mode)
            last_report = now

    return tick


def _prepare_source(
    name: str,
    pdf_path: Path | None,
    md_path: Path | None,
    conn: sqlite3.Connection,
) -> _PreparedSource | None:
    """Extract one source while model services remain fully unloaded."""
    status("Preparing", phase="prepare", item=name)

    source_path = pdf_path or md_path
    if source_path is None or not source_path.exists():
        _report_error_block(
            "source document is unavailable",
            cause=name,
            action="restore the source document and retry",
        )
        return None
    source_signature = _file_signature(source_path)
    document_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(documents)")
    }
    stored_source_signature: str | None = None
    stored_extract_signature = get_meta(conn, _source_checkpoint_key(name))
    if "source_signature" in document_columns:
        signature_row = conn.execute(
            "SELECT source_signature FROM documents WHERE name = ?", (name,)
        ).fetchone()
        if signature_row and signature_row[0]:
            stored_source_signature = str(signature_row[0])

    effective_extract_signature = stored_extract_signature or stored_source_signature
    needs_extract = bool(
        pdf_path
        and (
            not md_path
            or not md_path.exists()
            or effective_extract_signature != source_signature
        )
    )
    if needs_extract:
        assert pdf_path is not None
        try:
            start = time.time()
            markdown = extract_pdf(pdf_path)
            data_dir = config.get_data_dir()
            data_dir.mkdir(parents=True, exist_ok=True)
            md_path = data_dir / f"{name}.md"
            _write_text_atomic(md_path, markdown)
            set_meta(conn, _source_checkpoint_key(name), source_signature)
            conn.commit()
            lines = len(markdown.splitlines())
            status(
                f"Prepared {lines:,} lines in "
                f"{fmt_duration(time.time() - start).strip()}",
                phase="prepare",
                item=name,
            )
        except (OSError, RuntimeError, ValueError) as e:
            _report_error_block(
                "extract failed",
                cause=f"{name}: {e}",
                action="verify PDF is readable and retry",
            )
            return None

    elif pdf_path and stored_extract_signature != source_signature:
        # Migrate a completed pre-checkpoint extraction without repeating OCR.
        set_meta(conn, _source_checkpoint_key(name), source_signature)
        conn.commit()

    assert md_path is not None and md_path.exists()
    return _PreparedSource(name, md_path, source_signature)


def _tokenize_doc(
    source: _PreparedSource,
    conn: sqlite3.Connection,
    chunk_signature: str | None = None,
) -> tuple[int, list[dict], list[dict]]:
    """Create exact model-token chunks and synchronize their durable rows."""
    name = source.name
    md_path = source.markdown_path
    source_signature = source.source_signature
    chunk_signature = chunk_signature or _chunk_pipeline_signature()
    status("Tokenizing", phase="tokenize", item=name)
    document_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(documents)")
    }

    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    sections = parse_sections(lines)
    chunks = chunk_text(lines, count_tokens_batch=get_token_counts_batch)

    row = conn.execute("SELECT id FROM documents WHERE name = ?", (name,)).fetchone()
    if row:
        doc_id = row[0]
        if {"source_signature", "chunk_signature"} <= document_columns:
            conn.execute(
                """
                UPDATE documents
                SET path = ?, line_count = ?, expected_chunks = ?,
                    source_signature = ?, chunk_signature = ?
                WHERE id = ?
                """,
                (
                    str(md_path),
                    len(lines),
                    len(chunks),
                    source_signature,
                    chunk_signature,
                    doc_id,
                ),
            )
        elif "source_signature" in document_columns:
            conn.execute(
                """
                UPDATE documents
                SET path = ?, line_count = ?, expected_chunks = ?, source_signature = ?
                WHERE id = ?
                """,
                (str(md_path), len(lines), len(chunks), source_signature, doc_id),
            )
        else:
            conn.execute(
                """
                UPDATE documents
                SET path = ?, line_count = ?, expected_chunks = ?
                WHERE id = ?
                """,
                (str(md_path), len(lines), len(chunks), doc_id),
            )
    else:
        if {"source_signature", "chunk_signature"} <= document_columns:
            cursor = conn.execute(
                """
                INSERT INTO documents
                    (name, path, line_count, expected_chunks,
                     source_signature, chunk_signature)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    str(md_path),
                    len(lines),
                    len(chunks),
                    source_signature,
                    chunk_signature,
                ),
            )
        elif "source_signature" in document_columns:
            cursor = conn.execute(
                """
                INSERT INTO documents
                    (name, path, line_count, expected_chunks, source_signature)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, str(md_path), len(lines), len(chunks), source_signature),
            )
        else:
            cursor = conn.execute(
                """
                INSERT INTO documents (name, path, line_count, expected_chunks)
                VALUES (?, ?, ?, ?)
                """,
                (name, str(md_path), len(lines), len(chunks)),
            )
        doc_id = cursor.lastrowid
        assert doc_id is not None

    # Sync sections by start_line to keep section IDs stable across re-runs.
    existing_section_rows = conn.execute(
        """
        SELECT id, start_line, end_line, title, level
        FROM sections
        WHERE doc_id = ?
        ORDER BY id
        """,
        (doc_id,),
    ).fetchall()
    existing_sections_by_start: dict[int, tuple[int, int | None, str, int]] = {}
    stale_section_ids: list[int] = []
    for row_data in existing_section_rows:
        section_id, start_line, end_line, title, level = row_data
        if start_line in existing_sections_by_start:
            stale_section_ids.append(section_id)
            continue
        existing_sections_by_start[start_line] = (section_id, end_line, title, level)

    planned_section_starts: set[int] = set()
    for s in sections:
        start_line = s["start_line"]
        if start_line in planned_section_starts:
            continue
        planned_section_starts.add(start_line)
        existing_section = existing_sections_by_start.get(start_line)
        if existing_section is None:
            conn.execute(
                """
                INSERT INTO sections (doc_id, title, level, start_line, end_line)
                VALUES (?, ?, ?, ?, ?)
                """,
                (doc_id, s["title"], s["level"], start_line, s["end_line"]),
            )
            continue
        section_id, end_line, title, level = existing_section
        if end_line != s["end_line"] or title != s["title"] or level != s["level"]:
            conn.execute(
                """
                UPDATE sections
                SET title = ?, level = ?, end_line = ?
                WHERE id = ?
                """,
                (s["title"], s["level"], s["end_line"], section_id),
            )

    stale_section_ids.extend(
        section_id
        for start_line, (
            section_id,
            _end,
            _title,
            _level,
        ) in existing_sections_by_start.items()
        if start_line not in planned_section_starts
    )
    if stale_section_ids:
        placeholders = ",".join("?" * len(stale_section_ids))
        conn.execute(
            f"DELETE FROM sections WHERE id IN ({placeholders})",
            tuple(stale_section_ids),
        )

    section_rows = conn.execute(
        "SELECT id, start_line FROM sections WHERE doc_id = ?", (doc_id,)
    ).fetchall()
    section_ids = {r[1]: r[0] for r in section_rows}

    existing_rows = conn.execute(
        """
        SELECT id, start_line, end_line, word_count, text, section_id,
               section_path, search_text, is_index
        FROM chunks
        WHERE doc_id = ?
        ORDER BY id
        """,
        (doc_id,),
    ).fetchall()
    existing_by_start: dict[
        int, tuple[int, int, int, str, int | None, str, str, int]
    ] = {}
    duplicate_ids: list[int] = []
    for row_data in existing_rows:
        (
            chunk_id,
            start_line,
            end_line,
            word_count,
            text_value,
            section_id,
            section_path,
            search_text,
            is_idx,
        ) = row_data
        if start_line in existing_by_start:
            duplicate_ids.append(chunk_id)
            continue
        existing_by_start[start_line] = (
            chunk_id,
            end_line,
            word_count,
            text_value,
            section_id,
            str(section_path),
            str(search_text),
            is_idx,
        )

    planned_starts: set[int] = set()
    changed_chunk_ids: list[int] = []
    for chunk in chunks:
        start_line = chunk["start_line"]
        if start_line in planned_starts:
            continue
        planned_starts.add(start_line)
        sec_idx = find_section(sections, start_line)
        sec_line = sections[sec_idx]["start_line"] if sec_idx is not None else None
        sec_id = section_ids.get(sec_line)
        sec_path = str(sections[sec_idx]["path"]) if sec_idx is not None else ""
        is_idx = 1 if is_index_like(chunk["text"]) else 0
        base_search_text = contextualized_text(name, sec_path, chunk["text"])

        existing = existing_by_start.get(start_line)
        if existing is None:
            conn.execute(
                """
                INSERT INTO chunks
                    (doc_id, section_id, section_path, start_line, end_line,
                     word_count, text, search_text, is_index)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    sec_id,
                    sec_path,
                    start_line,
                    chunk["end_line"],
                    chunk["word_count"],
                    chunk["text"],
                    base_search_text,
                    is_idx,
                ),
            )
            continue

        (
            chunk_id,
            end_line,
            word_count,
            text_value,
            old_section_id,
            old_section_path,
            _old_search_text,
            old_is_idx,
        ) = existing
        content_changed = (
            end_line != chunk["end_line"]
            or word_count != chunk["word_count"]
            or text_value != chunk["text"]
            or old_is_idx != is_idx
        )
        retrieval_changed = content_changed or old_section_path != sec_path
        if retrieval_changed:
            conn.execute(
                """
                UPDATE chunks
                SET section_id = ?, section_path = ?, end_line = ?, word_count = ?,
                    text = ?, search_text = ?, is_index = ?, embedding = NULL,
                    embedding_signature = NULL
                WHERE id = ?
                """,
                (
                    sec_id,
                    sec_path,
                    chunk["end_line"],
                    chunk["word_count"],
                    chunk["text"],
                    base_search_text,
                    is_idx,
                    chunk_id,
                ),
            )
            changed_chunk_ids.append(chunk_id)
        elif old_section_id != sec_id:
            conn.execute(
                "UPDATE chunks SET section_id = ? WHERE id = ?",
                (sec_id, chunk_id),
            )

    stale_ids = [
        row[0]
        for start_line, row in existing_by_start.items()
        if start_line not in planned_starts
    ]
    stale_ids.extend(duplicate_ids)
    if stale_ids:
        placeholders = ",".join("?" * len(stale_ids))
        conn.execute(
            f"DELETE FROM chunks WHERE id IN ({placeholders})", tuple(stale_ids)
        )

    if changed_chunk_ids:
        placeholders = ",".join("?" * len(changed_chunk_ids))
        conn.execute(
            f"DELETE FROM chunk_contexts WHERE chunk_id IN ({placeholders})",
            tuple(changed_chunk_ids),
        )

    conn.commit()

    return doc_id, chunks, sections


def _load_prepared_doc(
    conn: sqlite3.Connection, doc_id: int
) -> tuple[list[dict], list[dict]]:
    """Load one prepared document for a model phase, keeping project RAM bounded."""
    chunks = [
        {
            "start_line": int(row[0]),
            "end_line": int(row[1]),
            "word_count": int(row[2]),
            "text": str(row[3]),
        }
        for row in conn.execute(
            """
            SELECT start_line, end_line, word_count, text
            FROM chunks
            WHERE doc_id = ?
            ORDER BY start_line, id
            """,
            (doc_id,),
        ).fetchall()
    ]
    sections = [
        {
            "title": str(row[0]),
            "level": int(row[1]),
            "start_line": int(row[2]),
            "end_line": int(row[3]) if row[3] is not None else None,
        }
        for row in conn.execute(
            """
            SELECT title, level, start_line, end_line
            FROM sections
            WHERE doc_id = ?
            ORDER BY start_line, id
            """,
            (doc_id,),
        ).fetchall()
    ]
    return chunks, add_section_paths(sections)


def _signature(parts: list[str]) -> str:
    raw = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _context_pipeline_signature() -> str:
    return _signature(
        [
            config.CONTEXT_MODEL,
            config.CONTEXT_MODEL_HF_REPO,
            config.CONTEXT_MODEL_HF_FILE,
            CONTEXT_SYSTEM_PROMPT,
            CONTEXT_USER_PROMPT,
            "input:previous-tail=500,target=full,next-head=500,section=breadcrumb",
            "response:json-schema,max_tokens=192,temperature=0,reasoning=low",
            "validation:sentence.v2",
        ]
    )


def _embedding_pipeline_signature() -> str:
    return _signature(
        [
            config.EMBEDDING_MODEL,
            str(config.EMBEDDING_DIM),
            QUERY_TASK,
            _EMBED_PREFIX_VERSION,
        ]
    )


def _chunk_pipeline_signature() -> str:
    return _signature(
        [
            str(config.CHUNK_TARGET_TOKENS),
            config.EMBEDDING_MODEL,
            "structure-token-chunks.v4",
        ]
    )


def _current_tokenized_doc_id(
    conn: sqlite3.Connection,
    source: _PreparedSource,
    chunk_signature: str,
) -> int | None:
    """Return a document only when its durable chunk checkpoint is complete."""
    document_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(documents)")
    }
    if not {"source_signature", "chunk_signature"} <= document_columns:
        return None
    row = conn.execute(
        """
        SELECT id, expected_chunks
        FROM documents
        WHERE name = ? AND source_signature = ? AND chunk_signature = ?
        """,
        (source.name, source.source_signature, chunk_signature),
    ).fetchone()
    if row is None or row[1] is None:
        return None
    doc_id, expected_chunks = int(row[0]), int(row[1])
    actual_chunks = int(
        conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)
        ).fetchone()[0]
    )
    return doc_id if actual_chunks == expected_chunks else None


def _context_work_pending(
    conn: sqlite3.Connection,
    context_signature: str,
    *,
    force_rebuild: bool = False,
) -> bool:
    context_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(chunk_contexts)")
    }
    if "pipeline_signature" in context_columns:
        return bool(
            conn.execute(
                """
                SELECT EXISTS(
                    SELECT 1 FROM chunks c
                    LEFT JOIN chunk_contexts cc ON cc.chunk_id = c.id
                    WHERE cc.chunk_id IS NULL OR TRIM(cc.context) = ''
                       OR cc.pipeline_signature IS NULL
                       OR cc.pipeline_signature <> ?
                )
                """,
                (context_signature,),
            ).fetchone()[0]
        )
    if force_rebuild:
        return bool(conn.execute("SELECT EXISTS(SELECT 1 FROM chunks)").fetchone()[0])
    return bool(
        conn.execute(
            """
            SELECT EXISTS(
                SELECT 1 FROM chunks c
                LEFT JOIN chunk_contexts cc ON cc.chunk_id = c.id
                WHERE cc.chunk_id IS NULL OR TRIM(cc.context) = ''
            )
            """
        ).fetchone()[0]
    )


def _embedding_work_pending(
    conn: sqlite3.Connection,
    embedding_signature: str,
    *,
    force_rebuild: bool = False,
) -> bool:
    chunk_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(chunks)")
    }
    if "embedding_signature" in chunk_columns:
        return bool(
            conn.execute(
                """
                SELECT EXISTS(
                    SELECT 1 FROM chunks
                    WHERE embedding IS NULL OR embedding_signature IS NULL
                       OR embedding_signature <> ?
                )
                """,
                (embedding_signature,),
            ).fetchone()[0]
        )
    if force_rebuild:
        return bool(conn.execute("SELECT EXISTS(SELECT 1 FROM chunks)").fetchone()[0])
    return bool(
        conn.execute(
            "SELECT EXISTS(SELECT 1 FROM chunks WHERE embedding IS NULL)"
        ).fetchone()[0]
    )


def _sync_index_signatures(conn: sqlite3.Connection) -> _IndexSignatureState:
    current_context = _context_pipeline_signature()
    current_embed = _embedding_pipeline_signature()
    current_chunk = _chunk_pipeline_signature()

    stored_context = get_meta(conn, _META_CONTEXT_SIG)
    stored_embed = get_meta(conn, _META_EMBED_SIG)
    stored_chunk = get_meta(conn, _META_CHUNK_SIG)

    context_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(chunk_contexts)")
    }
    chunk_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(chunks)")
    }
    document_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(documents)")
    }

    # Schema v4 only had a project-wide completion marker. A completed v4
    # index can be upgraded without paying for tokenization again.
    if "chunk_signature" in document_columns and stored_chunk == current_chunk:
        conn.execute(
            """
            UPDATE documents
            SET chunk_signature = ?
            WHERE chunk_signature IS NULL
              AND expected_chunks IS NOT NULL
              AND expected_chunks = (
                  SELECT COUNT(*) FROM chunks WHERE chunks.doc_id = documents.id
              )
            """,
            (current_chunk,),
        )
        conn.commit()

    has_contexts = conn.execute(
        "SELECT EXISTS(SELECT 1 FROM chunk_contexts)"
    ).fetchone()[0]
    has_embeddings = conn.execute(
        "SELECT EXISTS(SELECT 1 FROM chunks WHERE embedding IS NOT NULL)"
    ).fetchone()[0]
    has_chunks = conn.execute("SELECT EXISTS(SELECT 1 FROM chunks)").fetchone()[0]

    if "pipeline_signature" in context_columns:
        context_changed = bool(
            conn.execute(
                """
                SELECT EXISTS(
                    SELECT 1 FROM chunk_contexts
                    WHERE TRIM(context) = ''
                       OR pipeline_signature IS NULL
                       OR pipeline_signature <> ?
                )
                """,
                (current_context,),
            ).fetchone()[0]
        )
    else:
        context_changed = bool(has_contexts) and stored_context != current_context

    if "embedding_signature" in chunk_columns:
        embed_changed = bool(
            conn.execute(
                """
                SELECT EXISTS(
                    SELECT 1 FROM chunks
                    WHERE embedding IS NOT NULL
                      AND (embedding_signature IS NULL OR embedding_signature <> ?)
                )
                """,
                (current_embed,),
            ).fetchone()[0]
        )
    else:
        embed_changed = bool(has_embeddings) and stored_embed != current_embed

    unknown_chunk_checkpoint = False
    if "chunk_signature" in document_columns:
        chunk_changed = bool(
            conn.execute(
                """
                SELECT EXISTS(
                    SELECT 1 FROM documents d
                    WHERE d.chunk_signature IS NOT NULL
                      AND d.chunk_signature <> ?
                      AND EXISTS(SELECT 1 FROM chunks c WHERE c.doc_id = d.id)
                )
                """,
                (current_chunk,),
            ).fetchone()[0]
        )
        unknown_chunk_checkpoint = bool(
            conn.execute(
                """
                SELECT EXISTS(
                    SELECT 1 FROM documents d
                    WHERE d.chunk_signature IS NULL
                      AND EXISTS(SELECT 1 FROM chunks c WHERE c.doc_id = d.id)
                )
                """
            ).fetchone()[0]
        )
    else:
        chunk_changed = bool(has_chunks) and stored_chunk != current_chunk

    if context_changed:
        emit(
            "pipeline_changed",
            "Context pipeline changed. Refreshing contexts and embeddings.",
            phase="prepare",
        )
    elif embed_changed:
        emit(
            "pipeline_changed",
            "Embedding pipeline changed. Refreshing embeddings.",
            phase="prepare",
        )

    if chunk_changed:
        emit(
            "pipeline_changed",
            "Chunking changed. Synchronizing document chunks.",
            phase="prepare",
        )

    final_markers_incomplete = bool(
        (has_contexts and stored_context != current_context)
        or (has_embeddings and stored_embed != current_embed)
        or (has_chunks and stored_chunk != current_chunk)
    )
    if (
        (final_markers_incomplete or unknown_chunk_checkpoint)
        and not context_changed
        and not embed_changed
        and not chunk_changed
    ):
        emit(
            "pipeline_resuming",
            "Resuming interrupted index. Reusing completed work.",
            phase="prepare",
        )

    return _IndexSignatureState(
        force_rebuild_context=context_changed,
        force_rebuild_embed=(embed_changed or context_changed),
        context_sig=current_context,
        embed_sig=current_embed,
        chunk_sig=current_chunk,
    )


def _commit_index_signatures(
    conn: sqlite3.Connection, signature_state: _IndexSignatureState
) -> None:
    set_meta(conn, _META_CONTEXT_SIG, signature_state.context_sig)
    set_meta(conn, _META_EMBED_SIG, signature_state.embed_sig)
    set_meta(conn, _META_CHUNK_SIG, signature_state.chunk_sig)
    conn.commit()


def _prune_missing_documents(conn: sqlite3.Connection, source_names: set[str]) -> int:
    """Remove indexed rows whose source document is no longer in the project."""
    existing = {
        str(row[0]) for row in conn.execute("SELECT name FROM documents").fetchall()
    }
    stale = sorted(existing - source_names)
    if stale:
        placeholders = ",".join("?" for _ in stale)
        conn.execute(f"DELETE FROM documents WHERE name IN ({placeholders})", stale)

    has_index_meta = bool(
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'index_meta'"
        ).fetchone()
    )
    checkpoint_rows = (
        conn.execute(
            "SELECT key FROM index_meta WHERE key LIKE ?",
            (f"{_META_SOURCE_SIG_PREFIX}%",),
        ).fetchall()
        if has_index_meta
        else []
    )
    stale_checkpoint_keys = [
        str(row[0])
        for row in checkpoint_rows
        if str(row[0])[len(_META_SOURCE_SIG_PREFIX) :] not in source_names
    ]
    if stale_checkpoint_keys:
        placeholders = ",".join("?" for _ in stale_checkpoint_keys)
        conn.execute(
            f"DELETE FROM index_meta WHERE key IN ({placeholders})",
            stale_checkpoint_keys,
        )

    if not stale and not stale_checkpoint_keys:
        return 0
    conn.commit()
    if stale:
        emit(
            "sources_reconciled",
            f"Removed {len(stale)} stale indexed document(s)",
            phase="prepare",
            data={"documents": stale},
        )
    return len(stale)


def _make_progress_reporter(
    name: str, total: int, *, item: str | None = None
) -> Callable[[int], None]:
    """Publish progress; each renderer decides how often and where to draw it."""

    def publish(done: int) -> None:
        progress(name, done, total, item=item, unit="chunks")

    return publish


def _generate_contexts(
    name: str,
    doc_id: int,
    chunks: list[dict],
    sections: list[dict],
    conn: sqlite3.Connection,
    force_rebuild_context: bool = False,
    runtime_tick: Callable[[bool], None] | None = None,
) -> None:
    """Generate context summaries for chunks that don't have them yet."""
    chunk_rows = conn.execute(
        "SELECT id, start_line FROM chunks WHERE doc_id = ?", (doc_id,)
    ).fetchall()
    chunk_id_by_start = {r[1]: r[0] for r in chunk_rows}

    context_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(chunk_contexts)")
    }
    chunk_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(chunks)")}
    context_sig = _context_pipeline_signature()
    if "pipeline_signature" in context_columns:
        existing_contexts = {
            r[0]
            for r in conn.execute(
                """
                SELECT chunk_id
                FROM chunk_contexts
                WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?)
                  AND TRIM(context) <> ''
                  AND pipeline_signature = ?
                """,
                (doc_id, context_sig),
            ).fetchall()
        }
    elif force_rebuild_context:
        existing_contexts = set()
    else:
        existing_contexts = {
            r[0]
            for r in conn.execute(
                """
                SELECT chunk_id
                FROM chunk_contexts
                WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?)
                  AND TRIM(context) <> ''
                """,
                (doc_id,),
            ).fetchall()
        }
    chunks_needing_context = []
    # Protect against duplicate start_line entries in chunk lists. This keeps.
    # context generation idempotent across interrupted/retried runs.
    planned_chunk_ids = set(existing_contexts)
    for i, chunk in enumerate(chunks):
        chunk_id = chunk_id_by_start.get(chunk["start_line"])
        if chunk_id is None:
            continue
        if chunk_id in planned_chunk_ids:
            continue
        chunks_needing_context.append((i, chunk, chunk_id))
        planned_chunk_ids.add(chunk_id)

    if not chunks_needing_context:
        progress("context", len(chunks), len(chunks), item=name, unit="chunks")
        return

    try:
        total = len(chunks_needing_context)
        absolute_total = len(chunk_id_by_start)
        absolute_done_base = absolute_total - total
        emit_progress = _make_progress_reporter("context", absolute_total, item=name)
        for done, (i, chunk, chunk_id) in enumerate(chunks_needing_context, start=1):
            sec_idx = find_section(sections, chunk["start_line"])
            sec_title = sections[sec_idx]["path"] if sec_idx is not None else None
            prev_text = chunks[i - 1]["text"] if i > 0 else ""
            next_text = chunks[i + 1]["text"] if i + 1 < len(chunks) else ""

            ctx = ""
            for attempt in range(_CONTEXT_RETRY_ATTEMPTS):
                try:
                    candidate = generate_context(
                        name, sec_title, chunk["text"], prev_text, next_text
                    )
                    ctx = candidate.strip()
                    if not ctx:
                        raise RuntimeError("context model returned empty content")
                    break
                except Exception:
                    if attempt == (_CONTEXT_RETRY_ATTEMPTS - 1):
                        raise
                    stop_server(config.CONTEXT_SERVER_URL)
                    time.sleep(_CONTEXT_RECYCLE_PAUSE_S)

            assert ctx

            if "pipeline_signature" in context_columns:
                conn.execute(
                    """
                    INSERT INTO chunk_contexts
                        (chunk_id, context, model, pipeline_signature)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        context = excluded.context,
                        model = excluded.model,
                        pipeline_signature = excluded.pipeline_signature
                    """,
                    (chunk_id, ctx, config.CONTEXT_MODEL, context_sig),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO chunk_contexts (chunk_id, context, model)
                    VALUES (?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        context = excluded.context,
                        model = excluded.model
                    """,
                    (chunk_id, ctx, config.CONTEXT_MODEL),
                )
            search_text = contextualized_text(
                name, sec_title, chunk["text"], context=ctx
            )
            if {"embedding_signature", "search_text"} <= chunk_columns:
                conn.execute(
                    """
                    UPDATE chunks
                    SET search_text = ?, embedding = NULL, embedding_signature = NULL
                    WHERE id = ?
                    """,
                    (search_text, chunk_id),
                )
            elif "embedding_signature" in chunk_columns:
                conn.execute(
                    """
                    UPDATE chunks
                    SET embedding = NULL, embedding_signature = NULL
                    WHERE id = ?
                    """,
                    (chunk_id,),
                )
            else:
                conn.execute(
                    "UPDATE chunks SET embedding = NULL WHERE id = ?",
                    (chunk_id,),
                )
            conn.commit()
            emit_progress(absolute_done_base + done)
            if runtime_tick:
                runtime_tick(False)

    except Exception as e:
        conn.rollback()
        raise RuntimeError(f"context generation failed for {name}: {e}") from e


def _embed_doc(
    name: str,
    doc_id: int,
    chunks: list[dict],
    sections: list[dict],
    conn: sqlite3.Connection,
    force_rebuild_embed: bool = False,
    runtime_tick: Callable[[bool], None] | None = None,
) -> None:
    """Embed all chunks that are missing embeddings."""
    chunk_rows = conn.execute(
        "SELECT id, start_line FROM chunks WHERE doc_id = ?", (doc_id,)
    ).fetchall()
    chunk_id_by_start = {r[1]: r[0] for r in chunk_rows}

    context_map: dict[int, str] = {
        r[0]: r[1]
        for r in conn.execute(
            "SELECT cc.chunk_id, cc.context FROM chunk_contexts cc JOIN chunks c ON cc.chunk_id = c.id WHERE c.doc_id = ?",
            (doc_id,),
        ).fetchall()
    }

    chunk_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(chunks)")}
    embed_sig = _embedding_pipeline_signature()
    if "embedding_signature" in chunk_columns:
        embedded_ids = {
            r[0]
            for r in conn.execute(
                """
                SELECT id FROM chunks
                WHERE doc_id = ? AND embedding IS NOT NULL
                  AND embedding_signature = ?
                """,
                (doc_id, embed_sig),
            ).fetchall()
        }
    elif force_rebuild_embed:
        embedded_ids = set()
    else:
        embedded_ids = {
            r[0]
            for r in conn.execute(
                "SELECT id FROM chunks WHERE doc_id = ? AND embedding IS NOT NULL",
                (doc_id,),
            ).fetchall()
        }
    pending_embed = []
    for i, chunk in enumerate(chunks):
        chunk_id = chunk_id_by_start.get(chunk["start_line"])
        if chunk_id is not None and chunk_id not in embedded_ids:
            pending_embed.append((i, chunk, chunk_id))

    if not pending_embed:
        progress("embed", len(chunks), len(chunks), item=name, unit="chunks")
        return

    try:
        total = len(pending_embed)
        absolute_total = len(chunks)
        absolute_done_base = len(embedded_ids)
        emit_progress = _make_progress_reporter("embed", absolute_total, item=name)
        done = 0
        embedded_since_recycle = 0

        for window_start in range(0, total, _EMBED_WINDOW_CHUNKS):
            window = pending_embed[window_start : window_start + _EMBED_WINDOW_CHUNKS]
            window_texts: list[str] = []
            window_chunk_ids: list[int] = []

            for i, chunk, chunk_id in window:
                sec_idx = find_section(sections, chunk["start_line"])
                sec_title = sections[sec_idx]["path"] if sec_idx is not None else None
                llm_ctx = context_map.get(chunk_id)
                window_texts.append(
                    contextualized_text(name, sec_title, chunk["text"], context=llm_ctx)
                )
                window_chunk_ids.append(chunk_id)

            remaining_texts = list(window_texts)
            remaining_chunk_ids = list(window_chunk_ids)
            window_done = 0

            for attempt in range(2):
                attempt_done = 0

                def _persist_batch(
                    batch_indices: list[int],
                    batch_embeddings: list[list[float]],
                    _stats: dict[str, float | int],
                    remaining_chunk_ids: list[int] = remaining_chunk_ids,
                ) -> None:
                    nonlocal attempt_done, window_done, done
                    rows = [
                        (
                            embedding_to_blob(emb),
                            remaining_chunk_ids[batch_idx],
                        )
                        for batch_idx, emb in zip(batch_indices, batch_embeddings)
                    ]
                    if "embedding_signature" in chunk_columns:
                        conn.executemany(
                            """
                            UPDATE chunks
                            SET embedding = ?, embedding_signature = ?
                            WHERE id = ?
                            """,
                            [(blob, embed_sig, chunk_id) for blob, chunk_id in rows],
                        )
                    else:
                        conn.executemany(
                            "UPDATE chunks SET embedding = ? WHERE id = ?",
                            rows,
                        )
                    conn.commit()

                    batch_count = len(rows)
                    attempt_done += batch_count
                    window_done += batch_count
                    done += batch_count
                    emit_progress(absolute_done_base + done)
                    if runtime_tick:
                        runtime_tick(False)

                try:
                    get_embeddings_batch(remaining_texts, on_batch=_persist_batch)
                    break
                except Exception:
                    if attempt_done > 0:
                        remaining_texts = remaining_texts[attempt_done:]
                        remaining_chunk_ids = remaining_chunk_ids[attempt_done:]
                    if attempt == 1:
                        raise
                    stop_server(config.EMBEDDING_SERVER_URL)
                    time.sleep(_EMBED_RECYCLE_PAUSE_S)
                    run_embedding_canary(requests=_EMBED_RECYCLE_CANARY_REQUESTS)

            embedded_since_recycle += window_done

            has_more = (window_start + len(window)) < total
            if has_more and embedded_since_recycle >= _EMBED_RECYCLE_CHUNKS:
                stop_server(config.EMBEDDING_SERVER_URL)
                time.sleep(_EMBED_RECYCLE_PAUSE_S)
                run_embedding_canary(requests=_EMBED_RECYCLE_CANARY_REQUESTS)
                embedded_since_recycle = 0

    except Exception as e:
        conn.rollback()
        raise RuntimeError(f"embedding failed for {name}: {e}") from e


def _doc_status_label(status: dict) -> str:
    """Return a human-readable status label for a document."""
    chunks = status.get("chunks", 0)
    if not chunks:
        return "pending"
    total = status.get("expected") or chunks
    ctx = status.get("contextualized", 0)
    embedded = status.get("embedded", 0)
    # All done.
    if status.get("complete"):
        return "ready"
    # Context generation in progress.
    if ctx < total:
        pct = _progress_pct(ctx, total)
        return f"context {pct}%"
    # Context done, embedding in progress.
    pct = _progress_pct(embedded, total)
    return f"embed {pct}%"


def list_docs(docs: list[dict] | None = None) -> None:
    """List all documents and their indexing status."""
    if docs is None:
        docs = find_docs()
    db_path = config.get_db_path()

    conn = None
    if db_path.exists():
        conn = connect_readonly()

    table = make_table()
    table.add_column("Name")
    table.add_column("Size", justify="right")
    table.add_column("Status", justify="right")

    ready_count = 0
    context_count = 0
    embedding_count = 0
    expected_count = 0
    context_signature = _context_pipeline_signature()
    embedding_signature = _embedding_pipeline_signature()
    for d in docs:
        doc_status = (
            get_doc_status(
                conn,
                d["name"],
                context_signature=context_signature,
                embedding_signature=embedding_signature,
            )
            if conn
            else {}
        )
        label = _doc_status_label(doc_status)
        if label == "ready":
            ready_count += 1
        context_count += int(doc_status.get("contextualized", 0))
        embedding_count += int(doc_status.get("embedded", 0))
        expected_count += int(doc_status.get("expected") or 0)
        table.add_row(d["name"], fmt_size(d["size"]), label)

    if conn:
        conn.close()

    render_table(table)
    progress_parts = [f"{ready_count}/{len(docs)} documents ready"]
    if expected_count:
        progress_parts.extend(
            [
                f"{context_count}/{expected_count} contexts",
                f"{embedding_count}/{expected_count} embeddings",
            ]
        )
    emit(
        "list_completed",
        "  ".join(progress_parts),
        data={
            "ready": ready_count,
            "documents": len(docs),
            "contexts": context_count,
            "embeddings": embedding_count,
            "chunks": expected_count,
        },
    )


def search_semantic(
    query: str,
    limit: int = 10,
    verbose: bool = False,
) -> None:
    """Perform semantic search and display results."""
    if not config.get_db_path().exists():
        raise RuntimeError("index database not found")

    conn = connect_readonly()
    cache = get_cache()
    try:
        query_emb = get_query_embedding(query)

        # Compute min candidates before checking cache so we only accept cached.
        # results that searched at least as broadly as we need.
        total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        min_candidates = compute_candidates(total_chunks, limit)
        cached_vectors = cache.get(query_emb, min_candidates)
        if cached_vectors:
            status("Cache hit", phase="search")
            vector_results = cached_vectors
        else:
            vector_results = get_vector_candidates(
                conn,
                query_emb,
                limit,
                candidates=min_candidates,
            )
            cache.put(query_emb, vector_results)
        results, n_vector, n_fts = fuse_and_rank(
            conn,
            vector_results,
            query,
            limit,
        )
    finally:
        conn.close()

    if verbose and n_fts:
        emit(
            "search_diagnostic",
            f"Hybrid candidates: {n_vector} vector + {n_fts} full-text",
            phase="search",
            data={"vector_candidates": n_vector, "fts_candidates": n_fts},
        )

    if not results:
        emit("search_completed", "No results", phase="search", data={"count": 0})
        return

    for i, r in enumerate(results, 1):
        path = r.get("path")
        if path:
            shown_path = _display_path(Path(path))
            location = f"{shown_path}:{r['start']}-{r['end']}"
        else:
            location = f"{r['doc']}.md:{r['start']}-{r['end']}"
        diagnostic = None
        if verbose:
            tags = [
                tag
                for tag, present in (
                    ("fts", r.get("fts_hit")),
                    ("idx", r.get("is_idx")),
                )
                if present
            ]
            diagnostic = f"vec {r['embed_score']:.2f}  rrf {r['rrf_score']:.4f}" + (
                f"  {'  '.join(tags)}" if tags else ""
            )
        report_result(
            {
                "rank": i,
                "location": location,
                "document": r["doc"],
                "start_line": r["start"],
                "end_line": r["end"],
                "text": r["text"],
                "score": r["display_score"] if verbose else None,
                "diagnostic": diagnostic,
                "last": i == len(results),
            }
        )


def _activate_project_from_ref(
    project_ref: str,
    allow_create_from_dir: bool = False,
) -> projects.Project:
    """Activate a project by id/name/path. Optionally auto-add missing dirs."""
    if allow_create_from_dir:
        raw_ref = project_ref.strip()
        if (
            raw_ref
            and projects.is_reserved_project_id(raw_ref)
            and "/" not in raw_ref
            and "\\" not in raw_ref
        ):
            reserved = ", ".join(sorted(projects.RESERVED_PROJECT_IDS))
            _report_error_block(
                "project name is reserved",
                cause=f"'{raw_ref}' conflicts with a CLI command ({reserved})",
                action="rename the docs folder and retry indexing",
            )
            sys.exit(2)
        path_ref = Path(project_ref).expanduser()
        if path_ref.exists() and path_ref.is_dir():
            existing = projects.get_project(project_ref)
            if existing is None:
                try:
                    created = projects.add_project(path_ref)
                except ValueError as e:
                    message = str(e)
                    summary = (
                        "project name is reserved"
                        if "reserved project id" in message
                        else "cannot add project"
                    )
                    _report_error_block(
                        summary,
                        cause=message,
                        action="rename the docs folder and retry indexing",
                    )
                    sys.exit(2)
                projects.activate(created, create_dirs=True)
                return created
    project = projects.get_project(project_ref)
    if project is None:
        _report_error_block(
            "project not found",
            cause=project_ref,
            action="run: sova projects",
        )
        sys.exit(1)
    assert project is not None
    projects.activate(project, create_dirs=allow_create_from_dir)
    return project


def _run_search_mode(query: str, limit: int) -> None:
    status(f'Query "{_preview(query)}"', phase="search")
    try:
        ok, msg = check_servers(
            mode="search",
            fast_only=True,
        )
        if not ok:
            ok, msg = check_servers(
                on_status=lambda message: status(
                    message.replace(":", "", 1), phase="search"
                ),
                mode="search",
            )
    except KeyboardInterrupt:
        emit("interrupted", "Search interrupted")
        sys.exit(130)

    if not ok:
        _report_error(RuntimeError(msg))
        _report_relevant_service_diags(RuntimeError(msg), mode="search")
        sys.exit(1)
    status(msg, phase="search")
    try:
        search_semantic(query, limit, verbose=False)
    except KeyboardInterrupt:
        emit("interrupted", "Search interrupted")
        sys.exit(130)
    except (OSError, RuntimeError, sqlite3.Error, ValueError) as e:
        _report_error(e)
        _report_relevant_service_diags(e, mode="search")
        sys.exit(1)


_DOWNLOAD_SERVICES = [
    ("embedding", "com.sova.embedding", config.EMBEDDING_SERVER_URL),
    ("chat", "com.sova.chat", config.CONTEXT_SERVER_URL),
]
_DOWNLOAD_NAME_WIDTH = max(len(name) for name, _, _ in _DOWNLOAD_SERVICES)
# Give launchd time to spawn the service before treating a dead process.
# with no cached model and no download progress as a failure.
_DOWNLOAD_STALL_TIMEOUT_S = 30.0


def _run_download_mode() -> None:
    """Download both model files by briefly starting each service."""
    emit("run_started", "Downloading models")
    needs_install = False
    downloaded_any = False
    for name, label, url in _DOWNLOAD_SERVICES:
        if not is_service_installed(label):
            emit(
                "download_skipped",
                "service not installed",
                level="warning",
                item=name,
                data={"item_width": _DOWNLOAD_NAME_WIDTH},
            )
            needs_install = True
            continue
        if is_model_cached(label):
            emit(
                "download_cached",
                "cached",
                item=name,
                data={"item_width": _DOWNLOAD_NAME_WIDTH},
            )
            continue
        downloaded_any = True
        start_service(label)
        last_status: str | None = None
        stall_started: float | None = None
        try:
            while True:
                model_status = get_model_status(label)
                if model_status != last_status:
                    display_status = (
                        "preparing" if model_status == "starting" else model_status
                    )
                    emit(
                        "status",
                        display_status,
                        phase="download",
                        item=name,
                        data={"item_width": _DOWNLOAD_NAME_WIDTH},
                    )
                    last_status = model_status
                if is_model_cached(label):
                    break
                # Fail instead of polling forever when the service died.
                # without producing a cached model or download progress.
                if model_status.startswith("downloading") or is_service_running(label):
                    stall_started = None
                elif stall_started is None:
                    stall_started = time.monotonic()
                elif (time.monotonic() - stall_started) > _DOWNLOAD_STALL_TIMEOUT_S:
                    diagnostics = get_service_diagnostics(url)
                    _report_error_block(
                        "model download failed",
                        cause=diagnostics
                        or f"{name} service stopped before its model was cached",
                        action="check the model configuration and re-run: sova-install",
                        detail=f"log: ~/.sova/logs/{name}.err.log",
                    )
                    sys.exit(1)
                time.sleep(1)
        except KeyboardInterrupt:
            emit("interrupted", "Download interrupted")
            stop_server(url, suppress_interrupt=True)
            sys.exit(130)
        stop_server(url)
        emit(
            "download_completed",
            "downloaded",
            item=name,
            data={"item_width": _DOWNLOAD_NAME_WIDTH},
        )
    if needs_install:
        _report_error_block(
            "model services are not installed",
            action="run: sova-install",
        )
        raise typer.Exit(1)
    if downloaded_any:
        emit("completed", "Model download complete")
    else:
        emit("completed", "All models are cached")


def _run_list_mode() -> None:
    docs = find_docs()
    try:
        list_docs(docs)
    except sqlite3.OperationalError as e:
        cause = str(e).strip() or "sqlite extension failed to initialize"
        _report_error_block(
            "database extension unavailable",
            cause=cause,
            action="reinstall and retry: sova-install",
        )
        sys.exit(1)
    except (OSError, RuntimeError, sqlite3.Error, ValueError) as e:
        _report_error(e)
        sys.exit(1)


def _run_index_mode() -> None:
    if not config.get_docs_dir():
        _report_error_block(
            "docs directory is not configured",
            action="run: sova index /path/to/pdfs",
        )
        sys.exit(1)

    try:
        conn = init_db()
    except (OSError, sqlite3.Error) as e:
        _report_error_block("failed to initialize database", cause=str(e))
        sys.exit(1)
    try:
        signature_state = _sync_index_signatures(conn)
    except sqlite3.Error as e:
        _report_error_block(
            "failed to synchronize index metadata",
            cause=str(e),
            action="retry indexing or inspect local database state",
        )
        sys.exit(1)

    docs = find_docs()
    if not docs:
        conn.close()
        _report_error_block(
            "no documents found",
            cause="the project contains no PDF or Markdown source documents",
            action="add source documents to the project directory and retry",
        )
        sys.exit(1)
    start_time = time.time()
    interrupted = False
    failed = False
    sources: list[_PreparedSource] = []
    prepared: list[tuple[str, int]] = []

    project_id = config.get_active_project_id() or "project"
    emit(
        "run_started",
        f"Indexing {project_id}  {len(docs)} documents",
        phase="prepare",
        data={"project": project_id, "documents": len(docs)},
    )

    try:
        # Preparation owns the machine first. Keeping both models unloaded gives
        # PDF layout analysis and OCR all available unified memory.
        status("Preparing documents", phase="prepare")
        stop_server(config.CONTEXT_SERVER_URL, suppress_interrupt=True)
        stop_server(config.EMBEDDING_SERVER_URL, suppress_interrupt=True)
        try:
            for doc_index, doc in enumerate(docs, start=1):
                report_scope(doc_index, len(docs))
                source = _prepare_source(doc["name"], doc["pdf"], doc["md"], conn)
                if source is None:
                    failed = True
                    break
                sources.append(source)
        except KeyboardInterrupt:
            interrupted = True
            emit("interrupting", "Saving prepared documents")
        except (OSError, RuntimeError, sqlite3.Error, ValueError) as e:
            failed = True
            _report_error(e)

        # Exact chunk boundaries use the tokenizer embedded in the real GGUF.
        # This dedicated phase starts only after OCR and releases the embedding
        # model before the larger context model is admitted.
        if not interrupted and not failed:
            tokenization_plan = [
                (
                    source,
                    _current_tokenized_doc_id(
                        conn, source, signature_state.chunk_sig
                    ),
                )
                for source in sources
            ]
            needs_tokenizer = any(doc_id is None for _, doc_id in tokenization_plan)
            if not needs_tokenizer:
                prepared.extend(
                    (source.name, doc_id)
                    for source, doc_id in tokenization_plan
                    if doc_id is not None
                )
                status("Reused tokenized documents", phase="tokenize")
            else:
                report_scope(None, None)
                status("Loading model", phase="tokenize")
                ok, msg = check_servers(
                    on_status=lambda message: status(
                        message.replace(":", "", 1), phase="tokenize"
                    ),
                    mode="index_embed",
                )
                if not ok:
                    failed = True
                    _report_error(RuntimeError(msg))
                    _report_relevant_service_diags(
                        RuntimeError(msg), mode="index_embed"
                    )
                else:
                    status(msg, phase="tokenize")
                    tokenize_runtime_tick = _make_runtime_reporter(
                        "index.tokenize", "embedding", mode="index"
                    )
                    tokenize_runtime_tick(True)
                    try:
                        for doc_index, (source, current_doc_id) in enumerate(
                            tokenization_plan, start=1
                        ):
                            report_scope(doc_index, len(tokenization_plan))
                            if current_doc_id is not None:
                                prepared.append((source.name, current_doc_id))
                                continue
                            doc_id, chunks, sections = _tokenize_doc(
                                source, conn, signature_state.chunk_sig
                            )
                            prepared.append((source.name, doc_id))
                            del chunks, sections
                            tokenize_runtime_tick(False)
                    except KeyboardInterrupt:
                        interrupted = True
                        emit("interrupting", "Saving tokenized documents")
                    except (OSError, RuntimeError, sqlite3.Error, ValueError) as e:
                        failed = True
                        _report_error(e)
                        _report_relevant_service_diags(e, mode="index_embed")
                    finally:
                        stop_server(
                            config.EMBEDDING_SERVER_URL, suppress_interrupt=True
                        )

        # Context is generated only after every source is fully prepared. The
        # model is loaded once for the whole project and unloaded before embed.
        if (
            not interrupted
            and not failed
            and _context_work_pending(
                conn,
                signature_state.context_sig,
                force_rebuild=signature_state.force_rebuild_context,
            )
        ):
            report_scope(None, None)
            status("Loading model", phase="context")
            ok, msg = check_servers(
                on_status=lambda message: status(
                    message.replace(":", "", 1), phase="context"
                ),
                mode="index_context",
            )
            if not ok:
                failed = True
                _report_error(RuntimeError(msg))
                _report_relevant_service_diags(RuntimeError(msg), mode="index_context")
            else:
                status(msg, phase="context")
                context_runtime_tick = _make_runtime_reporter(
                    "index.context", "chat", mode="index"
                )
                context_runtime_tick(True)
                try:
                    for doc_index, (name, doc_id) in enumerate(prepared, start=1):
                        report_scope(doc_index, len(prepared))
                        chunks, sections = _load_prepared_doc(conn, doc_id)
                        _generate_contexts(
                            name,
                            doc_id,
                            chunks,
                            sections,
                            conn,
                            force_rebuild_context=signature_state.force_rebuild_context,
                            runtime_tick=context_runtime_tick,
                        )
                        del chunks, sections
                except KeyboardInterrupt:
                    interrupted = True
                    emit("interrupting", "Saving generated contexts")
                except (OSError, RuntimeError, sqlite3.Error) as e:
                    failed = True
                    _report_error(e)
                    _report_relevant_service_diags(e, mode="index_context")
                finally:
                    stop_server(config.CONTEXT_SERVER_URL, suppress_interrupt=True)
        elif not interrupted and not failed:
            status("Reused generated contexts", phase="context")

        # Embedding starts with the context model fully released.
        if (
            not interrupted
            and not failed
            and _embedding_work_pending(
                conn,
                signature_state.embed_sig,
                force_rebuild=signature_state.force_rebuild_embed,
            )
        ):
            report_scope(None, None)
            status("Loading model", phase="embed")
            stop_server(config.EMBEDDING_SERVER_URL)
            time.sleep(2)
            ok, msg = check_servers(
                on_status=lambda message: status(
                    message.replace(":", "", 1), phase="embed"
                ),
                mode="index_embed",
            )
            if not ok:
                failed = True
                _report_error(RuntimeError(msg))
                _report_relevant_service_diags(RuntimeError(msg), mode="index_embed")
            else:
                status(msg, phase="embed")
                embed_runtime_tick = _make_runtime_reporter(
                    "index.embed", "embedding", mode="index"
                )
                embed_runtime_tick(True)
                try:
                    status("Checking model", phase="embed")
                    run_embedding_canary()
                    for doc_index, (name, doc_id) in enumerate(prepared, start=1):
                        report_scope(doc_index, len(prepared))
                        chunks, sections = _load_prepared_doc(conn, doc_id)
                        _embed_doc(
                            name,
                            doc_id,
                            chunks,
                            sections,
                            conn,
                            force_rebuild_embed=signature_state.force_rebuild_embed,
                            runtime_tick=embed_runtime_tick,
                        )
                        del chunks, sections
                except KeyboardInterrupt:
                    interrupted = True
                    emit("interrupting", "Saving generated embeddings")
                    stop_server(config.EMBEDDING_SERVER_URL, suppress_interrupt=True)
                except (OSError, RuntimeError, sqlite3.Error) as e:
                    failed = True
                    _report_error(e)
                    _report_relevant_service_diags(e, mode="index_embed")
                    stop_server(config.EMBEDDING_SERVER_URL, suppress_interrupt=True)
        elif not interrupted and not failed:
            status("Reused embeddings", phase="embed")

        # Finalize the searchable vector index only after all durable rows exist.
        if not interrupted and not failed:
            report_scope(None, None)
            status("Building vector index", phase="finalize")
            try:
                quantize_vectors(conn)
                _prune_missing_documents(conn, {str(doc["name"]) for doc in docs})
                _commit_index_signatures(conn, signature_state)
            except (OSError, RuntimeError, sqlite3.Error) as e:
                failed = True
                _report_error_block(
                    "failed to finalize index",
                    cause=str(e),
                    action="retry indexing",
                )

        get_cache().clear()
        if interrupted:
            stop_server(config.CONTEXT_SERVER_URL, suppress_interrupt=True)
            stop_server(config.EMBEDDING_SERVER_URL, suppress_interrupt=True)
    finally:
        conn.close()

    elapsed = fmt_duration(time.time() - start_time).strip()
    if interrupted:
        emit(
            "interrupted",
            f"Index interrupted after {elapsed}. Completed chunks are saved.",
            data={"elapsed": elapsed},
        )
        sys.exit(130)
    if failed:
        emit("failed", f"Index failed after {elapsed}", level="error")
        sys.exit(1)
    emit(
        "completed",
        f"Index complete. {len(docs)} documents in {elapsed}.",
        data={"documents": len(docs), "elapsed": elapsed},
    )


def _run_projects_mode() -> None:
    rows = projects.list_projects()
    if not rows:
        emit(
            "projects_empty",
            "No projects. Run: sova index /path/to/pdfs",
        )
        return
    table = make_table()
    table.add_column("Id")
    table.add_column("Docs")
    for p in rows:
        table.add_row(
            p.project_id,
            _display_path(p.docs_dir),
        )
    render_table(table)


def _run_doctor_mode() -> None:
    """Run read-only integrity checks for the active project."""
    db_path = config.get_db_path()
    if not db_path.exists():
        _report_error_block(
            "database not ready",
            cause="index database not found",
            action="run: sova index <project>",
        )
        raise typer.Exit(1)
    conn = connect_readonly()
    try:
        findings = audit_database(
            conn,
            expected_signatures={
                _META_CONTEXT_SIG: _context_pipeline_signature(),
                _META_EMBED_SIG: _embedding_pipeline_signature(),
                _META_CHUNK_SIG: _chunk_pipeline_signature(),
            },
            expected_schema_version=SCHEMA_VERSION,
        )
        source_docs = find_docs()
        source_names = {str(doc["name"]) for doc in source_docs}
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if "documents" not in tables:
            indexed_signatures = {}
        else:
            document_columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(documents)").fetchall()
            }
            if "source_signature" in document_columns:
                indexed_signatures = {
                    str(name): str(signature) if signature else ""
                    for name, signature in conn.execute(
                        "SELECT name, source_signature FROM documents ORDER BY name"
                    )
                }
            else:
                indexed_signatures = {
                    str(row[0]): ""
                    for row in conn.execute("SELECT name FROM documents ORDER BY name")
                }
    finally:
        conn.close()
    indexed_names = set(indexed_signatures)
    stale_documents = sorted(indexed_names - source_names)
    if stale_documents:
        findings.append(
            Finding(
                "sources.missing",
                "Indexed documents have no current source: "
                + ", ".join(stale_documents[:5]),
                len(stale_documents),
            )
        )
    changed_sources = []
    for doc in source_docs:
        name = str(doc["name"])
        source_path = doc.get("pdf") or doc.get("md")
        if (
            name in indexed_signatures
            and indexed_signatures[name]
            and isinstance(source_path, Path)
            and _file_signature(source_path) != indexed_signatures[name]
        ):
            changed_sources.append(name)
    if changed_sources:
        findings.append(
            Finding(
                "sources.changed",
                "Source content has changed since indexing: "
                + ", ".join(changed_sources[:5]),
                len(changed_sources),
            )
        )
    data_dir = config.get_data_dir()
    generated_names = (
        {path.stem for path in data_dir.glob("*.md")} if data_dir.exists() else set()
    )
    orphan_generated = sorted(generated_names - source_names)
    if orphan_generated:
        findings.append(
            Finding(
                "sources.orphan_generated_markdown",
                "Generated Markdown has no current source: "
                + ", ".join(orphan_generated[:5]),
                len(orphan_generated),
            )
        )
    if not findings:
        emit("audit_completed", "Database checks passed", data={"findings": 0})
        return
    for finding in findings:
        emit(
            "audit_finding",
            f"{finding.message}: {finding.count}",
            level="warning",
            data={
                "code": finding.code,
                "count": finding.count,
                "message": finding.message,
            },
        )
    emit(
        "audit_completed",
        f"Database audit found {len(findings)} issue(s)",
        level="warning",
        data={"findings": len(findings)},
    )
    raise typer.Exit(1)


app = typer.Typer(
    name="sova",
    add_completion=False,
    rich_markup_mode=None,
    help="Local document search",
    epilog='Examples: sova index /path/to/pdfs; sova search <project> "<query>"',
)


@app.callback()
def configure_cli(
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit newline-delimited JSON for agents and scripts",
    ),
) -> None:
    configure_output("json" if json_output else "auto")


@app.command("help", help="Show help")
def help_command(ctx: typer.Context) -> None:
    root = ctx.parent or ctx
    help_text = root.get_help()
    if is_json_output():
        emit("help", "Sova command help", data={"text": help_text})
    else:
        print(help_text)


@app.command("projects", help="List configured projects")
def projects_command() -> None:
    _run_projects_mode()


@app.command("doctor", help="Check a project database without changing it")
def doctor_command(
    project: str = typer.Argument(..., help="Project id/path"),
) -> None:
    _activate_project_from_ref(project)
    _run_doctor_mode()


@app.command("download", help="Download all model files")
def download_command() -> None:
    _run_download_mode()


@app.command("remove", help="Remove project from Sova")
def remove_command(
    project: str = typer.Argument(..., help="Project id/path"),
    delete_data: bool = typer.Option(
        False,
        "--delete-data",
        help="Also delete the local index and extracted documents",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Confirm deletion without prompting",
    ),
) -> None:
    if delete_data and not yes:
        if is_json_output() or not sys.stdin.isatty():
            _report_error_block(
                "confirmation required",
                action="re-run with --delete-data --yes",
            )
            raise typer.Exit(2)
        confirmed = typer.confirm(
            f"Delete all local data for '{project}'?",
            default=False,
        )
        if not confirmed:
            emit("cancelled", "No changes made")
            return
    try:
        removed = projects.remove_project(project, delete_data=delete_data)
    except ValueError as e:
        _report_error_block(
            "project not found",
            cause=str(e).replace("project not found: ", ""),
            action="run: sova projects",
        )
        sys.exit(1)
    except OSError as e:
        _report_error_block(
            "project data could not be deleted",
            cause=str(e),
            action="fix filesystem access and retry; the project remains registered",
        )
        sys.exit(1)
    outcome = (
        f"data deleted from {_display_path(removed.root_dir)}"
        if delete_data
        else f"data kept at {_display_path(removed.root_dir)}"
    )
    emit("project_removed", f"Removed {removed.project_id}. {outcome}.")


@app.command("list", help="List docs and indexing status")
def list_command(
    project: str = typer.Argument(..., help="Project id/path"),
) -> None:
    _activate_project_from_ref(project)
    _run_list_mode()


@app.command("index", help="Index project docs")
def index_command(
    project: str = typer.Argument(..., help="Project id/path"),
) -> None:
    _activate_project_from_ref(project, allow_create_from_dir=True)
    _run_index_mode()


@app.command("search", help="Search project docs")
def search_command(
    project: Annotated[
        str,
        typer.Argument(help="Project id/path"),
    ],
    query: Annotated[list[str], typer.Argument(help="Search query text")],
    limit: int = typer.Option(10, "-n", "--limit", help="Max results (default: 10)"),
) -> None:
    _activate_project_from_ref(project)
    _run_search_mode(" ".join(query), limit)


_COMMAND_NAMES = set(projects.RESERVED_PROJECT_IDS)


def _argv_with_default_search(argv: list[str]) -> list[str]:
    """Route `sova <project> <query>` to the hidden search command."""
    if not argv:
        return argv
    command_index = 1 if argv[0] == "--json" and len(argv) > 1 else 0
    head = argv[command_index]
    if head in _COMMAND_NAMES or head.startswith("-"):
        return argv
    return [*argv[:command_index], "search", *argv[command_index:]]


def _handle_interrupt() -> None:
    emit("interrupting", "Stopping services")
    stop_server(config.CONTEXT_SERVER_URL, suppress_interrupt=True)
    stop_server(config.EMBEDDING_SERVER_URL, suppress_interrupt=True)
    emit("interrupted", "Interrupted")
    sys.exit(130)


def main() -> None:
    """Main entry point."""
    config.clear_active_project()
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    previous_sigpipe = signal.getsignal(signal.SIGPIPE)
    signal.signal(signal.SIGTERM, lambda _signum, _frame: _handle_interrupt())
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    try:
        argv = sys.argv[1:]
        json_requested = "--json" in argv
        configure_output("json" if json_requested else "auto")
        if "--_watchdog" in argv:
            from sova.llama_client import cleanup_idle_services

            cleanup_idle_services()
            return
        command = typer.main.get_command(app)
        if json_requested and "--help" in argv:
            help_args = [arg for arg in argv if arg not in {"--json", "--help"}]
            root_context = typer.Context(command, info_name="sova")
            target = command
            target_context = root_context
            if help_args and isinstance(command, TyperGroup):
                subcommand = command.commands.get(help_args[0])
                if subcommand is not None:
                    target = subcommand
                    target_context = typer.Context(
                        target,
                        info_name=help_args[0],
                        parent=root_context,
                    )
            help_text = target.get_help(target_context)
            emit("help", "Sova command help", data={"text": help_text})
            return
        try:
            exit_code = command.main(
                args=_argv_with_default_search(argv),
                prog_name="sova",
                standalone_mode=False,
            )
            if isinstance(exit_code, int) and exit_code != 0:
                sys.exit(exit_code)
        except click_exceptions.Abort:
            _handle_interrupt()
        except click_exceptions.ClickException as e:
            if json_requested:
                _report_error_block(
                    "invalid command",
                    cause=e.format_message(),
                    action="run: sova --json help",
                )
            else:
                e.show()
            sys.exit(e.exit_code)
        except projects.RegistryError as e:
            _report_error_block(
                "project registry is invalid",
                cause=str(e),
                action="fix ~/.sova/projects/registry.json or re-create it via indexing",
            )
            sys.exit(1)
    finally:
        close_output()
        config.clear_active_project()
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGPIPE, previous_sigpipe)


if __name__ == "__main__":
    main()
