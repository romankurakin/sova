"""Command-line interface and Rich UI."""

import argparse
import hashlib
import re
import sqlite3
import sys
import time
from collections import deque
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from rich.console import Group
from rich.live import Live
from rich.text import Text

from sova.cache import get_cache
from sova import config
from sova import projects
from sova.db import (
    connect_readonly,
    embedding_to_blob,
    get_meta,
    get_doc_status,
    init_db,
    quantize_vectors,
    set_meta,
)
from sova.extract import (
    chunk_text,
    extract_pdf,
    find_docs,
    find_section,
    parse_sections,
)
from sova.llama_client import (
    CONTEXT_SYSTEM_PROMPT,
    CONTEXT_USER_PROMPT,
    QUERY_TASK,
    check_servers,
    generate_context,
    get_model_status,
    get_service_diagnostics,
    get_services_runtime_status,
    get_embeddings_batch,
    get_query_embedding,
    is_model_cached,
    is_service_installed,
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
    console,
    fmt_duration,
    format_line,
    make_table,
    report as ui_report,
    render_table,
    report_progress,
)


def _display_path(path: Path) -> str:
    """Render path with ~ for home-relative locations."""
    home = Path.home()
    try:
        rel = path.relative_to(home)
        return "~" if str(rel) == "." else f"~/{rel}"
    except ValueError:
        return str(path)


SOVA_ASCII = """\
   ___
  (o o)
 (  V  )
/|  |  |\\
  "   " """

# Keep embedding work bounded so Python memory and llama-server lifetime stay.
# controlled on large indexes.
_EMBED_WINDOW_CHUNKS = 256
_EMBED_RECYCLE_CHUNKS = 1800
_EMBED_RECYCLE_PAUSE_S = 2.0
_EMBED_RECYCLE_CANARY_REQUESTS = 4
_EMBED_PREFIX_VERSION = "chunk-prefix.v1"
_CONTEXT_RETRY_ATTEMPTS = 2
_CONTEXT_RECYCLE_PAUSE_S = 2.0
_RUNTIME_REFRESH_S = 20.0
_LIVE_PROGRESS_PCT_STEP = 2
_LIVE_PROGRESS_REFRESH_S = 6.0

_META_CONTEXT_SIG = "pipeline.context.signature"
_META_EMBED_SIG = "pipeline.embedding.signature"
_META_CHUNK_SIG = "pipeline.chunk.signature"


@dataclass(frozen=True)
class _IndexSignatureState:
    force_rebuild_context: bool
    force_rebuild_embed: bool
    context_sig: str
    embed_sig: str
    chunk_sig: str


def _preview(text: str, max_chars: int = 48) -> str:
    """Short single-line preview for headers."""
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 1].rstrip() + "…"


class _IndexLiveView:
    """Minimal live view that keeps phase/runtime pinned above event log."""

    def __init__(self) -> None:
        self._phase = "-"
        self._runtime = "-"
        self._events: deque[str] = deque(maxlen=18)
        self._live: Live | None = None

    def start(self) -> None:
        if self._live is not None:
            return
        self._live = Live(
            self._render(),
            console=console,
            screen=False,
            transient=False,
            refresh_per_second=4,
        )
        self._live.__enter__()

    def stop(self) -> None:
        if self._live is None:
            return
        self._live.__exit__(None, None, None)
        self._live = None

    def emit(self, name: str, msg: str) -> None:
        if name == "phase":
            self._phase = msg
            self._refresh()
            return
        if name == "runtime":
            self._runtime = msg
            self._refresh()
            return
        line = format_line(name, msg)
        # Keep progress lines stable: update the latest context/embed/server line.
        # in place instead of appending a new event every tick.
        if name in {"context", "embed", "server"} and self._events:
            prefix = format_line(name, "")
            if self._events[-1].startswith(prefix):
                self._events[-1] = line
                self._refresh()
                return
        self._events.append(line)
        self._refresh()

    def _refresh(self) -> None:
        if self._live is None:
            return
        self._live.update(self._render())

    def _render(self) -> Group:
        header = [
            format_line("phase", self._phase),
            format_line("runtime", self._runtime),
            "",
        ]
        events = (
            list(self._events) if self._events else [format_line("event", "waiting")]
        )
        return Group(*header, *events)


_ACTIVE_INDEX_VIEW: _IndexLiveView | None = None


def report(name: str, msg: str) -> None:
    """Route report lines to live index view when active."""
    if _ACTIVE_INDEX_VIEW is not None:
        _ACTIVE_INDEX_VIEW.emit(name, msg)
        return
    ui_report(name, msg)


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


def _fmt_gib(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f} GiB"


def _fmt_rss(value_mib: float | None) -> str:
    if value_mib is None:
        return "-"
    if value_mib >= 1024:
        return f"{value_mib / 1024.0:.1f} GiB"
    return f"{value_mib:.0f} MiB"


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
    report("error", f"[red]{summary}[/red]")
    if cause:
        report("cause", cause)
    if action:
        report("action", action)
    if detail:
        report("detail", detail)


def _infer_service_name(message: str) -> str | None:
    low = message.lower()
    if ":8081" in low or "embedding" in low:
        return "embedding"
    if ":8082" in low or "rerank" in low:
        return "reranker"
    if ":8083" in low or "context" in low or "chat" in low:
        return "chat"
    return None


def _classify_error(message: str) -> tuple[str, str | None, str | None]:
    low = message.lower()
    if "memory hard-cap exceeded" in low:
        return (
            "model does not fit current memory budget",
            message,
            "close extra apps and retry, or reduce reserve for this mode",
        )
    if "physical batch size" in low or "too large to process" in low:
        return (
            "reranker request exceeds server batch capacity",
            message,
            "run sova-install to update reranker service settings, then retry",
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
        else "reranker"
        if url == config.RERANKER_SERVER_URL
        else "chat"
        if url == config.CONTEXT_SERVER_URL
        else "service"
    )
    diag = get_service_diagnostics(url)
    if diag:
        report("service", f"{name} {diag}")


def _report_relevant_service_diags(
    exc: BaseException,
    mode: str,
    *,
    include_reranker: bool = True,
) -> None:
    text = _format_error_chain(exc).lower()
    urls: list[str] = []
    if "8081" in text or "embedding" in text:
        urls.append(config.EMBEDDING_SERVER_URL)
    if include_reranker and ("8082" in text or "rerank" in text):
        urls.append(config.RERANKER_SERVER_URL)
    if "8083" in text or "context" in text or "chat" in text:
        urls.append(config.CONTEXT_SERVER_URL)
    if not urls:
        if mode == "search":
            urls = [config.EMBEDDING_SERVER_URL]
            if include_reranker:
                urls.append(config.RERANKER_SERVER_URL)
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


def _service_status_line(service_name: str, *, with_memory: bool = False) -> str:
    rows = get_services_runtime_status()
    for row in rows:
        if row["name"] != service_name:
            continue
        state = str(row["state"])
        rss = _fmt_rss(row["rss_mib"] if isinstance(row["rss_mib"], float) else None)
        if state == "running":
            label = "[green]running[/green]"
        elif state == "starting":
            label = "[yellow]starting[/yellow]"
        elif state == "stopped":
            label = "[dim]stopped[/dim]"
        else:
            label = "[dim]not installed[/dim]"
        if with_memory:
            return f"{service_name} {label} | rss {rss}"
        return f"{service_name} {label}"
    if with_memory:
        return f"{service_name} [dim]unknown[/dim] | rss -"
    return f"{service_name} [dim]unknown[/dim]"


def _report_phase_runtime(phase: str, service_name: str, mode: str = "index") -> None:
    """Compact runtime snapshot for indexing phases."""
    try:
        effective = config.get_effective_available_gib()
        reserve = config.get_memory_reserve_gib(mode)
        budget_now = max(0.0, round(effective - reserve, 2))
        service = _service_status_line(service_name, with_memory=False)
        report("phase", f"{phase} (updated {time.strftime('%H:%M:%S')})")
        if mode == "index":
            report("runtime", f"free-for-model {_fmt_gib(budget_now)} | {service}")
        else:
            cap = config.get_memory_hard_cap_gib(mode)
            report("runtime", f"cap {_fmt_gib(cap)} | {service}")
    except Exception:
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


def _prepare_doc(
    name: str,
    pdf_path: Path | None,
    md_path: Path | None,
    conn: sqlite3.Connection,
) -> tuple[int, list[dict], list[dict]] | None:
    """Extract, chunk, and store a document.  Returns (doc_id, chunks, sections) or None."""
    report("doc", name)

    extracted_now = False
    if not md_path or not md_path.exists():
        if not pdf_path:
            _report_error_block(
                "extract failed",
                cause=f"{name}: source PDF not found",
                action="check docs directory and document mapping, then retry",
            )
            return None
        try:
            start = time.time()
            markdown = extract_pdf(pdf_path)
            data_dir = config.get_data_dir()
            data_dir.mkdir(parents=True, exist_ok=True)
            md_path = data_dir / f"{name}.md"
            md_path.write_text(markdown, encoding="utf-8")
            lines = len(markdown.splitlines())
            report("extract", f"{lines:>9,} lines  {fmt_duration(time.time() - start)}")
            extracted_now = True
        except Exception as e:
            _report_error_block(
                "extract failed",
                cause=f"{name}: {e}",
                action="verify PDF is readable and retry",
            )
            return None

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
            "UPDATE documents SET path = ?, line_count = ?, expected_chunks = ? WHERE id = ?",
            (str(md_path), len(lines), len(chunks), doc_id),
        )
    else:
        cursor = conn.execute(
            "INSERT INTO documents (name, path, line_count, expected_chunks) VALUES (?, ?, ?, ?)",
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
        SELECT id, start_line, end_line, word_count, text, section_id, is_index
        FROM chunks
        WHERE doc_id = ?
        ORDER BY id
        """,
        (doc_id,),
    ).fetchall()
    existing_by_start: dict[int, tuple[int, int, int, str, int | None, int]] = {}
    duplicate_ids: list[int] = []
    for row_data in existing_rows:
        chunk_id, start_line, end_line, word_count, text_value, section_id, is_idx = (
            row_data
        )
        if start_line in existing_by_start:
            duplicate_ids.append(chunk_id)
            continue
        existing_by_start[start_line] = (
            chunk_id,
            end_line,
            word_count,
            text_value,
            section_id,
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
        is_idx = 1 if is_index_like(chunk["text"]) else 0

        existing = existing_by_start.get(start_line)
        if existing is None:
            conn.execute(
                """
                INSERT INTO chunks (doc_id, section_id, start_line, end_line, word_count, text, is_index)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    sec_id,
                    start_line,
                    chunk["end_line"],
                    chunk["word_count"],
                    chunk["text"],
                    is_idx,
                ),
            )
            continue

        chunk_id, end_line, word_count, text_value, old_section_id, old_is_idx = (
            existing
        )
        content_changed = (
            end_line != chunk["end_line"]
            or word_count != chunk["word_count"]
            or text_value != chunk["text"]
            or old_is_idx != is_idx
        )
        if content_changed:
            conn.execute(
                """
                UPDATE chunks
                SET section_id = ?, end_line = ?, word_count = ?, text = ?, is_index = ?, embedding = NULL
                WHERE id = ?
                """,
                (
                    sec_id,
                    chunk["end_line"],
                    chunk["word_count"],
                    chunk["text"],
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


def _signature(parts: list[str]) -> str:
    raw = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _context_pipeline_signature() -> str:
    return _signature(
        [
            config.CONTEXT_MODEL,
            CONTEXT_SYSTEM_PROMPT,
            CONTEXT_USER_PROMPT,
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
    return _signature([str(config.CHUNK_SIZE), "chunk-text.v1"])


def _sync_index_signatures(conn: sqlite3.Connection) -> _IndexSignatureState:
    current_context = _context_pipeline_signature()
    current_embed = _embedding_pipeline_signature()
    current_chunk = _chunk_pipeline_signature()

    stored_context = get_meta(conn, _META_CONTEXT_SIG)
    stored_embed = get_meta(conn, _META_EMBED_SIG)
    stored_chunk = get_meta(conn, _META_CHUNK_SIG)

    context_changed = stored_context is not None and stored_context != current_context
    embed_changed = stored_embed is not None and stored_embed != current_embed
    chunk_changed = stored_chunk is not None and stored_chunk != current_chunk

    if context_changed:
        report("event", "context pipeline changed; refreshing contexts and embeddings")
    elif embed_changed:
        report("event", "embedding pipeline changed; refreshing embeddings")

    if chunk_changed:
        report("event", "chunking settings changed; syncing chunk rows during prepare")

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

    if force_rebuild_context:
        existing_contexts: set[int] = set()
    else:
        existing_contexts = set(
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
        )
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
        count = len(chunks)
        report("context", f"{count:>9,} chunks")
        return

    try:
        start = time.time()
        total = len(chunks_needing_context)
        use_progress_bar = _ACTIVE_INDEX_VIEW is None
        progress_cm = report_progress("context") if use_progress_bar else nullcontext()
        done = 0
        last_reported_pct = -1
        last_report_ts = 0.0

        with progress_cm as progress:
            task = None
            if use_progress_bar:
                assert progress is not None
                task = progress.add_task("", total=total)
            for i, chunk, chunk_id in chunks_needing_context:
                sec_idx = find_section(sections, chunk["start_line"])
                sec_title = sections[sec_idx]["title"] if sec_idx is not None else None
                prev_text = chunks[i - 1]["text"] if i > 0 else ""
                next_text = chunks[i + 1]["text"] if i + 1 < len(chunks) else ""

                ctx = ""
                for attempt in range(_CONTEXT_RETRY_ATTEMPTS):
                    try:
                        candidate = generate_context(
                            name, sec_title, chunk["text"], prev_text, next_text
                        )
                        ctx = " ".join(candidate.split())
                        if not ctx:
                            raise RuntimeError("context model returned empty content")
                        break
                    except Exception:
                        if attempt == (_CONTEXT_RETRY_ATTEMPTS - 1):
                            raise
                        stop_server(config.CONTEXT_SERVER_URL)
                        time.sleep(_CONTEXT_RECYCLE_PAUSE_S)

                assert ctx

                conn.execute(
                    """
                    INSERT INTO chunk_contexts (chunk_id, context, model)
                    VALUES (?, ?, ?)
                    ON CONFLICT(chunk_id)
                    DO UPDATE SET context = excluded.context, model = excluded.model
                    """,
                    (chunk_id, ctx, config.CONTEXT_MODEL),
                )
                conn.execute(
                    "UPDATE chunks SET embedding = NULL WHERE id = ?",
                    (chunk_id,),
                )
                conn.commit()
                done += 1
                if use_progress_bar and task is not None:
                    assert progress is not None
                    progress.update(task, advance=1)
                else:
                    pct = _progress_pct(done, total)
                    now = time.monotonic()
                    if (
                        done == total
                        or pct >= (last_reported_pct + _LIVE_PROGRESS_PCT_STEP)
                        or (now - last_report_ts) >= _LIVE_PROGRESS_REFRESH_S
                    ):
                        report("context", f"{done:>9,}/{total:,} chunks")
                        last_reported_pct = pct
                        last_report_ts = now
                if runtime_tick:
                    runtime_tick(False)

        report("context", f"{total:>9,} chunks {fmt_duration(time.time() - start)}")

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

    embedded_ids = (
        set()
        if force_rebuild_embed
        else set(
            r[0]
            for r in conn.execute(
                "SELECT id FROM chunks WHERE doc_id = ? AND embedding IS NOT NULL",
                (doc_id,),
            ).fetchall()
        )
    )
    pending_embed = []
    for i, chunk in enumerate(chunks):
        chunk_id = chunk_id_by_start.get(chunk["start_line"])
        if chunk_id is not None and chunk_id not in embedded_ids:
            pending_embed.append((i, chunk, chunk_id))

    if not pending_embed:
        count = len(chunks)
        report("embed", f"{count:>9,} chunks")
        return

    try:
        start = time.time()
        total = len(pending_embed)
        absolute_total = len(chunks)
        absolute_done_base = len(embedded_ids)
        use_progress_bar = _ACTIVE_INDEX_VIEW is None
        progress_cm = report_progress("embed") if use_progress_bar else nullcontext()
        done = 0
        last_reported_pct = -1
        last_report_ts = 0.0

        with progress_cm as progress:
            task = None
            if use_progress_bar:
                assert progress is not None
                task = progress.add_task("", total=total)
            embedded_since_recycle = 0

            for window_start in range(0, total, _EMBED_WINDOW_CHUNKS):
                window = pending_embed[
                    window_start : window_start + _EMBED_WINDOW_CHUNKS
                ]
                window_texts: list[str] = []
                window_chunk_ids: list[int] = []

                for i, chunk, chunk_id in window:
                    sec_idx = find_section(sections, chunk["start_line"])
                    sec_title = (
                        sections[sec_idx]["title"] if sec_idx is not None else None
                    )
                    prefix = f"[{name}"
                    if sec_title:
                        prefix += f" | {sec_title}"
                    prefix += "]\n\n"

                    llm_ctx = context_map.get(chunk_id)
                    if llm_ctx and llm_ctx.strip():
                        prefix += llm_ctx.strip() + "\n\n"

                    window_texts.append(prefix + chunk["text"])
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
                    ) -> None:
                        nonlocal attempt_done, window_done, done
                        nonlocal last_reported_pct, last_report_ts
                        rows = [
                            (
                                embedding_to_blob(emb),
                                remaining_chunk_ids[batch_idx],
                            )
                            for batch_idx, emb in zip(batch_indices, batch_embeddings)
                        ]
                        conn.executemany(
                            "UPDATE chunks SET embedding = ? WHERE id = ?",
                            rows,
                        )
                        conn.commit()

                        batch_count = len(rows)
                        attempt_done += batch_count
                        window_done += batch_count
                        done += batch_count

                        if use_progress_bar and task is not None:
                            assert progress is not None
                            progress.update(task, advance=batch_count)
                        else:
                            pct = _progress_pct(done, total)
                            now = time.monotonic()
                            if (
                                done == total
                                or pct >= (last_reported_pct + _LIVE_PROGRESS_PCT_STEP)
                                or (now - last_report_ts) >= _LIVE_PROGRESS_REFRESH_S
                            ):
                                report(
                                    "embed",
                                    f"{(absolute_done_base + done):>9,}/{absolute_total:,} chunks",
                                )
                                last_reported_pct = pct
                                last_report_ts = now
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

        report("embed", f"{total:>9,} chunks {fmt_duration(time.time() - start)}")

    except Exception as e:
        conn.rollback()
        raise RuntimeError(f"embedding failed for {name}: {e}") from e


def _doc_status_label(status: dict) -> str:
    """Return a human-readable status label for a document."""
    chunks = status.get("chunks", 0)
    if not chunks:
        return "[dim]pending[/dim]"
    total = status.get("expected") or chunks
    ctx = status.get("contextualized", 0)
    embedded = status.get("embedded", 0)
    # All done.
    if embedded >= total:
        return "[green]ready[/green]"
    # Context generation in progress.
    if ctx < total:
        pct = _progress_pct(ctx, total)
        return f"[yellow]context {pct}%[/yellow]"
    # Context done, embedding in progress.
    pct = _progress_pct(embedded, total)
    return f"[yellow]embed {pct}%[/yellow]"


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
    table.add_column("Size", justify="right", style="dim")
    table.add_column("Status", justify="right")

    ready_count = 0
    for d in docs:
        status = get_doc_status(conn, d["name"]) if conn else {}
        label = _doc_status_label(status)
        if "ready" in label:
            ready_count += 1
        table.add_row(d["name"], fmt_size(d["size"]), label)

    if conn:
        conn.close()

    render_table(table)
    report("indexed", f"{ready_count}/{len(docs)}")


def show_stats(mode: str = "list") -> None:
    """Show document directory."""
    docs_dir = config.get_docs_dir()
    if docs_dir:
        report("docs", _display_path(docs_dir))
    else:
        report("docs", "[dim]not configured[/dim]")

    # Keep list output compact and search-focused.
    if mode == "list":
        try:
            cap_search = config.get_memory_hard_cap_gib("search")
            report(
                "search",
                f"cap {_fmt_gib(cap_search)} | "
                f"{_service_status_line('embedding', with_memory=False)} | "
                f"{_service_status_line('reranker', with_memory=False)}",
            )
        except Exception:
            pass
        return

    # For index summary keep only a minimal memory line.
    if mode == "index":
        try:
            cap_index = config.get_memory_hard_cap_gib("index")
            effective = config.get_effective_available_gib()
            report(
                "index", f"cap {_fmt_gib(cap_index)} | effective {_fmt_gib(effective)}"
            )
        except Exception:
            pass


def search_semantic(
    query: str,
    limit: int = 10,
    verbose: bool = False,
    use_reranker: bool = config.SEARCH_USE_RERANKER,
) -> None:
    """Perform semantic search and display results."""
    if not config.get_db_path().exists():
        _report_error_block(
            "database not ready",
            cause="no index database found",
            action="run indexing first",
        )
        return

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
            report("cache", "hit")
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
            use_reranker=use_reranker,
        )
    finally:
        conn.close()

    if verbose and n_fts:
        report("hybrid", f"{n_vector} vector + {n_fts} fts")

    if not results:
        report("result", "none")
        return

    for i, r in enumerate(results, 1):
        path = r.get("path")
        if path:
            shown_path = _display_path(Path(path))
            location = f"{shown_path}:{r['start']}-{r['end']}"
        else:
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
            rerank_str = (
                f"  [dim]rerank[/dim] {r['rerank_score']:.2f}"
                if "rerank_score" in r
                else ""
            )
            console.print(
                f"  [dim]vec[/dim] {r['embed_score']:.2f}"
                f". [dim]rrf[/dim] {r['rrf_score']:.4f}"
                f"{rerank_str}"
                f"  {tag_str}"
            )
        body = r["text"]
        lines = body.splitlines() or [body]
        for line in lines:
            console.print(Text(f"  {line}", style="dim"))
        if i < len(results):
            console.print()


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
            _report_error_block(
                "project name is reserved",
                cause=(
                    f"'{raw_ref}' conflicts with a CLI command "
                    "(projects, remove, list, index)"
                ),
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
                projects.activate(created)
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
    projects.activate(project)
    return project


def _run_search_mode(query: str, limit: int, use_reranker: bool) -> None:
    rerank_state = "on" if use_reranker else "off"
    report("mode", f'search | "{_preview(query)}" | reranker {rerank_state}')
    try:
        ok, msg = check_servers(
            mode="search",
            fast_only=True,
            use_reranker=use_reranker,
        )
    except KeyboardInterrupt:
        report("status", "interrupted")
        sys.exit(130)

    if ok:
        report("server", msg)
        try:
            search_semantic(query, limit, verbose=False, use_reranker=use_reranker)
        except KeyboardInterrupt:
            report("status", "interrupted")
            sys.exit(130)
        except Exception as e:
            _report_error(e)
            _report_relevant_service_diags(
                e,
                mode="search",
                include_reranker=use_reranker,
            )
            sys.exit(1)
        return

    try:
        with Live(
            Text(format_line("server", "checking")),
            console=console,
            screen=False,
            transient=True,
            refresh_per_second=4,
        ) as live:
            ok, msg = check_servers(
                on_status=lambda s: live.update(Text(format_line("server", s))),
                mode="search",
                use_reranker=use_reranker,
            )
    except KeyboardInterrupt:
        report("status", "interrupted")
        sys.exit(130)
    if not ok:
        _report_error(RuntimeError(msg))
        _report_relevant_service_diags(
            RuntimeError(msg),
            mode="search",
            include_reranker=use_reranker,
        )
        sys.exit(1)
    report("server", msg)
    try:
        search_semantic(query, limit, verbose=False, use_reranker=use_reranker)
    except KeyboardInterrupt:
        report("status", "interrupted")
        sys.exit(130)
    except Exception as e:
        _report_error(e)
        _report_relevant_service_diags(
            e,
            mode="search",
            include_reranker=use_reranker,
        )
        sys.exit(1)


_DOWNLOAD_SERVICES = [
    ("embedding", "com.sova.embedding", config.EMBEDDING_SERVER_URL),
    ("reranker", "com.sova.reranker", config.RERANKER_SERVER_URL),
    ("chat", "com.sova.chat", config.CONTEXT_SERVER_URL),
]
_DOWNLOAD_NAME_WIDTH = max(len(name) for name, _, _ in _DOWNLOAD_SERVICES)


def _run_download_mode() -> None:
    """Download all three model files by briefly starting each service."""
    report("mode", "download")
    needs_install = False
    downloaded_any = False
    for name, label, url in _DOWNLOAD_SERVICES:
        col = name.ljust(_DOWNLOAD_NAME_WIDTH)
        if not is_service_installed(label):
            report(
                "step",
                f"{col} | [yellow]not installed — run sova-install first[/yellow]",
            )
            needs_install = True
            continue
        if is_model_cached(label):
            report("step", f"{col} | cached")
            continue
        downloaded_any = True
        start_service(label)
        try:
            with Live(
                Text(format_line("step", f"{col} | starting")),
                console=console,
                refresh_per_second=2,
            ) as live:
                while True:
                    status = get_model_status(label)
                    live.update(Text(format_line("step", f"{col} | {status}")))
                    if is_model_cached(label):
                        break
                    time.sleep(1)
        except KeyboardInterrupt:
            report("status", "interrupted")
            stop_server(url, suppress_interrupt=True)
            sys.exit(130)
        stop_server(url)
        report("step", f"{col} | done")
    if needs_install:
        return
    if downloaded_any:
        report("status", "done")
    else:
        report("status", "all models cached")


def _run_list_mode() -> None:
    docs = find_docs()
    report("mode", f"list | {len(docs)} docs")
    try:
        list_docs(docs)
        show_stats(mode="list")
    except sqlite3.OperationalError as e:
        cause = str(e).strip() or "sqlite extension failed to initialize"
        _report_error_block(
            "database extension unavailable",
            cause=cause,
            action="reinstall and retry: sova-install",
        )
        sys.exit(1)
    except Exception as e:
        _report_error(e)
        sys.exit(1)


def _run_index_mode() -> None:
    if not config.get_docs_dir():
        _report_error_block(
            "docs directory is not configured",
            action="run: sova index /path/to/pdfs",
        )
        sys.exit(1)

    # Keep ASCII banner only for interactive indexing mode.
    console.print(SOVA_ASCII)

    try:
        conn = init_db()
    except Exception as e:
        _report_error_block("failed to initialize database", cause=str(e))
        sys.exit(1)
    report("database", "ready")
    try:
        signature_state = _sync_index_signatures(conn)
    except Exception as e:
        _report_error_block(
            "failed to synchronize index metadata",
            cause=str(e),
            action="retry indexing or inspect local database state",
        )
        sys.exit(1)

    docs = find_docs()

    start_time = time.time()
    interrupted = False
    failed = False
    prepared: list[tuple[str, int, list[dict], list[dict]]] = []

    global _ACTIVE_INDEX_VIEW
    _ACTIVE_INDEX_VIEW = _IndexLiveView()
    _ACTIVE_INDEX_VIEW.start()
    report("mode", f"index | {len(docs)} docs")

    try:
        # Phase 1: extract + context.
        report("event", "stopping search services")
        stop_server(config.EMBEDDING_SERVER_URL, suppress_interrupt=True)
        stop_server(config.RERANKER_SERVER_URL, suppress_interrupt=True)
        ok, msg = check_servers(
            on_status=lambda s: report("server", s),
            mode="index_context",
        )
        if not ok:
            failed = True
            _report_error(RuntimeError(msg))
            _report_relevant_service_diags(RuntimeError(msg), mode="index_context")
        else:
            report("server", msg)
            context_runtime_tick = _make_runtime_reporter(
                "index.context", "chat", mode="index"
            )
            context_runtime_tick(True)
            try:
                for doc in docs:
                    result = _prepare_doc(doc["name"], doc["pdf"], doc["md"], conn)
                    if result is None:
                        continue
                    doc_id, chunks, sections = result
                    _generate_contexts(
                        doc["name"],
                        doc_id,
                        chunks,
                        sections,
                        conn,
                        force_rebuild_context=signature_state.force_rebuild_context,
                        runtime_tick=context_runtime_tick,
                    )
                    prepared.append((doc["name"], doc_id, chunks, sections))
            except KeyboardInterrupt:
                interrupted = True
                report("status", "interrupt received, stopping services")
            except Exception as e:
                failed = True
                _report_error(e)
                _report_relevant_service_diags(e, mode="index_context")
            finally:
                stop_server(config.CONTEXT_SERVER_URL, suppress_interrupt=True)

        # Phase 2: embed all docs.
        if not interrupted and not failed:
            report("event", "switching to embedding")
            stop_server(config.EMBEDDING_SERVER_URL)
            time.sleep(2)
            ok, msg = check_servers(
                on_status=lambda s: report("server", s),
                mode="index_embed",
            )
            if not ok:
                failed = True
                _report_error(RuntimeError(msg))
                _report_relevant_service_diags(RuntimeError(msg), mode="index_embed")
            else:
                report("server", msg)
                embed_runtime_tick = _make_runtime_reporter(
                    "index.embed", "embedding", mode="index"
                )
                embed_runtime_tick(True)
                try:
                    report("embed", "canary probe")
                    run_embedding_canary()
                    for name, doc_id, chunks, sections in prepared:
                        report("doc", name)
                        _embed_doc(
                            name,
                            doc_id,
                            chunks,
                            sections,
                            conn,
                            force_rebuild_embed=signature_state.force_rebuild_embed,
                            runtime_tick=embed_runtime_tick,
                        )
                except KeyboardInterrupt:
                    interrupted = True
                    report("status", "interrupt received, stopping services")
                    stop_server(config.EMBEDDING_SERVER_URL, suppress_interrupt=True)
                except Exception as e:
                    failed = True
                    _report_error(e)
                    _report_relevant_service_diags(e, mode="index_embed")
                    stop_server(config.EMBEDDING_SERVER_URL, suppress_interrupt=True)

        # Phase 3: quantize.
        if not interrupted and not failed:
            report("quantize", "building index")
            quantize_vectors(conn)
            try:
                _commit_index_signatures(conn, signature_state)
            except Exception as e:
                failed = True
                _report_error_block(
                    "failed to finalize index metadata",
                    cause=str(e),
                    action="retry indexing",
                )

        get_cache().clear()
        if interrupted:
            stop_server(config.CONTEXT_SERVER_URL, suppress_interrupt=True)
            stop_server(config.EMBEDDING_SERVER_URL, suppress_interrupt=True)
            stop_server(config.RERANKER_SERVER_URL, suppress_interrupt=True)
            report("status", "services stopped")
    finally:
        conn.close()
        if _ACTIVE_INDEX_VIEW is not None:
            _ACTIVE_INDEX_VIEW.stop()
            _ACTIVE_INDEX_VIEW = None

    elapsed = fmt_duration(time.time() - start_time).strip()
    if interrupted:
        report("status", f"interrupted after {elapsed}")
        show_stats(mode="index")
        sys.exit(130)
    if failed:
        report("status", f"failed after {elapsed}")
        show_stats(mode="index")
        sys.exit(1)
    report("status", f"done in {elapsed}")
    show_stats(mode="index")


def _run_projects_mode() -> None:
    report("mode", "projects")
    rows = projects.list_projects()
    if not rows:
        report("projects", "none")
        report("hint", "run: sova index /path/to/pdfs")
        return
    table = make_table()
    table.add_column("Id")
    table.add_column("Docs", style="dim")
    for p in rows:
        table.add_row(
            p.project_id,
            _display_path(p.docs_dir),
        )
    render_table(table)


def _build_command_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="sova project CLI (default search: sova <project> <query>)",
        add_help=False,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("help", help="Show help", add_help=False)
    sub.add_parser("projects", help="List configured projects", add_help=False)
    sub.add_parser("download", help="Download all model files", add_help=False)

    p_remove = sub.add_parser("remove", help="Remove project from Sova", add_help=False)
    p_remove.add_argument("project", help="Project id/path")
    p_remove.add_argument(
        "--keep-data",
        action="store_true",
        help="Keep local project data under ~/.sova/projects/<id>",
    )

    p_list = sub.add_parser(
        "list", help="List docs and indexing status", add_help=False
    )
    p_list.add_argument("project", help="Project id/path")

    p_index = sub.add_parser("index", help="Index project docs", add_help=False)
    p_index.add_argument("project", help="Project id/path")
    return parser


def _run_command_cli(argv: list[str]) -> bool:
    commands = {
        "help",
        "projects",
        "download",
        "remove",
        "list",
        "index",
    }
    if not argv or argv[0] not in commands:
        return False
    parser = _build_command_parser()
    args = parser.parse_args(argv)

    if args.command == "help":
        _build_command_parser().print_help()
        return True
    if args.command == "projects":
        _run_projects_mode()
        return True
    if args.command == "download":
        _run_download_mode()
        return True
    if args.command == "remove":
        report("mode", "remove")
        try:
            removed = projects.remove_project(args.project, keep_data=args.keep_data)
        except ValueError as e:
            _report_error_block(
                "project not found",
                cause=str(e).replace("project not found: ", ""),
                action="run: sova projects",
            )
            sys.exit(1)
        report("project", f"removed {removed.project_id}")
        if args.keep_data:
            report("data", f"kept {_display_path(removed.root_dir)}")
        else:
            report("data", f"deleted {_display_path(removed.root_dir)}")
        return True
    project = _activate_project_from_ref(
        args.project,
        allow_create_from_dir=(args.command == "index"),
    )
    report("project", project.project_id)

    if args.command == "list":
        _run_list_mode()
        return True
    if args.command == "index":
        _run_index_mode()
        return True
    return True


def main() -> None:
    """Main entry point."""
    config.clear_active_project()
    try:
        try:
            argv = sys.argv[1:]
            if "--_watchdog" in argv:
                from sova.llama_client import cleanup_idle_services

                cleanup_idle_services()
                return
            if any(arg in {"-h", "--help"} for arg in argv):
                _report_error_block(
                    "unknown option",
                    cause=f"sova {' '.join(argv)}".rstrip(),
                    action="use: sova help",
                )
                sys.exit(2)
            if argv and argv[0] in {"list", "index", "remove"} and len(argv) == 1:
                _report_error_block(
                    "project is required",
                    cause=f"sova {argv[0]}",
                    action=f"use: sova {argv[0]} <project>",
                )
                sys.exit(2)
            known_commands = {"help", "projects", "download", "remove", "list", "index"}
            if len(argv) == 1 and argv[0] not in known_commands:
                only = Path(argv[0]).expanduser()
                if only.exists() and only.is_dir():
                    _report_error_block(
                        "query is required",
                        cause=f"sova {argv[0]}",
                        action="use: sova index /path/to/pdfs",
                    )
                    sys.exit(2)
                if projects.get_project(argv[0]) is None:
                    _report_error_block(
                        "unknown command or project",
                        cause=f"sova {argv[0]}",
                        action="run: sova projects",
                    )
                    sys.exit(2)
                _report_error_block(
                    "query is required",
                    cause=f"sova {argv[0]}",
                    action=f'use: sova {argv[0]} "<query>"',
                )
                sys.exit(2)
            if _run_command_cli(argv):
                return
            if argv:
                parser = argparse.ArgumentParser(
                    prog="sova",
                    description="Default search mode",
                )
                parser.add_argument("project", help="Project id/path")
                parser.add_argument("query", nargs="+", help="Search query text")
                parser.add_argument(
                    "-n",
                    "--limit",
                    type=int,
                    default=10,
                    help="Max results (default: 10)",
                )
                parser.add_argument(
                    "--reranker",
                    action="store_true",
                    default=config.SEARCH_USE_RERANKER,
                    help="Enable cross-encoder reranker (off by default)",
                )
                args = parser.parse_args(argv)
                project = _activate_project_from_ref(args.project)
                report("project", project.project_id)
                _run_search_mode(
                    " ".join(args.query),
                    args.limit,
                    use_reranker=bool(args.reranker),
                )
                return
            parser = _build_command_parser()
            parser.print_help()
            sys.exit(2)
        except projects.RegistryError as e:
            _report_error_block(
                "project registry is invalid",
                cause=str(e),
                action="fix ~/.sova/projects/registry.json or re-create it via indexing",
            )
            sys.exit(1)
        except KeyboardInterrupt:
            report("status", "interrupt received, stopping services")
            stop_server(config.CONTEXT_SERVER_URL, suppress_interrupt=True)
            stop_server(config.EMBEDDING_SERVER_URL, suppress_interrupt=True)
            stop_server(config.RERANKER_SERVER_URL, suppress_interrupt=True)
            report("status", "services stopped")
            report("status", "interrupted")
            sys.exit(130)
    finally:
        config.clear_active_project()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        report("status", "interrupt received, stopping services")
        stop_server(config.CONTEXT_SERVER_URL, suppress_interrupt=True)
        stop_server(config.EMBEDDING_SERVER_URL, suppress_interrupt=True)
        stop_server(config.RERANKER_SERVER_URL, suppress_interrupt=True)
        report("status", "services stopped")
        report("status", "interrupted")
        sys.exit(130)
