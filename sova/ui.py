"""Terminal output as a small, renderer-independent event stream."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TextIO

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

OutputMode = Literal["auto", "plain", "json"]


@dataclass(frozen=True)
class Event:
    """A stable user-facing event, independent of terminal rendering."""

    type: str
    message: str
    level: str = "info"
    phase: str | None = None
    item: str | None = None
    current: int | None = None
    total: int | None = None
    unit: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            key: value for key, value in asdict(self).items() if value is not None
        }
        if not payload["data"]:
            payload.pop("data")
        return payload


def _human_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _progress_text(event: Event) -> str:
    parts = [event.phase or "working"]
    if event.current is not None:
        amount = f"{event.current:,}"
        if event.total is not None:
            amount += f"/{event.total:,}"
        if event.unit:
            amount += f" {event.unit}"
        parts.append(amount)
    if event.item:
        parts.append(event.item)

    doc_index = event.data.get("document_index")
    doc_total = event.data.get("document_total")
    if doc_index is not None and doc_total is not None:
        parts.append(f"document {doc_index}/{doc_total}")

    rate = event.data.get("rate")
    if isinstance(rate, int | float) and rate > 0:
        parts.append(f"{rate:.2g} {event.unit or 'items'}/s")

    elapsed = event.data.get("elapsed_s")
    if isinstance(elapsed, int | float):
        parts.append(f"elapsed {_human_duration(float(elapsed))}")

    headroom = event.data.get("memory_headroom_gib")
    if isinstance(headroom, int | float) and headroom < 1.0:
        parts.append(f"{headroom:.1f} GiB free")
    return "  ".join(parts)


def _event_text(event: Event) -> str:
    """Render an optional work item as a stable first column."""
    if not event.item:
        return event.message
    raw_width = event.data.get("item_width", len(event.item))
    width = int(raw_width) if isinstance(raw_width, int | float) else len(event.item)
    return f"  {event.item.ljust(max(len(event.item), width))}  {event.message}"


def _status_text(event: Event) -> str:
    if event.type == "progress":
        return _progress_text(event)
    parts = [part for part in (event.phase, event.item, event.message) if part]
    return "  ".join(parts)


def _status_renderable(event: Event) -> Spinner:
    """Render every active operation as the same single status line."""
    return Spinner("dots", text=Text(_status_text(event)))


class _Renderer:
    def emit(self, event: Event) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None


class PlainRenderer(_Renderer):
    """Append-only output for pipes, CI, log files, and simple terminals."""

    def __init__(
        self,
        *,
        stdout: TextIO,
        stderr: TextIO,
        progress_interval_s: float = 30.0,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.progress_interval_s = progress_interval_s
        self._last_progress_at: dict[tuple[str | None, str | None], float] = {}
        self._last_status: str | None = None

    def emit(self, event: Event) -> None:
        if event.type == "result":
            self._render_result(event)
            return
        if event.type == "table":
            self._render_table(event)
            return
        if event.type == "progress":
            key = (event.phase, event.item)
            now = time.monotonic()
            final = (
                event.current is not None
                and event.total is not None
                and event.current >= event.total
            )
            if (
                not final
                and now - self._last_progress_at.get(key, 0.0)
                < self.progress_interval_s
            ):
                return
            self._last_progress_at[key] = now
            print(_progress_text(event), file=self.stderr, flush=True)
            return
        if event.type == "status":
            rendered = _status_text(event)
            if rendered == self._last_status:
                return
            self._last_status = rendered
        if event.type == "error":
            print(f"Error: {event.message}", file=self.stderr, flush=True)
            if cause := event.data.get("cause"):
                print(f"  {cause}", file=self.stderr, flush=True)
            if action := event.data.get("action"):
                print(f"  Try: {action}", file=self.stderr, flush=True)
            if detail := event.data.get("detail"):
                print(f"  {detail}", file=self.stderr, flush=True)
            return
        if event.type == "status":
            print(_status_text(event), file=self.stderr, flush=True)
        else:
            print(_event_text(event), file=self.stderr, flush=True)

    def _render_result(self, event: Event) -> None:
        record = event.data
        location = str(record.get("location", ""))
        score = record.get("score")
        if score is not None:
            print(f"{location}  {float(score):.2f}", file=self.stdout)
        else:
            print(location, file=self.stdout)
        if diagnostic := record.get("diagnostic"):
            print(f"  {diagnostic}", file=self.stdout)
        body = str(record.get("text", ""))
        for line in body.splitlines() or [body]:
            print(f"  {line}", file=self.stdout)
        if not record.get("last", False):
            print(file=self.stdout)

    def _render_table(self, event: Event) -> None:
        columns = [str(value) for value in event.data.get("columns", [])]
        rows = [[str(value) for value in row] for row in event.data.get("rows", [])]
        alignments = [str(value) for value in event.data.get("alignments", [])]
        sections = {int(value) for value in event.data.get("sections", [])}
        if event.message and event.message != "table":
            print(event.message, file=self.stdout)
        widths = [len(value) for value in columns]
        for row in rows:
            while len(widths) < len(row):
                widths.append(0)
            for index, value in enumerate(row):
                widths[index] = max(widths[index], len(value))
        if columns:
            print(
                "  ".join(
                    value.rjust(widths[index])
                    if index < len(alignments) and alignments[index] == "right"
                    else value.ljust(widths[index])
                    for index, value in enumerate(columns)
                ).rstrip(),
                file=self.stdout,
            )
        for row_index, row in enumerate(rows):
            if row_index in sections:
                print(file=self.stdout)
            print(
                "  ".join(
                    value.rjust(widths[index])
                    if index < len(alignments) and alignments[index] == "right"
                    else value.ljust(widths[index])
                    for index, value in enumerate(row)
                ).rstrip(),
                file=self.stdout,
            )


class JsonRenderer(_Renderer):
    """Newline-delimited JSON for agents and automation."""

    def __init__(self, stdout: TextIO) -> None:
        self.stdout = stdout

    def emit(self, event: Event) -> None:
        print(
            json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True),
            file=self.stdout,
            flush=True,
        )


class LiveRenderer(_Renderer):
    """A compact TTY renderer that updates one stable status region."""

    def __init__(self, *, stdout: TextIO, stderr: TextIO) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.console = Console(file=stderr, stderr=True, highlight=False)
        self._live: Live | None = None

    def _ensure_live(self) -> Live:
        if self._live is None:
            self._live = Live(
                console=self.console,
                refresh_per_second=4,
                transient=True,
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self._live.start()
        return self._live

    def _set_status(self, event: Event) -> None:
        self._ensure_live().update(_status_renderable(event), refresh=True)

    def _print_permanent(self, message: str, *, style: str | None = None) -> None:
        if self._live is not None:
            self._live.console.print(message, style=style)
        else:
            self.console.print(message, style=style)

    def emit(self, event: Event) -> None:
        if event.type in {"progress", "status", "interrupting"}:
            self._set_status(event)
            return
        if event.type == "result":
            self.close()
            PlainRenderer(stdout=self.stdout, stderr=self.stderr)._render_result(event)
            return
        if event.type == "table":
            self.close()
            PlainRenderer(stdout=self.stdout, stderr=self.stderr)._render_table(event)
            return
        if event.type == "error":
            self.close()
            self._print_permanent(f"Error: {event.message}", style="bold red")
            if cause := event.data.get("cause"):
                self._print_permanent(f"  {cause}")
            if action := event.data.get("action"):
                self._print_permanent(f"  Try: {action}")
            if detail := event.data.get("detail"):
                self._print_permanent(f"  {detail}", style="dim")
            return
        if event.type in {"completed", "failed", "interrupted", "cancelled"}:
            self.close()
        style = "yellow" if event.level == "warning" else None
        self._print_permanent(_event_text(event), style=style)

    def close(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None


class Reporter:
    """Stateful event publisher shared by all CLI commands."""

    def __init__(self, renderer: _Renderer) -> None:
        self.renderer = renderer
        self.document_index: int | None = None
        self.document_total: int | None = None
        self._progress_started: dict[tuple[str, str | None], tuple[float, int]] = {}
        self._memory_headroom_gib: float | None = None
        self._low_memory_reported = False

    def emit(self, event: Event) -> None:
        self.renderer.emit(event)

    def scope(self, document_index: int | None, document_total: int | None) -> None:
        self.document_index = document_index
        self.document_total = document_total

    def status(
        self,
        message: str,
        *,
        phase: str | None = None,
        item: str | None = None,
    ) -> None:
        self.emit(Event("status", message, phase=phase, item=item))

    def progress(
        self,
        phase: str,
        current: int,
        total: int,
        *,
        item: str | None = None,
        unit: str = "items",
    ) -> None:
        key = (phase, item)
        now = time.monotonic()
        started, initial = self._progress_started.setdefault(key, (now, current))
        elapsed = max(0.0, now - started)
        advanced = max(0, current - initial)
        data: dict[str, Any] = {"elapsed_s": round(elapsed, 3)}
        if elapsed > 0 and advanced > 0:
            data["rate"] = round(advanced / elapsed, 4)
        if self.document_index is not None and self.document_total is not None:
            data["document_index"] = self.document_index
            data["document_total"] = self.document_total
        if self._memory_headroom_gib is not None:
            data["memory_headroom_gib"] = self._memory_headroom_gib
        self.emit(
            Event(
                "progress",
                "",
                phase=phase,
                item=item,
                current=current,
                total=total,
                unit=unit,
                data=data,
            )
        )

    def runtime(self, *, memory_headroom_gib: float | None) -> None:
        self._memory_headroom_gib = memory_headroom_gib
        if memory_headroom_gib is None:
            return
        if memory_headroom_gib < 0.5 and not self._low_memory_reported:
            self._low_memory_reported = True
            self.emit(
                Event(
                    "warning",
                    f"Low memory: {memory_headroom_gib:.1f} GiB free",
                    level="warning",
                    data={"memory_headroom_gib": memory_headroom_gib},
                )
            )
        elif memory_headroom_gib >= 1.0:
            self._low_memory_reported = False

    def close(self) -> None:
        self.renderer.close()


_reporter: Reporter | None = None
_output_mode: OutputMode = "auto"


def configure_output(mode: OutputMode = "auto") -> Reporter:
    """Configure output for one CLI invocation."""
    global _output_mode, _reporter
    _output_mode = mode
    if _reporter is not None:
        _reporter.close()
    if mode == "json":
        renderer: _Renderer = JsonRenderer(sys.stdout)
    elif mode == "plain" or not sys.stderr.isatty():
        renderer = PlainRenderer(stdout=sys.stdout, stderr=sys.stderr)
    else:
        renderer = LiveRenderer(stdout=sys.stdout, stderr=sys.stderr)
    _reporter = Reporter(renderer)
    return _reporter


def is_json_output() -> bool:
    return _output_mode == "json"


def get_reporter() -> Reporter:
    global _reporter
    if _reporter is None:
        _reporter = Reporter(PlainRenderer(stdout=sys.stdout, stderr=sys.stderr))
    return _reporter


def close_output() -> None:
    global _reporter
    if _reporter is not None:
        _reporter.close()
    _reporter = None


def emit(
    event_type: str,
    message: str,
    *,
    level: str = "info",
    phase: str | None = None,
    item: str | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    get_reporter().emit(
        Event(
            event_type,
            message,
            level=level,
            phase=phase,
            item=item,
            data=data or {},
        )
    )


def status(
    message: str,
    *,
    phase: str | None = None,
    item: str | None = None,
) -> None:
    get_reporter().status(message, phase=phase, item=item)


def progress(
    phase: str,
    current: int,
    total: int,
    *,
    item: str | None = None,
    unit: str = "items",
) -> None:
    get_reporter().progress(phase, current, total, item=item, unit=unit)


def scope(document_index: int | None, document_total: int | None) -> None:
    get_reporter().scope(document_index, document_total)


def runtime(*, memory_headroom_gib: float | None) -> None:
    get_reporter().runtime(memory_headroom_gib=memory_headroom_gib)


def result(data: dict[str, Any]) -> None:
    emit("result", str(data.get("location", "result")), data=data)


def report_error(
    summary: str,
    *,
    cause: str | None = None,
    action: str | None = None,
    detail: str | None = None,
) -> None:
    data = {
        key: value
        for key, value in {"cause": cause, "action": action, "detail": detail}.items()
        if value
    }
    emit("error", summary, level="error", data=data)


class Table:
    """Small structured table that renders consistently in text and JSON."""

    def __init__(self, *, title: str | None = None, show_header: bool = True) -> None:
        self.title = title
        self.show_header = show_header
        self._columns: list[str] = []
        self._alignments: list[str] = []
        self._rows: list[tuple[str, ...]] = []
        self._sections: list[int] = []

    def add_column(self, header: str = "", **style: Any) -> None:
        self._columns.append(str(header))
        self._alignments.append(str(style.get("justify", "left")))

    def add_row(self, *cells: object) -> None:
        self._rows.append(tuple("" if cell is None else str(cell) for cell in cells))

    def add_section(self) -> None:
        if self._rows and len(self._rows) not in self._sections:
            self._sections.append(len(self._rows))

    @property
    def row_count(self) -> int:
        return len(self._rows)


def make_table(
    *,
    title: str | None = None,
    show_header: bool = True,
    header_style: str = "dim",
) -> Table:
    del header_style
    return Table(title=title, show_header=show_header)


def render_table(
    table: Table, *, gap_before: bool = False, gap_after: bool = False
) -> None:
    del gap_before, gap_after
    emit(
        "table",
        table.title or "table",
        data={
            "columns": table._columns if table.show_header else [],
            "alignments": table._alignments,
            "rows": table._rows,
            "sections": table._sections,
        },
    )


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:>6.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:>6.1f}m"
    return f"{seconds / 3600:>6.1f}h"
