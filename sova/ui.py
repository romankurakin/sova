"""Shared plain-text UI helpers."""

import time
from typing import Self

_LABEL_WIDTH = 9

# Sentinel marking a section break between table rows.
_SECTION_BREAK: tuple[str, ...] = ("\x00section-break\x00",)


def _label(name: str) -> str:
    padded = f"{name}:".ljust(_LABEL_WIDTH)
    return f"{padded}"


def format_line(name: str, msg: str) -> str:
    """Build one formatted output line."""
    return f"{_label(name)} {msg}"


def report(name: str, msg: str) -> None:
    print(format_line(name, msg), flush=True)


def print_gap(lines: int = 1) -> None:
    """Print vertical spacing between output sections."""
    for _ in range(max(0, lines)):
        print()


class Table:
    """Minimal aligned plain-text table."""

    def __init__(self, *, title: str | None = None, show_header: bool = True) -> None:
        self.title = title
        self.show_header = show_header
        self._columns: list[dict] = []
        self._rows: list[tuple[str, ...]] = []

    def add_column(self, header: str = "", *, justify: str = "left", **_style) -> None:
        self._columns.append({"header": str(header), "justify": justify})

    def add_row(self, *cells: object) -> None:
        self._rows.append(tuple("" if c is None else str(c) for c in cells))

    def add_section(self) -> None:
        """Mark a section break, rendered as a blank line."""
        self._rows.append(_SECTION_BREAK)

    @property
    def row_count(self) -> int:
        return sum(1 for row in self._rows if row is not _SECTION_BREAK)

    def render_lines(self) -> list[str]:
        data_rows = [row for row in self._rows if row is not _SECTION_BREAK]
        n_cols = max([len(self._columns)] + [len(row) for row in data_rows] or [0])
        if n_cols == 0:
            return []
        columns = list(self._columns)
        while len(columns) < n_cols:
            columns.append({"header": "", "justify": "left"})

        grid: list[tuple[str, ...] | None] = []
        if self.show_header and any(c["header"] for c in columns):
            grid.append(tuple(c["header"] for c in columns))
        for row in self._rows:
            if row is _SECTION_BREAK:
                grid.append(None)
            else:
                grid.append(row + ("",) * (n_cols - len(row)))

        widths = [
            max(
                (len(row[i]) for row in grid if row is not None and i < len(row)),
                default=0,
            )
            for i in range(n_cols)
        ]
        lines: list[str] = []
        if self.title:
            lines.append(self.title)
        for row in grid:
            if row is None:
                lines.append("")
                continue
            cells = []
            for i, cell in enumerate(row):
                if columns[i]["justify"] == "right":
                    cells.append(cell.rjust(widths[i]))
                else:
                    cells.append(cell.ljust(widths[i]))
            lines.append("  ".join(cells).rstrip())
        return lines


def make_table(
    *,
    title: str | None = None,
    show_header: bool = True,
    header_style: str = "dim",
) -> Table:
    """Create a table with shared CLI defaults."""
    del header_style
    return Table(title=title, show_header=show_header)


def render_table(
    table: Table, *, gap_before: bool = False, gap_after: bool = False
) -> None:
    """Render a table with optional spacing around it."""
    if gap_before:
        print_gap()
    for line in table.render_lines():
        print(line)
    if gap_after:
        print_gap()


def report_error(
    summary: str,
    *,
    cause: str | None = None,
    action: str | None = None,
    detail: str | None = None,
) -> None:
    """Print a structured, user-facing error block."""
    report("error", summary)
    if cause:
        report("cause", cause)
    if action:
        report("action", action)
    if detail:
        report("detail", detail)


def report_mode(mode: str, detail: str | None = None) -> None:
    """Print mode header in a consistent, low-noise format."""
    if detail:
        report("mode", f"{mode} | {detail}")
    else:
        report("mode", mode)


def report_step(step: str, detail: str | None = None) -> None:
    """Print pipeline step marker."""
    if detail:
        report("step", f"{step} | {detail}")
    else:
        report("step", step)


def report_runtime(summary: str) -> None:
    """Print compact runtime status."""
    report("runtime", summary)


def report_event(message: str) -> None:
    """Print generic event line."""
    report("event", message)


class Progress:
    """Throttled plain-text progress reporting."""

    def __init__(self, name: str, min_interval_s: float = 5.0) -> None:
        self._name = name
        self._min_interval_s = min_interval_s
        self._tasks: list[dict] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            for task in self._tasks:
                self._emit(task, force=True)
        return False

    def add_task(
        self, description: str = "", total: int | None = None, completed: int = 0, **_kw
    ) -> int:
        del description
        self._tasks.append(
            {"total": total, "done": completed, "last_ts": 0.0, "printed": None}
        )
        return len(self._tasks) - 1

    def update(self, task_id: int, advance: int = 1, **_kw) -> None:
        task = self._tasks[task_id]
        task["done"] += advance
        done_all = task["total"] is not None and task["done"] >= task["total"]
        self._emit(task, force=done_all)

    def _emit(self, task: dict, force: bool = False) -> None:
        if task["printed"] == task["done"]:
            return
        now = time.monotonic()
        if not force and (now - task["last_ts"]) < self._min_interval_s:
            return
        task["last_ts"] = now
        task["printed"] = task["done"]
        if task["total"] is not None:
            report(self._name, f"{task['done']:,}/{task['total']:,}")
        else:
            report(self._name, f"{task['done']:,}")


def report_progress(name: str) -> Progress:
    return Progress(name)


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:>6.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:>6.1f}m"
    return f"{seconds / 3600:>6.1f}h"
