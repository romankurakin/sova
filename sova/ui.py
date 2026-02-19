"""Shared Rich UI helpers."""

from collections.abc import Callable

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

console = Console()
_LABEL_WIDTH = 9


def _label(name: str) -> str:
    padded = f"{name}:".ljust(_LABEL_WIDTH)
    return f"{padded}"


def format_line(name: str, msg: str) -> str:
    """Build one formatted output line."""
    return f"{_label(name)} {msg}"


def report(name: str, msg: str) -> None:
    console.print(format_line(name, msg))


def print_gap(lines: int = 1) -> None:
    """Print vertical spacing between output sections."""
    for _ in range(max(0, lines)):
        console.print()


def make_table(
    *,
    title: str | None = None,
    show_header: bool = True,
    header_style: str = "dim",
) -> Table:
    """Create a table with shared CLI defaults."""
    return Table(title=title, show_header=show_header, header_style=header_style)


def render_table(table: Table, *, gap_before: bool = False, gap_after: bool = False) -> None:
    """Render a table with optional spacing around it."""
    if gap_before:
        print_gap()
    console.print(table)
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
    report("error", f"[red]{summary}[/red]")
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


def with_status(
    make_text: Callable[[str], str],
    func: Callable[[], tuple[bool, str]],
) -> tuple[bool, str]:
    """Run a health/probe function while showing a spinner status line."""
    with console.status("", spinner="dots") as status:
        ok, msg = func()
        status.update(make_text(msg))
    return ok, msg


def report_progress(name: str) -> Progress:
    return Progress(
        TextColumn(_label(name)),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:>6.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:>6.1f}m"
    return f"{seconds / 3600:>6.1f}h"
