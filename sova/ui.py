"""Shared Rich UI helpers."""

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

console = Console()


def _label(name: str) -> str:
    padded = f"{name}:".ljust(8)
    return f"{padded}"


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


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:>6.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:>6.1f}m"
    return f"{seconds / 3600:>6.1f}h"
