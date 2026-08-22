"""Quiet boundary around third-party code and command-line tools.

External programs never own Sova's terminal. Their output is captured here and
is only surfaced as a concise, structured error by the caller.
"""

from __future__ import annotations

import contextlib
import io
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

MessageAdapter = Callable[[TextIO], Callable[[], None] | None]

_CAPTURE_LOCK = threading.RLock()
_MAX_ERROR_LINES = 4


@dataclass(frozen=True)
class ExternalOutput:
    """Captured diagnostics from one external operation."""

    stdout: str = ""
    stderr: str = ""
    messages: str = ""

    def summary(self) -> str | None:
        """Return a bounded, useful diagnostic instead of a raw output dump."""
        lines: list[str] = []
        for block in (self.stderr, self.messages, self.stdout):
            for line in block.splitlines():
                clean = line.strip()
                if clean and clean not in lines:
                    lines.append(clean)
        if not lines:
            return None
        return " | ".join(lines[-_MAX_ERROR_LINES:])


class ExternalToolError(RuntimeError):
    """Normalized failure raised at Sova's external-tool boundary."""

    def __init__(
        self,
        operation: str,
        cause: str,
        *,
        output: ExternalOutput | None = None,
    ) -> None:
        self.operation = operation
        self.cause = cause.strip() or "external operation failed"
        self.output = output or ExternalOutput()
        detail = self.output.summary()
        message = f"{operation}: {self.cause}"
        if detail and detail not in self.cause:
            message += f" ({detail})"
        super().__init__(message)


def call_external[T](
    operation: str,
    action: Callable[[], T],
    *,
    messages: MessageAdapter | None = None,
) -> T:
    """Run synchronous third-party code without letting it write to the UI."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    message_stream = io.StringIO()

    # Redirecting process-wide Python streams is safe here because Sova's CLI
    # executes external library calls serially. The lock also makes that
    # contract explicit if background workers are added later.
    with _CAPTURE_LOCK:
        restore = messages(message_stream) if messages else None
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                return action()
        except ExternalToolError:
            raise
        except Exception as exc:
            output = ExternalOutput(
                stdout=stdout.getvalue(),
                stderr=stderr.getvalue(),
                messages=message_stream.getvalue(),
            )
            raise ExternalToolError(operation, str(exc), output=output) from exc
        finally:
            if restore:
                restore()


def run_process(
    args: Sequence[str | Path],
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]] | None = None,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    check: bool = False,
    text: bool = True,
    operation: str | None = None,
) -> subprocess.CompletedProcess[Any]:
    """Run a subprocess with captured output and normalized failures."""
    if runner is None:
        runner = subprocess.run
    command = [str(value) for value in args]
    label = operation or (Path(command[0]).name if command else "command")
    try:
        result = runner(
            command,
            cwd=cwd,
            env=dict(env) if env is not None else None,
            capture_output=True,
            check=False,
            text=text,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        output = ExternalOutput(
            stdout=_as_text(exc.stdout),
            stderr=_as_text(exc.stderr),
        )
        raise ExternalToolError(
            label, f"timed out after {timeout:g}s", output=output
        ) from exc
    except OSError as exc:
        raise ExternalToolError(label, str(exc)) from exc

    if check and result.returncode != 0:
        output = ExternalOutput(
            stdout=_as_text(result.stdout),
            stderr=_as_text(result.stderr),
        )
        raise ExternalToolError(
            label, f"exited with status {result.returncode}", output=output
        )
    return result


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
