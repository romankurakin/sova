"""Tests for the single boundary around third-party tools."""

import io
import subprocess
import sys

import pytest

from sova.external import ExternalToolError, call_external, run_process


def test_call_external_discards_output_on_success(capsys):
    def noisy_tool() -> str:
        print("third-party stdout")
        print("third-party stderr", file=sys.stderr)
        return "ok"

    assert call_external("noisy tool", noisy_tool) == "ok"
    assert capsys.readouterr() == ("", "")


def test_call_external_normalizes_and_bounds_failure():
    def broken_tool() -> None:
        for index in range(10):
            print(f"noise {index}", file=sys.stderr)
        raise ValueError("invalid input")

    with pytest.raises(ExternalToolError) as caught:
        call_external("converter", broken_tool)

    message = str(caught.value)
    assert message.startswith("converter: invalid input")
    assert "noise 9" in message
    assert "noise 0" not in message


def test_run_process_captures_output_instead_of_streaming(capsys):
    result = run_process(
        [
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr)",
        ],
        operation="test command",
    )

    assert result.stdout == "out\n"
    assert result.stderr == "err\n"
    assert capsys.readouterr() == ("", "")


def test_run_process_turns_nonzero_exit_into_one_error():
    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            ["tool"], 7, stdout="details\n", stderr="bad option\n"
        )

    with pytest.raises(ExternalToolError, match="exited with status 7"):
        run_process(["tool"], runner=runner, check=True, operation="conversion")


def test_message_adapter_is_restored_after_failure():
    active = io.StringIO()
    restored: list[bool] = []

    def adapter(target):
        nonlocal active
        previous = active
        active = target

        def restore() -> None:
            nonlocal active
            active = previous
            restored.append(True)

        return restore

    def broken() -> None:
        active.write("library diagnostic\n")
        raise RuntimeError("boom")

    with pytest.raises(ExternalToolError, match="library diagnostic"):
        call_external("library", broken, messages=adapter)
    assert restored == [True]
