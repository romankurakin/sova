"""Public benchmark commands have one entry point and no stage-toggle flags."""

from click import unstyle
from typer.testing import CliRunner

from benchmarks import cli


def help_text(*args):
    # Typer forces Rich colors when GITHUB_ACTIONS is set, splitting flags with ANSI codes.
    result = CliRunner().invoke(cli.app, [*args, "--help"])
    assert result.exit_code == 0
    return unstyle(result.output)


def test_help_lists_full_workflow():
    output = help_text()
    for command in ("prepare", "judge", "run", "report"):
        assert command in output
        command_help = help_text(command)
        for flag in ("--offline", "--source-dir", "--data-dir"):
            assert flag not in command_help
    prepare_help = help_text("prepare")
    for flag in ("--extract", "--chunk", "--resume", "--catch-up"):
        assert flag not in prepare_help
    judge_help = help_text("judge")
    assert "--max-duration" not in judge_help
    assert "--max-pairs" in judge_help


def test_report_before_preparation_does_not_start_models(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DEFAULT_DATA", tmp_path)
    result = CliRunner().invoke(cli.app, ["report"])
    assert result.exit_code == 0
    assert "preparation_incomplete" in result.output


def test_judge_before_pool_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DEFAULT_DATA", tmp_path)
    result = CliRunner().invoke(cli.app, ["judge", "codex"])
    assert result.exit_code == 1
    assert "prepare" in result.output
