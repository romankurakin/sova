"""Public benchmark commands have one entry point and no stage-toggle flags."""

from typer.testing import CliRunner

from benchmarks import cli


def test_help_lists_full_workflow():
    result = CliRunner().invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for command in ("prepare", "judge", "run", "report"):
        assert command in result.output
        help_result = CliRunner().invoke(cli.app, [command, "--help"])
        assert help_result.exit_code == 0
        for flag in ("--offline", "--source-dir", "--data-dir"):
            assert flag not in help_result.output
    result = CliRunner().invoke(cli.app, ["prepare", "--help"])
    for flag in ("--extract", "--chunk", "--resume", "--catch-up"):
        assert flag not in result.output
    result = CliRunner().invoke(cli.app, ["judge", "--help"])
    assert "--max-duration" not in result.output
    assert "--max-pairs" in result.output


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
