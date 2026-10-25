"""External judge isolation and explicit failure semantics without inference."""

import json

import pytest

from benchmarks.judge import JudgeCLI, classify_error, subscription_environment
from benchmarks.judging_schema import GradeAnswer


def test_subscription_environment_excludes_api_credentials(monkeypatch):
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"):
        monkeypatch.setenv(key, "must-not-inherit")
    monkeypatch.setenv("USER", "keychain-account")
    env = subscription_environment()
    assert env["USER"] == "keychain-account"
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_BASE_URL" not in env


@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_command_disables_tools_and_has_structured_output(tmp_path, provider):
    cli = object.__new__(JudgeCLI)
    cli.provider = provider
    cli.executable = provider
    from benchmarks.build_suite import judge_config

    cli.config = judge_config()[provider]
    command = cli.command(tmp_path, GradeAnswer.model_json_schema())
    if provider == "codex":
        assert "--ignore-user-config" in command
        assert 'forced_login_method="chatgpt"' in command
        assert "features.shell_tool=false" in command
        assert "--output-schema" in command
        assert (
            json.loads((tmp_path / "schema.json").read_text())["additionalProperties"]
            is False
        )
    else:
        assert "--bare" not in command
        assert command[command.index("--tools") + 1] == ""
        assert "--safe-mode" in command
        assert "--strict-mcp-config" in command


def test_only_confirmed_quota_errors_are_quota():
    assert classify_error("usage limit reached") == "paused_quota"
    for message in (
        "You've hit your session limit · resets 3:45pm",
        "You’ve hit your weekly limit · resets Mon 12:00am",
        "You've hit your Sonnet limit · resets 3:45pm",
        "Usage limit reached. You've reached your usage limit.",
        "You're out of credits. Your workspace is out of credits.",
        "You've reached your workspace credit limit",
    ):
        assert classify_error(message) == "paused_quota", message
    throttle = (
        "API Error: Server is temporarily limiting requests (not your usage limit)"
    )
    assert classify_error(throttle) == "unknown_error"
    assert classify_error("unauthorized") == "authentication_error"
    assert classify_error("maximum context length") == "input_length_error"
    assert classify_error("unexpected 500") == "unknown_error"
