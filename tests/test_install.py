"""Installer behavior tests."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from sova import install


@pytest.mark.parametrize(
    ("shell", "filename", "marker"),
    [
        ("bash", "sova.sh", "__sova_projects"),
        ("zsh", "_sova", "__sova_projects"),
        ("fish", "sova.fish", "__fish_sova_projects"),
    ],
)
def test_shell_completions_are_native_and_do_not_invoke_sova(
    monkeypatch, tmp_path: Path, shell: str, filename: str, marker: str
) -> None:
    completion_dir = tmp_path / shell
    monkeypatch.setattr(
        install, "_bash_completion_path", lambda: completion_dir / "sova.sh"
    )
    monkeypatch.setattr(
        install, "_zsh_completion_path", lambda: completion_dir / "_sova"
    )
    monkeypatch.setattr(install, "_fish_completions_dir", lambda: completion_dir)

    assert install._install_shell_completion(shell) is True

    script = (completion_dir / filename).read_text(encoding="utf-8")
    assert marker in script
    assert "projects/registry.json" in script
    assert "_SOVA_COMPLETE" not in script
    assert "_TYPER_COMPLETE" not in script
    assert " command sova" not in script


def test_fish_completion_returns_commands_and_projects_without_running_sova(
    monkeypatch, tmp_path: Path
) -> None:
    fish = shutil.which("fish")
    if fish is None:
        pytest.skip("fish is not installed")

    completion_dir = tmp_path / "fish" / "completions"
    sova_home = tmp_path / "sova-home"
    registry = sova_home / "projects" / "registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            {
                "projects": [
                    {
                        "id": "operating-system-documents",
                        "docs_dir": "/documents",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "local-docs").mkdir()
    (tmp_path / "not-a-project.txt").write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(install, "_fish_completions_dir", lambda: completion_dir)
    install._install_native_completion("fish")

    env = {
        **os.environ,
        "SOVA_HOME": str(sova_home),
        "SOVA_COMPLETION_TEST_SCRIPT": str(completion_dir / "sova.fish"),
    }

    def complete(commandline: str) -> str:
        result = subprocess.run(
            [
                fish,
                "-N",
                "-c",
                'source "$SOVA_COMPLETION_TEST_SCRIPT"; complete -C "$argv[1]"',
                "--",
                commandline,
            ],
            env=env,
            cwd=tmp_path,
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout

    root = complete("sova ")
    assert "search\tSearch project documents" in root
    assert "operating-system-documents\tProject" in root

    project = complete("sova search ")
    assert "operating-system-documents\tProject" in project
    assert "local-docs/\tDirectory" in project
    assert "not-a-project.txt" not in project


def test_bash_completion_returns_commands_projects_and_directories(
    monkeypatch, tmp_path: Path
) -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not installed")

    completion_path = tmp_path / "sova.sh"
    sova_home = tmp_path / "sova-home"
    registry = sova_home / "projects" / "registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps({"projects": [{"id": "reference-docs", "docs_dir": "/documents"}]}),
        encoding="utf-8",
    )
    (tmp_path / "local-docs").mkdir()
    (tmp_path / "not-a-project.txt").write_text("file", encoding="utf-8")
    monkeypatch.setattr(install, "_bash_completion_path", lambda: completion_path)
    install._install_native_completion("bash")

    env = {
        **os.environ,
        "SOVA_HOME": str(sova_home),
        "SOVA_COMPLETION_TEST_SCRIPT": str(completion_path),
    }

    def complete(words: list[str], current: int) -> list[str]:
        encoded = " ".join(json.dumps(word) for word in words)
        command = (
            'source "$SOVA_COMPLETION_TEST_SCRIPT"; '
            f"COMP_WORDS=({encoded}); COMP_CWORD={current}; "
            "_sova_completion; printf '%s\\n' \"${COMPREPLY[@]}\""
        )
        result = subprocess.run(
            [
                bash,
                "--noprofile",
                "--norc",
                "-c",
                command,
            ],
            env=env,
            cwd=tmp_path,
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.splitlines()

    root = complete(["sova", ""], 1)
    assert "search" in root
    assert "reference-docs" in root

    project = complete(["sova", "search", ""], 2)
    assert "reference-docs" in project
    assert "local-docs" in project
    assert "not-a-project.txt" not in project


def test_zsh_completion_has_valid_syntax(monkeypatch, tmp_path: Path) -> None:
    zsh = shutil.which("zsh")
    if zsh is None:
        pytest.skip("zsh is not installed")

    completion_path = tmp_path / "_sova"
    monkeypatch.setattr(install, "_zsh_completion_path", lambda: completion_path)
    install._install_native_completion("zsh")

    subprocess.run(
        [zsh, "-n", str(completion_path)],
        capture_output=True,
        check=True,
        text=True,
    )
