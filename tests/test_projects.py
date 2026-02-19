"""Tests for project registry behavior."""

import json
from pathlib import Path

import pytest

from sova import projects


def _isolate_registry(monkeypatch, tmp_path: Path) -> Path:
    root = tmp_path / "projects"
    monkeypatch.setattr(projects, "_PROJECTS_ROOT", root)
    monkeypatch.setattr(projects, "_REGISTRY_PATH", root / "registry.json")
    return root


def test_reserved_command_name_raises(monkeypatch, tmp_path):
    _isolate_registry(monkeypatch, tmp_path)
    docs = tmp_path / "list"
    docs.mkdir(parents=True)

    with pytest.raises(ValueError, match="reserved project id: list"):
        projects.add_project(docs)


def test_duplicate_folder_name_raises(monkeypatch, tmp_path):
    _isolate_registry(monkeypatch, tmp_path)
    docs1 = tmp_path / "docs-parent-1" / "docs"
    docs2 = tmp_path / "docs-parent-2" / "docs"
    docs1.mkdir(parents=True)
    docs2.mkdir(parents=True)
    projects.add_project(docs1)
    with pytest.raises(ValueError, match="project id already exists: docs"):
        projects.add_project(docs2)


def test_legacy_name_field_is_ignored_and_id_kept(monkeypatch, tmp_path):
    _isolate_registry(monkeypatch, tmp_path)
    docs = tmp_path / "foo"
    docs.mkdir(parents=True)
    reg = {
        "version": 1,
        "projects": [
            {
                "id": "legacy-id",
                "name": "legacy-name",
                "docs_dir": str(docs),
            }
        ],
    }
    projects._REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    projects._REGISTRY_PATH.write_text(
        json.dumps(reg, indent=2) + "\n",
        encoding="utf-8",
    )

    rows = projects.list_projects()

    assert len(rows) == 1
    assert rows[0].project_id == "legacy-id"
    # We do not rewrite registry on read; legacy fields are simply ignored at runtime.
    saved = json.loads(projects._REGISTRY_PATH.read_text(encoding="utf-8"))
    assert saved["projects"][0]["id"] == "legacy-id"
    assert saved["projects"][0]["name"] == "legacy-name"


def test_remove_project_deletes_local_data_by_default(monkeypatch, tmp_path):
    _isolate_registry(monkeypatch, tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir(parents=True)
    project = projects.add_project(docs)
    assert project.root_dir.exists()

    projects.remove_project(project.project_id)

    assert not project.root_dir.exists()


def test_remove_project_can_keep_local_data(monkeypatch, tmp_path):
    _isolate_registry(monkeypatch, tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir(parents=True)
    project = projects.add_project(docs)
    assert project.root_dir.exists()

    projects.remove_project(project.project_id, keep_data=True)

    assert project.root_dir.exists()
