"""Project registry and per-project storage layout."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from sova import config

_PROJECTS_ROOT = config.SOVA_HOME / "projects"
_REGISTRY_PATH = _PROJECTS_ROOT / "registry.json"
# Must stay in sync with the CLI command names in sova.cli.
RESERVED_PROJECT_IDS = frozenset(
    {
        "help",
        "projects",
        "doctor",
        "download",
        "remove",
        "list",
        "index",
        "search",
    }
)


class RegistryError(ValueError):
    """Raised when project registry format/content is invalid."""


def is_reserved_project_id(value: str) -> bool:
    """Return True when value collides with top-level CLI words."""
    return value.strip().lower() in RESERVED_PROJECT_IDS


@dataclass(frozen=True)
class Project:
    """Resolved project metadata and storage paths."""

    project_id: str
    docs_dir: Path
    root_dir: Path
    data_dir: Path
    db_path: Path


def _normalize(path: Path) -> Path:
    return path.expanduser().resolve()


def _default_registry() -> dict:
    return {"projects": []}


def _save_registry(reg: dict) -> None:
    _PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = _REGISTRY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(_REGISTRY_PATH)


def _validate_project_id(project_id: str, taken: set[str]) -> None:
    if not project_id:
        raise ValueError("invalid project id")
    if project_id.lower() in RESERVED_PROJECT_IDS:
        raise ValueError(
            f"reserved project id: {project_id} (rename docs folder and retry)"
        )
    if project_id in taken:
        raise ValueError(f"project id already exists: {project_id}")


def _load_registry() -> dict:
    if not _REGISTRY_PATH.exists():
        return _default_registry()

    try:
        raw = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        raise RegistryError(f"failed to parse registry JSON: {e}") from e

    if not isinstance(raw, dict):
        raise RegistryError("registry root must be a JSON object")

    raw_projects = raw.get("projects")
    if not isinstance(raw_projects, list):
        raise RegistryError("registry.projects must be a list")

    projects_list: list[dict] = []
    seen_ids: set[str] = set()

    for entry in raw_projects:
        if not isinstance(entry, dict):
            raise RegistryError("each registry.projects entry must be a JSON object")

        raw_docs = entry.get("docs_dir")
        if not isinstance(raw_docs, str):
            raise RegistryError("each project entry must include string docs_dir")

        docs_dir = str(_normalize(Path(raw_docs)))
        raw_id = entry.get("id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise RegistryError("each project entry must include string id")
        project_id = raw_id.strip()
        if is_reserved_project_id(project_id):
            raise RegistryError(f"reserved project id in registry: {project_id}")
        if project_id in seen_ids:
            raise RegistryError(f"duplicate project id in registry: {project_id}")
        seen_ids.add(project_id)
        projects_list.append({"id": project_id, "docs_dir": docs_dir})

    return {"projects": projects_list}


def _entry_to_project(entry: dict) -> Project:
    project_id = str(entry["id"])
    root_dir = _PROJECTS_ROOT / project_id
    return Project(
        project_id=project_id,
        docs_dir=_normalize(Path(str(entry["docs_dir"]))),
        root_dir=root_dir,
        data_dir=root_dir / "data",
        db_path=root_dir / "indexed.db",
    )


def _existing_ids(reg: dict) -> set[str]:
    ids: set[str] = set()
    for entry in reg.get("projects", []):
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            ids.add(entry["id"])
    return ids


def _find_by_docs_dir(reg: dict, docs_dir: Path) -> Project | None:
    target = str(_normalize(docs_dir))
    for entry in reg.get("projects", []):
        if not isinstance(entry, dict):
            continue
        raw_docs = entry.get("docs_dir")
        if not isinstance(raw_docs, str):
            continue
        if str(_normalize(Path(raw_docs))) == target:
            return _entry_to_project(entry)
    return None


def ensure_project_dirs(project: Project) -> None:
    project.root_dir.mkdir(parents=True, exist_ok=True)
    project.data_dir.mkdir(parents=True, exist_ok=True)


def list_projects() -> list[Project]:
    """Return all registered projects."""
    reg = _load_registry()
    out: list[Project] = []
    for entry in reg.get("projects", []):
        if not isinstance(entry, dict):
            continue
        if not isinstance(entry.get("id"), str) or not isinstance(
            entry.get("docs_dir"), str
        ):
            continue
        out.append(_entry_to_project(entry))
    return out


def add_project(docs_dir: Path) -> Project:
    """Register a docs directory as a project."""
    docs_dir = _normalize(docs_dir)
    if not docs_dir.exists() or not docs_dir.is_dir():
        raise ValueError(f"docs directory not found: {docs_dir}")

    reg = _load_registry()
    existing = _find_by_docs_dir(reg, docs_dir)
    if existing is not None:
        ensure_project_dirs(existing)
        return existing

    project_id = docs_dir.name.strip() or "project"
    _validate_project_id(project_id, _existing_ids(reg))

    entry = {"id": project_id, "docs_dir": str(docs_dir)}
    reg["projects"].append(entry)
    _save_registry(reg)
    project = _entry_to_project(entry)
    ensure_project_dirs(project)
    return project


def get_project(ref: str) -> Project | None:
    """Resolve project by docs path or id."""
    reg = _load_registry()
    ref = ref.strip()
    if not ref:
        return None

    maybe_path = Path(ref).expanduser()
    if maybe_path.exists() and maybe_path.is_dir():
        return _find_by_docs_dir(reg, _normalize(maybe_path))

    for entry in reg.get("projects", []):
        if not isinstance(entry, dict):
            continue
        if entry.get("id") == ref:
            return _entry_to_project(entry)
    return None


def remove_project(ref: str, *, delete_data: bool = False) -> Project:
    """Unregister a project, preserving local data unless explicitly requested."""
    project = get_project(ref)
    if project is None:
        raise ValueError(f"project not found: {ref}")

    # Destructive removal must succeed before the only registry reference is
    # dropped. A filesystem error then leaves the project fully recoverable.
    if delete_data and project.root_dir.exists():
        shutil.rmtree(project.root_dir)

    reg = _load_registry()
    kept: list[dict] = []
    for entry in reg.get("projects", []):
        if isinstance(entry, dict) and entry.get("id") == project.project_id:
            continue
        kept.append(entry)
    reg["projects"] = kept
    _save_registry(reg)
    return project


def activate(project: Project, *, create_dirs: bool = False) -> None:
    """Activate project paths in runtime config."""
    if create_dirs:
        ensure_project_dirs(project)
    config.activate_project(
        project_id=project.project_id,
        docs_dir=project.docs_dir,
        data_dir=project.data_dir,
        db_path=project.db_path,
    )
