"""Write benchmark records atomically with a content hash, and lock processes."""

import copy
import fcntl
import json
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from benchmarks.corpus import CorpusError, _write_atomic
from benchmarks.passages import _hash, _json

DEFAULT_DATA = Path(__file__).with_name("results") / "os-architectures-v1"


@lru_cache(maxsize=512)
def _read_stamped(path: Path, stamp: tuple) -> dict:
    value = json.loads(path.read_bytes())
    if value["sha256"] != _hash({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("checksum mismatch")
    return value


def read(path: Path) -> dict:
    try:
        stat = path.stat()
        stamp = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        return copy.deepcopy(_read_stamped(path, stamp))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CorpusError(f"Invalid benchmark record {path}: {exc}") from exc


def write(path: Path, value: dict, *, immutable: bool = False) -> dict:
    value = {k: v for k, v in value.items() if k != "sha256"}
    value["sha256"] = _hash(value)
    if immutable and path.exists():
        if read(path) != value:
            raise CorpusError(f"Immutable benchmark record differs: {path}")
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, _json(value))
    return value


@contextmanager
def locked(root: Path, name: str):
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{name}.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CorpusError(f"Another benchmark {name} process is running") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
