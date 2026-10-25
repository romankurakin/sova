"""Record the local environment and resource use. A missing value is null."""

import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path

from benchmarks.passages import _hash


def command_output(argv, cwd=None):
    try:
        result = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=10, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except OSError, subprocess.TimeoutExpired:
        return None


def environment_identity(repo=None):
    repo = repo or Path(__file__).parents[1]
    # The benchmark data is in the repository, but it is not code. New judgments or a
    # commit of results must not change the identity of a saved run.
    code = ["--", ".", ":(exclude)benchmarks/results"]
    tree = command_output(["git", "ls-tree", "-r", "HEAD", *code], repo)
    diff = command_output(["git", "diff", "HEAD", "--binary", *code], repo)
    tracked = command_output(
        ["git", "ls-files", "--others", "--exclude-standard", *code], repo
    )
    untracked = {
        p: _hash((repo / p).read_bytes().hex())
        for p in (tracked or "").splitlines()
        if (repo / p).is_file()
    }
    return {
        "python": platform.python_version(),
        "system": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "host_sha256": _hash(platform.node()),
        "hardware": command_output(["sysctl", "-n", "hw.model"])
        if platform.system() == "Darwin"
        else platform.processor(),
        "memory_bytes": command_output(["sysctl", "-n", "hw.memsize"])
        if platform.system() == "Darwin"
        else None,
        "code_sha256": _hash({"tree": tree, "diff": diff, "untracked": untracked}),
        "dependencies": {
            key: importlib.metadata.version(key) for key in ("numpy", "sqliteai-vector")
        },
    }


def model_rss():
    output = command_output(["ps", "-axo", "pid=,rss=,comm="])
    if output is None:
        return None
    rows = []
    for line in output.splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) == 3 and Path(fields[2]).name == "llama-server":
            rows.append({"pid": int(fields[0]), "rss_bytes": int(fields[1]) * 1024})
    return rows
