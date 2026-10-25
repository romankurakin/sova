"""Benchmark subprocess with its own database. It returns the production CLI output."""

import json
import resource
import sys
import time
from functools import wraps
from pathlib import Path

from sova import config, search


def main():
    request = json.load(sys.stdin)
    config.clear_active_project()
    config.DB_PATH = Path(request["database"])
    config.DATA_DIR = Path(request["cache_directory"])
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for key, value in request["ranking"].items():
        if not key.startswith("SEARCH_") or not hasattr(search, key):
            raise ValueError(f"Unsupported ranking setting {key}")
        setattr(search, key, value)
    from benchmarks.telemetry import model_rss
    from sova import cli
    from sova.ui import close_output, configure_output, emit

    stages, counts, originals = {}, {}, []

    def instrument(module, name):
        original = getattr(module, name)
        originals.append((module, name, original))

        @wraps(original)
        def measured(*args, **kwargs):
            started = time.perf_counter()
            try:
                result = original(*args, **kwargs)
                if name in ("get_vector_candidates", "search_fts"):
                    counts[name] = len(result)
                    counts[name + "_requested"] = kwargs.get("candidates", args[2])
                if name == "fuse_and_rank":
                    counts.update(final=len(result[0]), vector=result[1], fts=result[2])
                return result
            finally:
                stages[name] = stages.get(name, 0) + time.perf_counter() - started

        setattr(module, name, measured)

    for module, name in (
        (cli, "connect_readonly"),
        (cli, "get_query_embedding"),
        (cli, "get_vector_candidates"),
        (cli, "fuse_and_rank"),
        (search, "search_fts"),
    ):
        instrument(module, name)
    rss_before = model_rss()
    configure_output("json")
    try:
        cli.search_semantic(
            request["query"], request["limit"], use_cache=request["cache"]
        )
        memory = resource.getrusage(resource.RUSAGE_SELF)
        emit(
            "benchmark_diagnostics",
            "Benchmark measurements",
            data={
                "stage_seconds": stages,
                "candidate_counts": counts,
                "cli_peak_rss_bytes": memory.ru_maxrss
                * (1 if sys.platform == "darwin" else 1024),
                "cli_swaps": memory.ru_nswap,
                "model_rss_before": rss_before,
                "model_rss_after": model_rss(),
                "model_rss_measurement": "samples, not peak; all local llama-server processes",
                "os_page_cache_cleared": False,
            },
        )
    finally:
        for module, name, original in originals:
            setattr(module, name, original)
        close_output()


if __name__ == "__main__":
    main()
