"""Run measured searches on the benchmark index.

A run continues after an interruption.
"""

import json
import random
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

from benchmarks import embedders
from benchmarks.corpus import CorpusError
from benchmarks.experiments import load_experiment
from benchmarks.passages import _hash
from benchmarks.queries import load_evaluation_queries
from benchmarks.reading import _digest
from benchmarks.records import SearchRun
from benchmarks.search_interface import code_identity, open_backend
from benchmarks.storage import locked, read, write
from benchmarks.telemetry import environment_identity, model_rss
from sova import config, llama_client, search


def execute_cli(request, *, timeout=600):
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.worker"],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CorpusError("Search CLI timed out") from exc
    elapsed = time.perf_counter() - t0
    if result.returncode:
        raise CorpusError(
            f"Search CLI failed ({result.returncode}): {result.stderr[-1500:]}"
        )
    try:
        events = [
            json.loads(line) for line in result.stdout.splitlines() if line.strip()
        ]
        if any(event.get("level") == "error" for event in events):
            raise ValueError("CLI emitted an error")
        hits = [event["data"] for event in events if event.get("type") == "result"]
        if not hits and not any(
            event.get("type") == "search_completed" for event in events
        ):
            raise ValueError("CLI produced no completion or results")
        if len(hits) > request["limit"]:
            raise ValueError("CLI exceeded requested result limit")
    except (ValueError, KeyError, TypeError) as exc:
        raise CorpusError(f"Invalid CLI search response: {exc}") from exc
    return hits, {"elapsed_seconds": elapsed, "stderr": result.stderr, "events": events}


def _index_bytes(path: Path) -> int | None:
    if not path.parent.is_dir():
        return None
    return sum(file.stat().st_size for file in path.parent.iterdir() if file.is_file())


def _server_rss(samples) -> int | None:
    return sum(row["rss_bytes"] for row in samples) if samples else None


def compact_run(checkpoints, *, name, experiment, identity, queries, modes, backend):
    """Keep the hits, times and memory peaks of a run in one file."""
    native = experiment.native
    result: dict = {}
    cli_rss, server_rss = [], []
    for query in queries:
        records = [
            read(checkpoints / f"{mode}-{repeat}" / f"{query.id}.json")
            for mode in modes
            for repeat in range(experiment.passes)
        ]
        hits = records[0]["run"]["hits"]
        for record in records:
            server_rss.append(record.get("server_rss_bytes"))
            for event in record["timing"].get("events", []):
                if event.get("type") == "benchmark_diagnostics":
                    cli_rss.append(event["data"]["cli_peak_rss_bytes"])
                    server_rss.append(_server_rss(event["data"]["model_rss_after"]))
        result[query.id] = {
            "query_sha256": records[0]["run"]["query_sha256"],
            "hits": hits,
            "characters": sum(len(backend.items[pid]["text"]) for pid in hits),
            "seconds": {
                mode: [
                    record["timing"]["elapsed_seconds"]
                    for record in records
                    if record["mode"] == mode
                ]
                for mode in modes
            },
            # All passes returned the same passages in the same order.
            "stable": all(record["run"]["hits"] == hits for record in records),
        }
    server_rss = [value for value in server_rss if value is not None]
    return {
        "name": name,
        "description": experiment.description,
        "completed": datetime.now(UTC).isoformat(timespec="seconds"),
        "identity": identity,
        "measurement": "production_cli_subprocess" if native else "direct_adapter",
        "queries": result,
        "resources": {
            "cli_peak_rss_bytes": max(cli_rss) if cli_rss else None,
            "server_rss_bytes": max(server_rss) if server_rss else None,
            "index_bytes": _index_bytes(backend.index_path),
        },
    }


def run_benchmark(
    root,
    name="baseline",
    *,
    backend=None,
    executor=execute_cli,
    queries=None,
    notify=print,
):
    root = Path(root)
    experiment = load_experiment(name)
    queries = queries if queries is not None else load_evaluation_queries()
    originals = {}
    with (
        locked(root, "model-work"),
        locked(root, "run"),
        ExitStack() as stack,
    ):
        if backend is None:
            backend = stack.enter_context(open_backend(root, experiment, notify=notify))
        code = code_identity()
        if experiment.embedder is not None:
            code["benchmarks.embedders"] = _digest(
                Path(embedders.__file__).read_bytes()
            )
        identity: dict = {
            "experiment": name,
            "settings": experiment.settings(),
            "passage_snapshot": backend.snapshot["snapshot_sha256"],
            "index": backend.manifest["sha256"],
            "models": backend.manifest["inputs"]["models"],
            "code": code,
            "runner_sha256": _hash(Path(__file__).read_text()),
            "queries_sha256": _hash([q.model_dump() for q in queries]),
            "environment": environment_identity(),
        }
        run_id = _hash(identity)
        ready = root / "runs" / f"{run_id}.json"
        if ready.exists():
            read(ready)
            notify("Verified complete saved run; no new measurements were made")
            return ready
        # Local checkpoints for resume. Git keeps only the compact run file.
        checkpoints = root / "runs" / run_id
        try:
            for key, value in experiment.ranking.items():
                originals[key] = getattr(search, key)
                setattr(search, key, value)
            native = experiment.native
            if experiment.method != "bm25":
                backend.start()
                # The warm-up query is not in the measured records.
                # It does not change the bank.
                backend.retrieve(
                    queries[0].text,
                    limit=experiment.output_k,
                    method=experiment.method,
                    depth=experiment.depth,
                )
            modes = experiment.cache_modes if native else ("warm",)
            for mode in modes:
                for repeat in range(experiment.passes):
                    order = list(queries)
                    random.Random(f"260907:{mode}:{repeat}").shuffle(order)
                    for position, query in enumerate(order, 1):
                        path = checkpoints / f"{mode}-{repeat}" / f"{query.id}.json"
                        if path.exists():
                            previous = read(path)
                            if previous["run"]["status"] == "ok":
                                continue
                            write(
                                checkpoints
                                / "failed-attempts"
                                / f"{previous['sha256']}.json",
                                previous,
                                immutable=True,
                            )
                        if mode == "cold" and native:
                            llama_client.stop_server(
                                config.EMBEDDING_SERVER_URL, suppress_interrupt=True
                            )
                        request = {
                            "database": str(
                                backend.index_path.parent / backend.manifest["database"]
                            ),
                            "cache_directory": str(checkpoints / "query-cache"),
                            "ranking": experiment.ranking,
                            "query": query.text,
                            "limit": experiment.output_k,
                            "cache": mode == "cached",
                        }
                        server_rss = None
                        t0 = time.perf_counter()
                        try:
                            if native:
                                if mode == "cached":
                                    # Fill the cache for this query. Do not time it.
                                    executor(request)
                                results, timing = executor(request)
                                run = backend.from_cli(query, results)
                                timing["measurement"] = "production_cli_subprocess"
                            else:
                                ids = backend.retrieve(
                                    query.text,
                                    method=experiment.method,
                                    limit=experiment.output_k,
                                    depth=experiment.depth,
                                )
                                run = backend.run(query, ids)
                                timing = {
                                    "elapsed_seconds": time.perf_counter() - t0,
                                    "measurement": "direct_adapter",
                                }
                                # Sample after the timed search, not during it.
                                if experiment.method != "bm25":
                                    server_rss = _server_rss(model_rss())
                            if experiment.method != "bm25":
                                backend.models.verify()
                        except (OSError, RuntimeError, ValueError) as exc:
                            run = SearchRun(
                                query_id=query.id,
                                query_sha256=_hash(query.judge_input()),
                                status="error",
                                error=str(exc),
                                hits=(),
                            )
                            timing = {
                                "elapsed_seconds": time.perf_counter() - t0,
                                "measurement": "failed",
                            }
                        write(
                            path,
                            {
                                "run": run.model_dump(mode="json"),
                                "timing": timing,
                                "server_rss_bytes": server_rss,
                                "mode": mode,
                                "repeat": repeat,
                                "order_index": position - 1,
                            },
                        )
                        if run.status == "error":
                            raise CorpusError(
                                f"Search failed for {query.id}; saved output and progress: {run.error}"
                            )
                        notify(
                            f"{name} {mode} pass {repeat + 1}/{experiment.passes}: "
                            f"{position}/{len(order)} {query.id} "
                            f"{timing['elapsed_seconds']:.2f} s"
                        )
            write(
                ready,
                compact_run(
                    checkpoints,
                    name=name,
                    experiment=experiment,
                    identity=identity,
                    queries=queries,
                    modes=modes,
                    backend=backend,
                ),
                immutable=True,
            )
            return ready
        finally:
            for key, value in originals.items():
                setattr(search, key, value)
