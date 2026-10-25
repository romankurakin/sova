"""Benchmark commands: prepare, judge, run and report."""

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from benchmarks import passages as passage_module
from benchmarks.build_suite import freeze, prepare_pool
from benchmarks.corpus import CorpusError, prepare_sources
from benchmarks.experiments import load_experiment
from benchmarks.indexing import context_bodies_digest, load_index, prepare_index
from benchmarks.judge import JudgeFailure, run_judge
from benchmarks.passages import load_snapshot, prepare_passages
from benchmarks.queries import load_query_banks
from benchmarks.reading import _digest, prepare_reading
from benchmarks.report import build_report, experiments_markdown
from benchmarks.run_benchmark import run_benchmark
from benchmarks.storage import DEFAULT_DATA, locked, read, write
from sova import extract

app = typer.Typer(
    no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False
)


@app.callback()
def _cli() -> None:
    """Prepare the benchmark data and measure the quality of Sova search."""


def prepare_workflow():
    root = DEFAULT_DATA
    with locked(root, "model-work"):
        return _prepare_workflow(root)


def _prepare_workflow(root):
    core, probes = load_query_banks()
    by_language = Counter(q.language for q in core.queries + probes.queries)
    typer.echo(
        f"Queries: {by_language.total()} ("
        + ", ".join(f"{count} {lang}" for lang, count in sorted(by_language.items()))
        + ")"
    )
    prepare_sources(root, notify=typer.echo)
    reading = prepare_reading(root, notify=typer.echo)
    state_path = root / "preparation.json"
    previous = read(state_path) if state_path.exists() else None
    reuse_passages = False
    if previous:
        passages = root / previous["passages"]
        snapshot = load_snapshot(passages)
        pipeline = snapshot["inputs"]["pipeline"]
        reuse_passages = (
            snapshot["inputs"]["documents"]
            == passage_module._load_inputs(reading)[0]["documents"]
            and pipeline["code_sha256"] == _digest(Path(extract.__file__).read_bytes())
            and pipeline["adapter_sha256"]
            == _digest(Path(passage_module.__file__).read_bytes())
        )
    if reuse_passages:
        typer.echo(
            "Reusing the pinned passage boundaries independently of the current embedder."
        )
    else:
        passages = prepare_passages(reading, notify=typer.echo)
    state = {
        "passages": str(passages.relative_to(root)),
        "index": previous.get("index")
        if previous and previous["passages"] == str(passages.relative_to(root))
        else None,
    }
    write(state_path, state)
    if state["index"]:
        index = root / state["index"]
        # The index is not in git, so it can be missing. A changed context request
        # also needs a new index.
        if not index.exists() or load_index(index).get(
            "context_bodies_sha256"
        ) != context_bodies_digest(passages):
            state["index"] = None
            write(state_path, state)
    if not state["index"]:
        index = prepare_index(passages, notify=typer.echo)
        state["index"] = str(index.relative_to(root))
        write(state_path, state)
    typer.echo(f"Search index ready: {index}")
    prepare_pool(root, notify=typer.echo)
    suite = freeze(root)
    build_report(root)
    status = read(suite)["status"]
    typer.echo(
        f"Reference: {status}. "
        + (
            "Next: uv run bench run"
            if status == "ready"
            else "Continue: uv run bench judge codex; uv run bench judge claude; uv run bench prepare"
        )
    )
    return suite


def _failure(exc):
    typer.echo(str(exc), err=True)
    raise typer.Exit(
        2 if isinstance(exc, JudgeFailure) and exc.kind == "paused_quota" else 1
    ) from exc


@app.command()
def prepare():
    """Prepare all available data and continue from the last step. Judges do not run."""
    try:
        prepare_workflow()
    except (CorpusError, OSError, subprocess.TimeoutExpired) as exc:
        _failure(exc)


@app.command()
def judge(
    provider: Annotated[str, typer.Argument(help="codex or claude")],
    max_pairs: Annotated[
        int | None,
        typer.Option(min=1, help="Maximum new decisions in this invocation."),
    ] = None,
):
    """Grade the remaining jobs with the subscription of the selected judge."""
    try:
        result = run_judge(
            DEFAULT_DATA,
            provider,
            max_pairs=max_pairs,
            notify=typer.echo,
        )
        typer.echo(f"Judge: {result['status']}; completed {result['completed']} jobs.")
    except (CorpusError, OSError, subprocess.TimeoutExpired) as exc:
        _failure(exc)


@app.command()
def run(
    experiment: Annotated[
        str, typer.Argument(help="Name of a file in benchmarks/experiments/.")
    ] = "baseline",
):
    """Run all queries with one experiment. Save the ranked results and times."""
    try:
        root = DEFAULT_DATA
        load_experiment(experiment)
        state_path = root / "preparation.json"
        state = read(state_path) if state_path.exists() else None
        if not state or not state.get("index"):
            prepare_workflow()
        result = run_benchmark(root, experiment, notify=typer.echo)
        typer.echo(f"Run saved: {result}")
        report = build_report(root)
        typer.echo(
            f"Reference status: {report['status']}. Run bench prepare to pool new passages for judging."
        )
    except (CorpusError, OSError, subprocess.TimeoutExpired) as exc:
        _failure(exc)


@app.command()
def report():
    """Score every saved run and write the experiments table. Models do not start."""
    try:
        result = build_report(DEFAULT_DATA)
        if "experiments" not in result:
            typer.echo(json.dumps(result, indent=2))
            return
        typer.echo(experiments_markdown(result))
        typer.echo(f"Saved: {DEFAULT_DATA / 'EXPERIMENTS.md'}, report.json, results/")
    except (CorpusError, OSError, subprocess.TimeoutExpired) as exc:
        _failure(exc)


def main():
    app()
