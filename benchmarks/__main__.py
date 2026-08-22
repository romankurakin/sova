"""Benchmark CLI with Rich UI matching sova style."""

import hashlib
import re
import sys
import time
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TimeElapsedColumn
from rich.text import Text

from sova import config
from sova import projects as sova_projects
from sova.ui import (
    close_output,
    configure_output,
    emit,
    fmt_duration,
    make_table,
    render_table,
    report_error,
)

_BENCH_DIR = Path(__file__).parent
_SUITE_FILENAME = "suite.json"
_DRAFT_SUITE_FILENAME = "draft-suite.json"
DATA_DIR = config.DATA_DIR

# Long-running benchmark loops use a transient progress bar; durable output
# still goes through the same event stream as the main CLI.
console = Console(file=sys.stderr, stderr=True, highlight=False)


def _note(label: str, value: object, *, level: str = "info") -> None:
    emit(
        "benchmark_info",
        f"{label}: {value}",
        level=level,
        data={"label": label, "value": value},
    )


def _mode(name: str, detail: str | None = None) -> None:
    message = f"Benchmark {name}"
    if detail:
        message += f" · {detail}"
    emit("run_started", message, phase=name, data={"name": name, "detail": detail})


def get_data_dir() -> Path:
    override = DATA_DIR
    if isinstance(override, Path) and override != config.DATA_DIR:
        return override
    return config.get_data_dir()


def _metric_at(values: dict, k: int) -> float:
    """Get metric value by numeric/string key with 0 fallback."""
    raw = values.get(k, values.get(str(k), 0.0))
    try:
        return float(raw)
    except TypeError, ValueError:
        return 0.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _format_error_chain(exc: BaseException) -> str:
    """Render exception + cause chain in one compact line."""
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = str(current).strip() or current.__class__.__name__
        if text not in parts:
            parts.append(text)
        if current.__cause__ is not None:
            current = current.__cause__
            continue
        if current.__suppress_context__:
            break
        current = current.__context__
    return " | ".join(parts)


def _is_likely_oom(message: str) -> bool:
    low = message.lower()
    markers = (
        "out of memory",
        "outofmemory",
        "kiogpucommandbuffercallbackerroroutofmemory",
        "oom",
        "failed to allocate",
        "insufficient memory",
    )
    return any(marker in low for marker in markers)


def _classify_error(message: str) -> tuple[str, str | None, str | None]:
    low = message.lower()
    if "no database. run sova indexing first." in low:
        return (
            "database not ready",
            "no index database found",
            "run sova indexing first",
        )
    if "ground truth is missing" in low:
        return ("ground truth is missing", message, "run judge first")
    if "ground truth contains unjudged chunks" in low:
        return (
            "ground truth has unjudged chunks",
            message,
            "create a reviewed suite with complete pooled judgments",
        )
    if "memory hard-cap exceeded" in low:
        return (
            "model does not fit current memory budget",
            message,
            "close extra apps and retry",
        )
    if "server not reachable at" in low:
        return (
            "model server unavailable",
            message,
            "ensure services are installed/loaded (sova-install), then retry",
        )
    if "server timeout" in low:
        return (
            "model server timed out",
            message,
            "retry; if this repeats, lower concurrent system load",
        )
    if _is_likely_oom(message):
        return (
            "model ran out of memory",
            message,
            "close extra apps and retry",
        )
    return ("benchmark command failed", message, None)


def _report_relevant_service_diags(exc: BaseException) -> None:
    from sova import config
    from sova.llama_client import get_service_diagnostics

    text = _format_error_chain(exc).lower()
    urls: list[str] = []
    if "8081" in text or "embedding" in text:
        urls.append(config.EMBEDDING_SERVER_URL)
    if "8083" in text or "context" in text or "chat" in text or "judge" in text:
        urls.append(config.CONTEXT_SERVER_URL)
    if not urls:
        return

    def _svc_name(url: str) -> str:
        if url == config.EMBEDDING_SERVER_URL:
            return "embedding"
        if url == config.CONTEXT_SERVER_URL:
            return "chat"
        return "service"

    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        diag = get_service_diagnostics(url)
        if diag:
            _note("service", f"{_svc_name(url)} {diag}")


def _report_exception(exc: BaseException) -> None:
    text = re.sub(r"\s+", " ", _format_error_chain(exc)).strip()
    summary, cause, action = _classify_error(text)
    report_error(summary, cause=cause, action=action)
    _report_relevant_service_diags(exc)


def _load_ground_truth(path: Path) -> dict | None:
    """Load ground truth JSON, returning None if missing."""
    import json

    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text())
    except OSError, UnicodeError, json.JSONDecodeError:
        return None
    if not isinstance(loaded, dict) or not isinstance(loaded.get("queries"), list):
        return None
    return loaded


def _save_ground_truth(path: Path, gt: dict):
    """Atomically save ground truth JSON."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gt, indent=2))


def _build_ground_truth(
    queries_list: list[dict],
) -> dict:
    """Build compact ground truth envelope."""
    return {"queries": queries_list}


def cmd_judge():
    """Generate ground truth judgments with multi-source pooling."""
    import json

    from .judge import (
        JUDGE_MODEL,
        QUERY_SET,
        collect_query_subtopics,
        judge_query,
        should_use_debiasing,
    )
    from .search_interface import close_backend

    if not config.get_db_path().exists():
        report_error(
            "database not ready",
            cause="no index database found",
            action="run sova indexing first",
        )
        sys.exit(1)

    _mode("judge")

    checkpoint_path = get_data_dir() / "ground_truth_partial.json"
    output_path = _BENCH_DIR / _DRAFT_SUITE_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)

    k_per_strategy = 20
    use_debiasing = should_use_debiasing()

    # Load existing ground truth (supports incremental judging).
    existing_gt = _load_ground_truth(output_path)
    existing_queries: dict[str, dict] = {}
    if existing_gt:
        for q in existing_gt.get("queries", []):
            existing_queries[q["id"]] = q

    # Also load partial checkpoint (interrupted previous run).
    partial_gt = _load_ground_truth(checkpoint_path)
    if partial_gt:
        for q in partial_gt.get("queries", []):
            if q["id"] not in existing_queries:
                existing_queries[q["id"]] = q

    queries_to_process = list(QUERY_SET)
    spec_by_id = {spec.id: spec for spec in queries_to_process}

    # Reuse previous labels only when id/query/category still match.
    # This avoids stale judgments after query-set rewrites.
    stale_judgment_count = 0
    compatible_existing: dict[str, dict] = {}
    for query_id, prior in existing_queries.items():
        spec = spec_by_id.get(query_id)
        if spec is None:
            continue
        if prior.get("query") == spec.query and prior.get("category") == spec.category:
            compatible_existing[query_id] = prior
        else:
            stale_judgment_count += len(prior.get("judgments", []))
    existing_queries = compatible_existing

    _note("model", JUDGE_MODEL)
    _note("debias", "enabled" if use_debiasing else "disabled")
    _note("pooling", f"hybrid + fts + vector @ k={k_per_strategy}")
    if stale_judgment_count:
        _note(
            "stale", f"ignored {stale_judgment_count} stale judgments (query changed)"
        )
    if existing_queries:
        total_existing = sum(
            len(q.get("judgments", [])) for q in existing_queries.values()
        )
        _note(
            "existing",
            f"{total_existing} judgments across {len(existing_queries)} queries",
        )
    start = time.time()

    # For each query, build existing_judgments map for incremental judging.
    completed: dict[str, dict] = dict(existing_queries)
    new_judgments_total = 0
    current_query_label = "-"

    total = len(queries_to_process)
    done = 0

    def save_checkpoint():
        queries_list = [completed[s.id] for s in QUERY_SET if s.id in completed]
        gt = _build_ground_truth(queries_list)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps(gt, indent=2))

    progress = Progress(BarColumn(bar_width=30), TimeElapsedColumn())
    task = progress.add_task("", total=total, completed=done)

    def _display():
        return Group(
            Text(
                f"queries: {done}/{total}  new judgments: {new_judgments_total}  current: {current_query_label}"
            ),
            progress,
        )

    from .judge import JudgeError
    from .judge import Judgment as _J

    def _make_judgment_callback(spec, existing_chunk_ids, current_query_judgments):
        def _on_chunk_judged(j: _J):
            nonlocal new_judgments_total
            if j.chunk_id not in existing_chunk_ids:
                current_query_judgments.append(
                    {
                        "chunk_id": j.chunk_id,
                        "doc": j.doc,
                        "score": j.score,
                        "confidence": j.confidence,
                        "subtopics": j.subtopics,
                        "reason": j.reason,
                    }
                )
                existing_chunk_ids.add(j.chunk_id)
            new_judgments_total += 1

            # Update completed with partial progress and checkpoint.
            all_j_objs = [
                _J(
                    chunk_id=jd["chunk_id"],
                    doc=jd["doc"],
                    score=jd["score"],
                    reason=jd["reason"],
                    subtopics=jd.get("subtopics", []),
                )
                for jd in current_query_judgments
            ]
            extracted_subtopics = collect_query_subtopics(all_j_objs)
            existing_subtopics = completed.get(spec.id, {}).get("subtopics", [])
            all_subtopics = sorted(
                set(spec.subtopics + existing_subtopics + extracted_subtopics)
            )
            completed[spec.id] = {
                "id": spec.id,
                "query": spec.query,
                "category": spec.category,
                "subtopics": all_subtopics,
                "judgments": current_query_judgments,
            }
            save_checkpoint()
            live.update(_display())

        return _on_chunk_judged

    rate_limited = False
    try:
        with Live(_display(), console=console, transient=True) as live:
            for spec in queries_to_process:
                current_query_label = f"{spec.id} {spec.query[:56]}"
                live.update(_display())
                # Build map of already-judged chunk_ids for this query.
                existing_for_query: dict[int, int] = {}
                if spec.id in completed:
                    for j in completed[spec.id].get("judgments", []):
                        existing_for_query[j["chunk_id"]] = j["score"]

                # Per-chunk checkpoint: merge each judgment immediately.
                existing_judgment_list = completed.get(spec.id, {}).get("judgments", [])
                existing_chunk_ids = {j["chunk_id"] for j in existing_judgment_list}
                current_query_judgments = list(existing_judgment_list)
                on_chunk_judged = _make_judgment_callback(
                    spec, existing_chunk_ids, current_query_judgments
                )

                try:
                    judge_query(
                        spec,
                        verbose=False,
                        use_debiasing=use_debiasing,
                        existing_judgments=existing_for_query,
                        k_per_strategy=k_per_strategy,
                        on_chunk_judged=on_chunk_judged,
                    )
                except JudgeError as e:
                    save_checkpoint()
                    rate_limited = True
                    report_error(
                        "judge stopped",
                        cause=f"{spec.id} ({spec.query}): {e}",
                        action=f"rerun judge to continue from checkpoint ({done}/{total} queries saved)",
                    )
                    break

                done += 1
                progress.update(task, completed=done)
                live.update(_display())
    finally:
        close_backend()

    if rate_limited:
        sys.exit(1)

    if new_judgments_total > 0:
        _note(
            "judged",
            f"{new_judgments_total} new chunks in {fmt_duration(time.time() - start)}",
        )
    else:
        _note(
            "status",
            f"no new chunks to judge ({fmt_duration(time.time() - start).strip()})",
        )

    # Build final output in query order.
    queries_list = [completed[spec.id] for spec in QUERY_SET if spec.id in completed]
    ground_truth = _build_ground_truth(queries_list)

    total_judgments = 0
    score_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for q in queries_list:
        for j in q["judgments"]:
            total_judgments += 1
            score_counts[j["score"]] = score_counts.get(j["score"], 0) + 1

    _save_ground_truth(output_path, ground_truth)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    _note("total", f"{total_judgments} judgments")
    _note("saved", f"{output_path.name}")
    table = make_table(title="Score Distribution")
    table.add_column("Score")
    table.add_column("Count", justify="right")
    table.add_column("", justify="right")

    labels = ["Not Relevant", "Marginal", "Relevant", "Highly Relevant"]
    for score in range(4):
        count = score_counts[score]
        pct = count / total_judgments * 100 if total_judgments else 0
        bar = "\u2588" * int(pct / 5)
        table.add_row(f"{score} {labels[score]}", str(count), f"{pct:.0f}% {bar}")

    render_table(table, gap_before=True)


def cmd_run(
    name: str | None = None,
    *,
    runs: int = 1,
    description: str,
    baseline_name: str | None = None,
):
    """Run deterministic quality evaluation plus corpus-wide latency probes."""
    import json
    import statistics

    from .evaluate import (
        STANDARD_K,
        QueryResult,
        aggregate_by_category,
        aggregate_metrics,
        compute_metrics,
    )
    from .run_benchmark import run_search
    from .search_interface import (
        clear_cache,
        close_backend,
        measure_latency,
    )

    if not name:
        report_error(
            "name is required",
            action="usage: run <name> (e.g. phase1-baseline)",
        )
        sys.exit(1)
    description = description.strip()
    if not description:
        report_error(
            "experiment description is required",
            action="describe what changed and why this run exists",
        )
        sys.exit(2)

    results_dir = _BENCH_DIR / "results"
    json_path = results_dir / f"{name}.json"
    if json_path.exists():
        raise RuntimeError(
            f"benchmark result already exists: {json_path.name}; choose a new name"
        )

    _mode("run", name)

    gt_path = _BENCH_DIR / _SUITE_FILENAME
    if not gt_path.exists():
        report_error(
            "benchmark suite is missing",
            action=f"create {_SUITE_FILENAME} before running the benchmark",
        )
        sys.exit(1)

    try:
        ground_truth = json.loads(gt_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        report_error(
            "benchmark suite is invalid",
            cause=f"{gt_path.name}: {e}",
            action=f"regenerate {_SUITE_FILENAME}",
        )
        sys.exit(1)
    if not isinstance(ground_truth, dict) or not isinstance(
        ground_truth.get("queries"), list
    ):
        report_error(
            "benchmark suite schema is invalid",
            cause=gt_path.name,
            action=f"regenerate {_SUITE_FILENAME}",
        )
        sys.exit(1)

    suite = ground_truth.get("suite")
    if not isinstance(suite, dict) or suite.get("id") != "search":
        report_error(
            "benchmark suite identity is invalid",
            cause=f"{_SUITE_FILENAME} must declare suite.id=search",
        )
        sys.exit(1)

    # Capture corpus identity before loading the vector extension/searching.
    # Some native backends may update on-disk optimization metadata even when
    # the SQL connection itself is read-only.
    db_path = config.get_db_path()
    initial_database_sha256 = _sha256_file(db_path) if db_path.exists() else None
    expected_database_sha256 = suite.get("corpus", {}).get("database_sha256")
    if (
        expected_database_sha256 is not None
        and expected_database_sha256 != initial_database_sha256
    ):
        report_error(
            "benchmark corpus does not match the frozen suite",
            cause=f"expected {expected_database_sha256}, got {initial_database_sha256}",
            action="run against the database snapshot recorded by the suite",
        )
        sys.exit(1)

    k = 10
    k_values = STANDARD_K
    run_count = max(1, int(runs))

    _note("queries", str(len(ground_truth["queries"])))
    _note("runs", f"{run_count} (mean)")
    _note("unjudged-policy", "error")

    def _p95(arr: list[float]) -> float:
        s = sorted(arr)
        return s[int(len(s) * 0.95)] if len(s) >= 20 else s[-1]

    metric_names = [
        "ndcg",
        "mrr",
        "precision",
        "map",
        "recall",
        "hit_rate",
    ]
    category_metric_names = [
        "ndcg",
        "mrr",
        "map",
        "precision",
        "recall",
    ]

    def _average_metrics(samples: list[dict]) -> dict[str, dict[int, float]]:
        if not samples:
            return {name: {} for name in metric_names}
        out: dict[str, dict[int, float]] = {name: {} for name in metric_names}
        n = len(samples)
        for metric_name in metric_names:
            for kv in k_values:
                out[metric_name][kv] = (
                    sum(
                        _metric_at(sample.get(metric_name, {}), kv)
                        for sample in samples
                    )
                    / n
                )
        return out

    def _average_by_category(samples: list[dict]) -> dict[str, dict[str, float]]:
        if not samples:
            return {}
        totals: dict[str, dict[str, float]] = {}
        counts: dict[str, int] = {}
        for sample in samples:
            for cat, metrics in sample.items():
                counts[cat] = counts.get(cat, 0) + 1
                bucket = totals.setdefault(cat, {})
                for metric in category_metric_names:
                    bucket[metric] = bucket.get(metric, 0.0) + float(
                        metrics.get(metric, 0.0)
                    )
        averaged: dict[str, dict[str, float]] = {}
        for cat, metric_totals in totals.items():
            n = counts[cat]
            averaged[cat] = {
                metric: metric_totals.get(metric, 0.0) / n
                for metric in category_metric_names
            }
            count_values = [sample.get(cat, {}).get("count", 0) for sample in samples]
            if any(count_values):
                averaged[cat]["count"] = round(sum(count_values) / len(count_values))
        return averaged

    run_outputs: list[dict] = []
    all_start = time.time()

    for run_idx in range(1, run_count + 1):
        _note("pass", f"{run_idx}/{run_count}")
        run_start = time.time()
        try:
            clear_cache()
            latency_queries = [q["query"] for q in ground_truth["queries"]]

            _note("phase", f"latency probe ({run_idx}/{run_count})")
            # Two untimed requests warm model kernels and allocator state. The
            # reported distribution then covers every suite query, rather than
            # treating a five-query maximum as P95.
            measure_latency(latency_queries[:2])
            latency_data = measure_latency(latency_queries)
            latency_times = latency_data["total_times"]
            latency_p50 = statistics.median(latency_times)
            latency_p95 = _p95(latency_times)
            result_chars_p50 = statistics.median(latency_data.get("result_chars", [0]))

            results = []
            per_query: list[dict] = []

            _note("phase", f"evaluation ({run_idx}/{run_count})")
            with Progress(
                BarColumn(bar_width=30),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("", total=len(ground_truth["queries"]))

                for q in ground_truth["queries"]:
                    hits = run_search(q["query"], limit=max(k_values))

                    judgments = {j["chunk_id"]: j["score"] for j in q["judgments"]}
                    missing_chunk_ids = [
                        h["chunk_id"] for h in hits if h["chunk_id"] not in judgments
                    ]
                    if missing_chunk_ids:
                        preview = ", ".join(str(cid) for cid in missing_chunk_ids[:5])
                        raise RuntimeError(
                            "ground truth contains unjudged chunks "
                            f"for {q['id']} ({q['query']}): "
                            f"{len(missing_chunk_ids)} missing (sample: {preview})"
                        )

                    result_ids = [h["chunk_id"] for h in hits]
                    metrics = compute_metrics(result_ids, judgments, k_values=k_values)

                    results.append(
                        QueryResult(
                            query_id=q["id"],
                            query=q["query"],
                            category=q["category"],
                            metrics=metrics,
                        )
                    )
                    per_query.append(
                        {
                            "id": q["id"],
                            "category": q["category"],
                            "language": q.get("language", "unknown"),
                            "metrics_at_10": {
                                "ndcg": metrics.ndcg.get(10, 0.0),
                                "mrr": metrics.mrr.get(10, 0.0),
                                "map": metrics.map.get(10, 0.0),
                                "precision": metrics.precision.get(10, 0.0),
                                "recall": metrics.recall.get(10, 0.0),
                            },
                        }
                    )
                    progress.update(task, advance=1)

        finally:
            close_backend()

        # Separate negative queries from main metrics.
        positive_results = [r for r in results if r.category != "negative"]
        agg = aggregate_metrics(positive_results) if positive_results else {}

        by_cat = aggregate_by_category(results, k=k)
        run_duration = time.time() - run_start
        _note(
            "pass-summary",
            f"{run_idx}/{run_count} nDCG@10 {_metric_at(agg.get('ndcg', {}), 10):.3f} | "
            f"MRR@10 {_metric_at(agg.get('mrr', {}), 10):.3f} | P50 {latency_p50:.0f}ms",
        )

        run_outputs.append(
            {
                "duration_s": run_duration,
                "latency_ms": {"p50": latency_p50, "p95": latency_p95},
                "result_context_chars_p50": result_chars_p50,
                "metrics": agg,
                "by_category": by_cat,
                "per_query": per_query,
            }
        )

    latency_p50 = statistics.mean(r["latency_ms"]["p50"] for r in run_outputs)
    latency_p95 = statistics.mean(r["latency_ms"]["p95"] for r in run_outputs)
    result_context_chars_p50 = statistics.mean(
        r["result_context_chars_p50"] for r in run_outputs
    )
    p50_values = [r["latency_ms"]["p50"] for r in run_outputs]
    p95_values = [r["latency_ms"]["p95"] for r in run_outputs]

    agg = _average_metrics([r["metrics"] for r in run_outputs])
    by_cat = _average_by_category([r["by_category"] for r in run_outputs])
    per_query_output = run_outputs[0]["per_query"]
    by_language: dict[str, dict[str, float | int]] = {}
    for language in sorted({item["language"] for item in per_query_output}):
        items = [
            item
            for item in per_query_output
            if item["language"] == language and item["category"] != "negative"
        ]
        if not items:
            continue
        by_language[language] = {
            metric: sum(item["metrics_at_10"][metric] for item in items) / len(items)
            for metric in ("ndcg", "mrr", "map", "precision", "recall")
        }
        by_language[language]["count"] = len(items)
    _note("evaluated", f"in {fmt_duration(time.time() - all_start).strip()}")
    _note(
        "latency-spread",
        f"P50 {min(p50_values):.0f}-{max(p50_values):.0f}ms | "
        f"P95 {min(p95_values):.0f}-{max(p95_values):.0f}ms",
    )
    _note(
        "summary",
        " | ".join(
            [
                f"nDCG@10 {_metric_at(agg.get('ndcg', {}), 10):.3f}",
                f"MRR@10 {_metric_at(agg.get('mrr', {}), 10):.3f}",
                f"P50 {latency_p50:.0f}ms",
            ]
        ),
    )

    blank = "\u2014"
    table = make_table(title=f"Results ({run_count}-run mean)")
    table.add_column("Metric")
    for kv in k_values:
        table.add_column(f"@{kv}", justify="right")

    # Latency — single values in @10 column.
    table.add_row("Latency P50", *[blank] * (len(k_values) - 1), f"{latency_p50:.0f}ms")
    table.add_row("Latency P95", *[blank] * (len(k_values) - 1), f"{latency_p95:.0f}ms")
    table.add_row(
        "Context chars",
        *[blank] * (len(k_values) - 1),
        f"{result_context_chars_p50:.0f}",
    )
    table.add_section()

    # IR metrics at all k cutoffs.
    for metric, label in [
        ("ndcg", "nDCG"),
        ("mrr", "MRR"),
        ("precision", "Precision"),
        ("map", "MAP"),
        ("recall", "Recall"),
        ("hit_rate", "Hit Rate"),
    ]:
        row = [label] + [
            f"{_metric_at(agg.get(metric, {}), kv):.3f}" for kv in k_values
        ]
        table.add_row(*row)

    render_table(table)

    if by_cat:
        cat_table = make_table(title="By Category")
        cat_table.add_column("Category")
        cat_table.add_column("nDCG", justify="right")
        cat_table.add_column("MRR", justify="right")
        cat_table.add_column("Precision", justify="right")
        cat_table.add_column("Recall", justify="right")
        for cat, metrics in sorted(by_cat.items()):
            cat_table.add_row(
                cat,
                f"{metrics.get('ndcg', 0):.3f}",
                f"{metrics.get('mrr', 0):.3f}",
                f"{metrics.get('precision', 0):.3f}",
                f"{metrics.get('recall', 0):.3f}",
            )
        render_table(cat_table, gap_before=True)

    output = {
        "benchmark_schema_version": 2,
        "name": name,
        "experiment": description,
        "benchmark_suite": {
            "id": suite["id"],
            "schema_version": suite.get("schema_version"),
            "sha256": _sha256_file(gt_path),
        },
        "k": k,
        "runs": run_count,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "unjudged_policy": "error",
        "latency_ms": {"p50": round(latency_p50, 1), "p95": round(latency_p95, 1)},
        "context_payload": {
            "median_chars_per_top10": round(result_context_chars_p50, 1),
        },
        "metrics": agg,
        "by_category": by_cat,
        "by_language": by_language,
        "per_query": per_query_output,
    }

    results_dir.mkdir(parents=True, exist_ok=True)

    if baseline_name:
        import random

        baseline_path = results_dir / f"{baseline_name}.json"
        if not baseline_path.exists():
            raise RuntimeError(f"comparison baseline not found: {baseline_path.name}")
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        comparable_fields = [
            (
                "suite",
                baseline.get("benchmark_suite", {}).get("sha256"),
                output["benchmark_suite"]["sha256"],
            ),
            ("k", baseline.get("k"), k),
        ]
        mismatches = [label for label, old, new in comparable_fields if old != new]
        if mismatches:
            raise RuntimeError(
                "baseline is not comparable: " + ", ".join(mismatches) + " differ"
            )

        baseline_queries = {
            item["id"]: item
            for item in baseline.get("per_query", [])
            if item.get("category") != "negative"
        }
        candidate_queries = {
            item["id"]: item
            for item in output["per_query"]
            if item.get("category") != "negative"
        }
        if baseline_queries.keys() != candidate_queries.keys():
            raise RuntimeError("baseline query IDs do not match candidate query IDs")

        metrics = ("ndcg", "mrr", "map", "precision", "recall")
        paired_deltas: dict[str, list[float]] = {
            metric: [
                candidate_queries[query_id]["metrics_at_10"][metric]
                - baseline_queries[query_id]["metrics_at_10"][metric]
                for query_id in sorted(candidate_queries)
            ]
            for metric in metrics
        }
        mean_deltas = {
            metric: sum(values) / len(values)
            for metric, values in paired_deltas.items()
        }
        rng = random.Random(20260820)
        ndcg_deltas = paired_deltas["ndcg"]
        boot = sorted(
            sum(rng.choice(ndcg_deltas) for _ in ndcg_deltas) / len(ndcg_deltas)
            for _ in range(10_000)
        )
        output["comparison_to"] = {
            "name": baseline_name,
            "paired_positive_queries": len(ndcg_deltas),
            "mean_delta_at_10": mean_deltas,
            "ndcg_delta_95pct_bootstrap_ci": [boot[249], boot[9749]],
            "latency_delta_ms": {
                "p50": output["latency_ms"]["p50"] - baseline["latency_ms"]["p50"],
                "p95": output["latency_ms"]["p95"] - baseline["latency_ms"]["p95"],
            },
        }

    json_path.write_text(json.dumps(output, indent=2))

    _note("saved", f"{json_path.name}")


def cmd_show(run_name: str | None = None):
    """Show benchmark results."""
    import json

    results_dir = _BENCH_DIR / "results"
    if run_name == "list" or run_name is None:
        _mode("show", "list")
        skipped = 0
        runs = (
            sorted(
                results_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if results_dir.exists()
            else []
        )
        if not runs:
            _note("status", "no benchmark runs found")
            _note("hint", "run run <name> first")
            return

        table = make_table(title="Benchmark Runs")
        table.add_column("Name")
        table.add_column("Date", style="dim")
        table.add_column("nDCG", justify="right")
        table.add_column("Latency", justify="right", style="dim")

        for run_path in runs[:10]:
            try:
                data = json.loads(run_path.read_text())
            except OSError, UnicodeError, json.JSONDecodeError:
                skipped += 1
                continue
            k = data["k"]
            m = data.get("metrics", {})
            ndcg = m.get("ndcg", {})
            ndcg_val = ndcg.get(str(k), ndcg.get(k, 0))
            lat = data.get("latency_ms", {}).get("p50", 0)
            table.add_row(
                data.get("name", run_path.stem),
                data.get("created", "")[:10],
                f"{ndcg_val:.3f}",
                f"{lat:.0f}ms",
            )

        if table.row_count == 0:
            report_error(
                "benchmark results are unreadable",
                cause="all run files failed JSON parsing",
                action="rerun benchmark or remove invalid files in benchmarks/results",
            )
            sys.exit(1)

        render_table(table)
        if skipped:
            _note(
                "warning",
                f"skipped {skipped} invalid run file(s)",
                level="warning",
            )
        _note("hint", "use show <name> to view details")
        return
    results_path = results_dir / f"{run_name}.json"
    if not results_path.exists():
        report_error(
            "benchmark run not found",
            cause=f"{run_name}",
            action="use show to list available runs",
        )
        sys.exit(1)

    try:
        data = json.loads(results_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        report_error(
            "benchmark run is invalid",
            cause=f"{results_path.name}: {e}",
            action="rerun benchmark for this run name",
        )
        sys.exit(1)
    k = data["k"]
    m = data.get("metrics", {})

    run_label = data.get("name", run_name)
    created = data.get("created", "unknown")
    _mode("show", f"{run_label}")
    _note("date", created)
    from .evaluate import STANDARD_K

    def get_val(d, k):
        return d.get(str(k), d.get(k, 0))

    blank = "\u2014"
    lat = data.get("latency_ms", {})
    _note(
        "summary",
        " | ".join(
            [
                f"nDCG@10 {_metric_at(m.get('ndcg', {}), 10):.3f}",
                f"MRR@10 {_metric_at(m.get('mrr', {}), 10):.3f}",
                f"P50 {float(lat.get('p50', 0)):.0f}ms",
            ]
        ),
    )
    table = make_table(title="Results")
    table.add_column("Metric")
    for kv in STANDARD_K:
        table.add_column(f"@{kv}", justify="right")

    # Latency.
    if lat:
        table.add_row(
            "Latency P50",
            *[blank] * (len(STANDARD_K) - 1),
            f"{lat.get('p50', 0):.0f}ms",
        )
        p95 = lat.get("p95")
        if p95 is not None:
            table.add_row(
                "Latency P95", *[blank] * (len(STANDARD_K) - 1), f"{p95:.0f}ms"
            )
        table.add_section()

    # IR metrics.
    for metric, label in [
        ("ndcg", "nDCG"),
        ("mrr", "MRR"),
        ("precision", "Precision"),
        ("map", "MAP"),
        ("recall", "Recall"),
        ("hit_rate", "Hit Rate"),
    ]:
        row = [label] + [f"{get_val(m.get(metric, {}), kv):.3f}" for kv in STANDARD_K]
        table.add_row(*row)

    render_table(table, gap_before=True)
    by_cat = data.get("by_category", {})
    if by_cat:
        cat_table = make_table(title="By Category")
        cat_table.add_column("Category")
        cat_table.add_column("nDCG", justify="right")
        cat_table.add_column("MRR", justify="right")
        cat_table.add_column("Precision", justify="right")
        cat_table.add_column("Recall", justify="right")
        for cat, metrics in sorted(by_cat.items()):
            cat_table.add_row(
                cat,
                f"{metrics.get('ndcg', 0):.3f}",
                f"{metrics.get('mrr', 0):.3f}",
                f"{metrics.get('precision', 0):.3f}",
                f"{metrics.get('recall', 0):.3f}",
            )
        render_table(cat_table, gap_before=True)


def main():
    import argparse

    config.clear_active_project()
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="Sova benchmark suite",
    )
    sub = parser.add_subparsers(dest="command")

    p_judge = sub.add_parser("judge", help="Generate ground truth judgments")
    p_judge.add_argument("project", help="Project id/path")

    p_run = sub.add_parser("run", help="Run benchmark against ground truth")
    p_run.add_argument("project", help="Project id/path")
    p_run.add_argument("name", help="Benchmark run name (e.g. 'baseline')")
    p_run.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Complete deterministic passes (default: 1)",
    )
    p_run.add_argument(
        "--description",
        required=True,
        help="Human-readable description stored in the result JSON",
    )
    p_run.add_argument(
        "--baseline",
        default=None,
        help="Comparable result name for paired deltas and bootstrap CI",
    )

    p_show = sub.add_parser("show", help="Display benchmark results")
    p_show.add_argument("project", help="Project id/path")
    p_show.add_argument(
        "name", nargs="?", default=None, help="Run name (omit to list all)"
    )

    try:
        args = parser.parse_args()

        if not args.command:
            parser.print_help()
            sys.exit(0)

        project = sova_projects.get_project(args.project)
        if project is None:
            report_error(
                "project not found",
                cause=args.project,
                action="run: sova projects",
            )
            sys.exit(1)
        assert project is not None
        sova_projects.activate(project)
        _note("project", project.project_id)

        if args.command == "judge":
            cmd_judge()
        elif args.command == "run":
            cmd_run(
                name=args.name,
                runs=max(1, int(args.runs)),
                description=args.description,
                baseline_name=args.baseline,
            )
        elif args.command == "show":
            cmd_show(run_name=args.name)
    finally:
        config.clear_active_project()


if __name__ == "__main__":
    configure_output("auto")
    try:
        main()
    except KeyboardInterrupt:
        emit("interrupted", "Benchmark interrupted")
        sys.exit(130)
    except (OSError, RuntimeError, ValueError) as e:
        _report_exception(e)
        sys.exit(1)
    finally:
        close_output()
