"""Benchmark CLI with Rich UI matching sova style."""

import re
import sys
import time
from pathlib import Path
from typing import cast

from rich.console import Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TimeElapsedColumn
from rich.text import Text

from sova import config
from sova import projects as sova_projects
from sova.ui import (
    console,
    fmt_duration,
    make_table,
    print_gap,
    report,
    report_error,
    report_mode,
    report_progress,
    render_table,
)

_BENCH_DIR = Path(__file__).parent
DATA_DIR = config.DATA_DIR


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
            "rerun judge first, or run benchmark with --autofill",
        )
    if "memory hard-cap exceeded" in low:
        return (
            "model does not fit current memory budget",
            message,
            "close extra apps and retry",
        )
    if "physical batch size" in low or "too large to process" in low:
        return (
            "reranker request exceeds server batch capacity",
            message,
            "run sova-install to refresh reranker service settings",
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
    if "8082" in text or "rerank" in text:
        urls.append(config.RERANKER_SERVER_URL)
    if "8083" in text or "context" in text or "chat" in text or "judge" in text:
        urls.append(config.CONTEXT_SERVER_URL)
    if not urls:
        return

    def _svc_name(url: str) -> str:
        if url == config.EMBEDDING_SERVER_URL:
            return "embedding"
        if url == config.RERANKER_SERVER_URL:
            return "reranker"
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
            report("service", f"{_svc_name(url)} {diag}")


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
    except Exception:
        return None
    return _normalize_ground_truth(loaded)


def _normalize_ground_truth(raw: object) -> dict | None:
    """Normalize old/new ground truth envelopes to a stable in-memory schema."""
    if not isinstance(raw, dict):
        return None
    raw_dict = cast(dict[str, object], raw)
    queries = raw_dict.get("queries")
    if not isinstance(queries, list):
        return None
    return {"queries": queries}


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
    from .judge import (
        QUERY_SET,
        judge_query,
        JUDGE_MODEL,
        collect_query_subtopics,
        should_use_debiasing,
    )
    from .search_interface import close_backend
    import json

    if not config.get_db_path().exists():
        report_error(
            "database not ready",
            cause="no index database found",
            action="run sova indexing first",
        )
        sys.exit(1)

    report_mode("bench.judge")

    checkpoint_path = get_data_dir() / "ground_truth_partial.json"
    output_path = _BENCH_DIR / "ground_truth.json"
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

    report("model", JUDGE_MODEL)
    report("debias", "enabled" if use_debiasing else "disabled")
    report("pooling", f"hybrid + fts + vector @ k={k_per_strategy}")
    if stale_judgment_count:
        report(
            "stale", f"ignored {stale_judgment_count} stale judgments (query changed)"
        )
    if existing_queries:
        total_existing = sum(
            len(q.get("judgments", [])) for q in existing_queries.values()
        )
        report(
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

    from .judge import JudgeError, Judgment as _J

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

                try:
                    judge_query(
                        spec,
                        verbose=False,
                        use_debiasing=use_debiasing,
                        existing_judgments=existing_for_query,
                        k_per_strategy=k_per_strategy,
                        on_chunk_judged=_on_chunk_judged,
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
        report(
            "judged",
            f"{new_judgments_total} new chunks in {fmt_duration(time.time() - start)}",
        )
    else:
        report(
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

    report("total", f"{total_judgments} judgments")
    report("saved", f"{output_path.name}")
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
    autofill: bool = False,
    runs: int = 3,
    use_reranker: bool = config.SEARCH_USE_RERANKER,
):
    """Run benchmark as a 3-pass averaged baseline."""
    from .run_benchmark import run_search
    from .evaluate import (
        aggregate_metrics,
        aggregate_by_category,
        compute_metrics,
        compute_diversity_metrics,
        STANDARD_K,
        QueryResult,
    )
    from .search_interface import (
        measure_latency,
        clear_cache,
        close_backend,
        get_backend,
    )
    from .judge import judge_single_chunk, should_use_debiasing
    import json
    from pathlib import Path
    import statistics

    if not name:
        report_error(
            "name is required",
            action="usage: run <name> (e.g. phase1-baseline)",
        )
        sys.exit(1)

    report_mode("bench.run", name)

    gt_path = _BENCH_DIR / "ground_truth.json"
    if not gt_path.exists():
        report_error(
            "ground truth is missing",
            action="run judge first",
        )
        sys.exit(1)

    try:
        ground_truth = json.loads(gt_path.read_text())
    except Exception as e:
        report_error(
            "ground truth is invalid",
            cause=f"{gt_path.name}: {e}",
            action="rerun judge to regenerate ground_truth.json",
        )
        sys.exit(1)
    if not isinstance(ground_truth, dict) or not isinstance(
        ground_truth.get("queries"), list
    ):
        report_error(
            "ground truth schema is invalid",
            cause=gt_path.name,
            action="rerun judge to regenerate ground_truth.json",
        )
        sys.exit(1)

    k = 10
    k_values = STANDARD_K
    run_count = max(1, int(runs))

    report("queries", str(len(ground_truth["queries"])))
    report("autofill", "enabled" if autofill else "disabled")
    report("reranker", "enabled" if use_reranker else "disabled")
    report("runs", f"{run_count} (mean)")
    autofill_use_debiasing = should_use_debiasing()

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
        "doc_coverage",
        "subtopic_recall",
        "alpha_ndcg",
    ]
    category_metric_names = [
        "ndcg",
        "mrr",
        "map",
        "precision",
        "recall",
        "subtopic_recall",
        "doc_coverage",
    ]

    def _average_metrics(samples: list[dict]) -> dict[str, dict[int, float]]:
        if not samples:
            return {name: {} for name in metric_names}
        out: dict[str, dict[int, float]] = {name: {} for name in metric_names}
        n = len(samples)
        for name in metric_names:
            for kv in k_values:
                out[name][kv] = (
                    sum(_metric_at(sample.get(name, {}), kv) for sample in samples) / n
                )
        return out

    def _average_neg_fp(samples: list[dict]) -> dict[int, float]:
        if not samples:
            return {}
        n = len(samples)
        return {
            kv: sum(_metric_at(sample, kv) for sample in samples) / n for kv in k_values
        }

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
        report("pass", f"{run_idx}/{run_count}")
        run_start = time.time()
        try:
            clear_cache()
            latency_queries = [
                "ARM exception handling",
                "RISC-V trap handling",
                "memory protection unit",
                "process scheduling algorithm",
                "GIC interrupt priority",
            ]

            report("phase", f"latency probe ({run_idx}/{run_count})")
            latency_data = measure_latency(latency_queries, use_reranker=use_reranker)
            latency_times = latency_data["total_times"]
            latency_p50 = statistics.median(latency_times)
            latency_p95 = _p95(latency_times)

            results = []

            # Track auto-fill stats.
            autofill_count = 0
            unjudged_count = 0
            gt_modified = False

            report("phase", f"evaluation ({run_idx}/{run_count})")
            with report_progress("evaluating") as progress:
                task = progress.add_task("", total=len(ground_truth["queries"]))

                for q in ground_truth["queries"]:
                    hits = run_search(
                        q["query"],
                        limit=max(k_values),
                        use_reranker=use_reranker,
                    )

                    judgments = {j["chunk_id"]: j["score"] for j in q["judgments"]}
                    judgment_list = q["judgments"]

                    missing_chunk_ids = [
                        h["chunk_id"] for h in hits if h["chunk_id"] not in judgments
                    ]
                    if missing_chunk_ids and not autofill:
                        preview = ", ".join(str(cid) for cid in missing_chunk_ids[:5])
                        raise RuntimeError(
                            "ground truth contains unjudged chunks "
                            f"for {q['id']} ({q['query']}): {len(missing_chunk_ids)} missing "
                            f"(sample: {preview})"
                        )

                    if autofill:
                        for h in hits:
                            chunk_id = h["chunk_id"]
                            if chunk_id not in judgments:
                                # Auto-fill: judge on the fly.
                                backend = get_backend()
                                chunk_info = backend.get_chunk_text(chunk_id)
                                if chunk_info is None:
                                    unjudged_count += 1
                                    continue

                                doc, text = chunk_info
                                j = judge_single_chunk(
                                    q["query"],
                                    chunk_id,
                                    text,
                                    doc,
                                    use_debiasing=autofill_use_debiasing,
                                )

                                # Add to in-memory ground truth.
                                new_judgment = {
                                    "chunk_id": j.chunk_id,
                                    "doc": j.doc,
                                    "score": j.score,
                                    "confidence": j.confidence,
                                    "subtopics": j.subtopics,
                                    "reason": j.reason,
                                    "auto_filled": True,
                                }
                                judgment_list.append(new_judgment)
                                judgments[chunk_id] = j.score
                                autofill_count += 1
                                gt_modified = True

                    subtopics = {
                        j["chunk_id"]: j.get("subtopics", [])
                        for j in judgment_list
                        if j["score"] >= 2
                    }

                    result_ids = [h["chunk_id"] for h in hits]
                    metrics = compute_metrics(result_ids, judgments, k_values=k_values)
                    div_metrics = compute_diversity_metrics(
                        hits, judgments, subtopics, k_values=k_values
                    )

                    metrics.subtopic_recall = div_metrics.subtopic_recall
                    metrics.alpha_ndcg = div_metrics.alpha_ndcg
                    metrics.doc_coverage = div_metrics.doc_coverage

                    results.append(
                        QueryResult(
                            query_id=q["id"],
                            query=q["query"],
                            category=q["category"],
                            metrics=metrics,
                        )
                    )
                    progress.update(task, advance=1)

            # Save updated ground truth if auto-fill added judgments.
            if gt_modified:
                _save_ground_truth(gt_path, ground_truth)
        finally:
            close_backend()

        # Separate negative queries from main metrics.
        positive_results = [r for r in results if r.category != "negative"]
        negative_results = [r for r in results if r.category == "negative"]

        agg = aggregate_metrics(positive_results) if positive_results else {}

        neg_fp_rate = {}
        if negative_results:
            for kv in k_values:
                fp_counts = [r.metrics.precision.get(kv, 0) for r in negative_results]
                neg_fp_rate[kv] = sum(fp_counts) / len(fp_counts) if fp_counts else 0

        by_cat = aggregate_by_category(results, k=k)
        run_duration = time.time() - run_start
        report(
            "pass-summary",
            f"{run_idx}/{run_count} nDCG@10 {_metric_at(agg.get('ndcg', {}), 10):.3f} | "
            f"MRR@10 {_metric_at(agg.get('mrr', {}), 10):.3f} | P50 {latency_p50:.0f}ms",
        )

        run_outputs.append(
            {
                "duration_s": run_duration,
                "latency_ms": {"p50": latency_p50, "p95": latency_p95},
                "metrics": agg,
                "negative_fp_rate": neg_fp_rate,
                "by_category": by_cat,
                "auto_filled": autofill_count,
                "unjudged": unjudged_count,
            }
        )

    latency_p50 = statistics.mean(r["latency_ms"]["p50"] for r in run_outputs)
    latency_p95 = statistics.mean(r["latency_ms"]["p95"] for r in run_outputs)
    p50_values = [r["latency_ms"]["p50"] for r in run_outputs]
    p95_values = [r["latency_ms"]["p95"] for r in run_outputs]

    agg = _average_metrics([r["metrics"] for r in run_outputs])
    neg_fp_rate = _average_neg_fp([r["negative_fp_rate"] for r in run_outputs])
    by_cat = _average_by_category([r["by_category"] for r in run_outputs])
    autofill_count = sum(int(r["auto_filled"]) for r in run_outputs)
    unjudged_count = sum(int(r["unjudged"]) for r in run_outputs)

    report("evaluated", f"in {fmt_duration(time.time() - all_start).strip()}")
    report(
        "latency-spread",
        f"P50 {min(p50_values):.0f}-{max(p50_values):.0f}ms | "
        f"P95 {min(p95_values):.0f}-{max(p95_values):.0f}ms",
    )
    if autofill_count > 0:
        report("auto-fill", f"judged {autofill_count} new chunks")
    if unjudged_count > 0:
        report("unjudged", f"{unjudged_count} chunks (no judgments)")
    report(
        "summary",
        " | ".join(
            [
                f"nDCG@10 {_metric_at(agg.get('ndcg', {}), 10):.3f}",
                f"MRR@10 {_metric_at(agg.get('mrr', {}), 10):.3f}",
                f"P50 {latency_p50:.0f}ms",
            ]
        ),
    )
    print_gap()
    blank = "\u2014"
    table = make_table(title="Results (3-run mean)")
    table.add_column("Metric")
    for kv in k_values:
        table.add_column(f"@{kv}", justify="right")

    # Latency — single values in @10 column.
    table.add_row("Latency P50", *[blank] * (len(k_values) - 1), f"{latency_p50:.0f}ms")
    table.add_row("Latency P95", *[blank] * (len(k_values) - 1), f"{latency_p95:.0f}ms")
    table.add_section()

    # IR metrics at all k cutoffs.
    for metric, label in [
        ("ndcg", "nDCG"),
        ("mrr", "MRR"),
        ("precision", "Precision"),
        ("map", "MAP"),
        ("recall", "Recall"),
        ("hit_rate", "Hit Rate"),
        ("doc_coverage", "Doc-Cov"),
        ("subtopic_recall", "S-Recall"),
        ("alpha_ndcg", "\u03b1-nDCG"),
    ]:
        row = [label] + [
            f"{_metric_at(agg.get(metric, {}), kv):.3f}" for kv in k_values
        ]
        table.add_row(*row)

    # FP rate at bottom.
    if neg_fp_rate:
        table.add_section()
        table.add_row(
            "FP Rate", *[f"{_metric_at(neg_fp_rate, kv):.3f}" for kv in k_values]
        )

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
        "name": name,
        "k": k,
        "runs": run_count,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reranker_enabled": use_reranker,
        "latency_ms": {"p50": round(latency_p50, 1), "p95": round(latency_p95, 1)},
        "metrics": agg,
        "negative_fp_rate": neg_fp_rate,
        "by_category": by_cat,
        "auto_filled": autofill_count,
        "unjudged": unjudged_count,
    }

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / f"{name}.json"
    json_path.write_text(json.dumps(output, indent=2))

    report("saved", f"{json_path.name}")


def cmd_show(run_name: str | None = None):
    """Show benchmark results."""
    import json
    from pathlib import Path

    results_dir = Path(__file__).parent / "results"
    if run_name == "list" or run_name is None:
        report_mode("bench.show", "list")
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
            report("status", "no benchmark runs found")
            report("hint", "run run <name> first")
            return

        table = make_table(title="Benchmark Runs")
        table.add_column("Name")
        table.add_column("Date", style="dim")
        table.add_column("nDCG", justify="right")
        table.add_column("Latency", justify="right", style="dim")

        for run_path in runs[:10]:
            try:
                data = json.loads(run_path.read_text())
            except Exception:
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
            report("warning", f"skipped {skipped} invalid run file(s)")
        report("hint", "use show <name> to view details")
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
    except Exception as e:
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
    report_mode("bench.show", f"{run_label}")
    report("date", created)
    if "reranker_enabled" in data:
        report(
            "reranker",
            "enabled" if bool(data.get("reranker_enabled")) else "disabled",
        )
    from .evaluate import STANDARD_K

    def get_val(d, k):
        return d.get(str(k), d.get(k, 0))

    blank = "\u2014"
    lat = data.get("latency_ms", {})
    neg_fp = data.get("negative_fp_rate", {})
    report(
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
        ("doc_coverage", "Doc-Cov"),
        ("subtopic_recall", "S-Recall"),
        ("alpha_ndcg", "\u03b1-nDCG"),
    ]:
        row = [label] + [f"{get_val(m.get(metric, {}), kv):.3f}" for kv in STANDARD_K]
        table.add_row(*row)

    # FP rate.
    if neg_fp:
        table.add_section()
        table.add_row("FP Rate", *[f"{get_val(neg_fp, kv):.3f}" for kv in STANDARD_K])

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
    p_run.add_argument("name", help="Benchmark run name (e.g. 'baseline-v2')")
    p_run.add_argument(
        "--autofill",
        action="store_true",
        help="Judge unjudged chunks during run (disabled by default)",
    )
    p_run.add_argument(
        "--reranker",
        action="store_true",
        help="Enable cross-encoder reranker for this run (disabled by default)",
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
        report("project", project.project_id)

        if args.command == "judge":
            cmd_judge()
        elif args.command == "run":
            cmd_run(
                name=args.name,
                autofill=bool(args.autofill),
                use_reranker=bool(args.reranker),
            )
        elif args.command == "show":
            cmd_show(run_name=args.name)
    finally:
        config.clear_active_project()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        report("status", "interrupted")
        sys.exit(130)
    except Exception as e:
        _report_exception(e)
        sys.exit(1)
