"""Benchmark CLI with Rich UI matching sova style."""

import sys
import time
from pathlib import Path

from rich.console import Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from sova.config import DATA_DIR
from sova.ui import console, fmt_duration, report, report_progress

_BENCH_DIR = Path(__file__).parent


def _load_ground_truth(path: Path) -> dict | None:
    """Load ground truth JSON, returning None if missing."""
    import json

    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _save_ground_truth(path: Path, gt: dict):
    """Atomically save ground truth JSON."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gt, indent=2))


def _build_ground_truth(
    queries_list: list[dict],
    k_per_strategy: int,
    judge_model: str,
    use_debiasing: bool,
) -> dict:
    """Build v2.0 ground truth envelope."""
    return {
        "version": "2.0",
        "created": time.strftime("%Y-%m-%d"),
        "pooling": ["hybrid", "fts", "vector"],
        "k_per_strategy": k_per_strategy,
        "judge_model": judge_model,
        "use_debiasing": use_debiasing,
        "queries": queries_list,
    }


def cmd_judge(use_debiasing: bool = True):
    """Generate ground truth judgments with multi-source pooling."""
    from .judge import QUERY_SET, judge_query, JUDGE_MODEL, collect_query_subtopics
    from .search_interface import close_backend
    from sova.config import DB_PATH
    import json

    if not DB_PATH.exists():
        console.print("[red]error:[/red] no database, run sova indexing first")
        sys.exit(1)

    checkpoint_path = DATA_DIR / "ground_truth_partial.json"
    output_path = _BENCH_DIR / "ground_truth.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    k_per_strategy = 20

    # Load existing ground truth (supports incremental judging)
    existing_gt = _load_ground_truth(output_path)
    existing_queries: dict[str, dict] = {}
    if existing_gt:
        for q in existing_gt.get("queries", []):
            existing_queries[q["id"]] = q

    # Also load partial checkpoint (interrupted previous run)
    partial_gt = _load_ground_truth(checkpoint_path)
    if partial_gt:
        for q in partial_gt.get("queries", []):
            if q["id"] not in existing_queries:
                existing_queries[q["id"]] = q

    report("model", JUDGE_MODEL)
    report("mode", "debiasing enabled" if use_debiasing else "debiasing disabled")
    report("pooling", f"hybrid + fts + vector @ k={k_per_strategy}")
    if existing_queries:
        total_existing = sum(
            len(q.get("judgments", [])) for q in existing_queries.values()
        )
        report(
            "existing",
            f"{total_existing} judgments across {len(existing_queries)} queries",
        )
    console.print()

    start = time.time()

    # For each query, build existing_judgments map for incremental judging
    completed: dict[str, dict] = dict(existing_queries)
    new_judgments_total = 0

    queries_to_process = list(QUERY_SET)

    total = len(queries_to_process)
    done = 0

    def save_checkpoint():
        queries_list = [completed[s.id] for s in QUERY_SET if s.id in completed]
        gt = _build_ground_truth(
            queries_list, k_per_strategy, JUDGE_MODEL, use_debiasing
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps(gt, indent=2))

    progress = Progress(BarColumn(bar_width=30), TimeElapsedColumn())
    task = progress.add_task("", total=total, completed=done)

    def _display():
        return Group(
            Text(f"queries: {done}/{total}  new judgments: {new_judgments_total}"),
            progress,
        )

    from .judge import JudgeError, Judgment as _J

    rate_limited = False
    with Live(_display(), console=console, transient=True) as live:
        for spec in queries_to_process:
            # Build map of already-judged chunk_ids for this query
            existing_for_query: dict[int, int] = {}
            if spec.id in completed:
                for j in completed[spec.id].get("judgments", []):
                    existing_for_query[j["chunk_id"]] = j["score"]

            # Per-chunk checkpoint: merge each judgment immediately
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

                # Update completed with partial progress and checkpoint
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
                console.print(f"[yellow]stopped:[/yellow] {e}")
                console.print(
                    f"\nprogress saved ({done}/{total} queries). re-run to continue."
                )
                break

            done += 1
            progress.update(task, completed=done)
            live.update(_display())

    if rate_limited:
        sys.exit(1)

    close_backend()

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

    # Build final output in query order
    queries_list = [completed[spec.id] for spec in QUERY_SET if spec.id in completed]
    ground_truth = _build_ground_truth(
        queries_list, k_per_strategy, JUDGE_MODEL, use_debiasing
    )

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
    console.print()
    table = Table(title="Score Distribution", show_header=True, header_style="dim")
    table.add_column("Score")
    table.add_column("Count", justify="right")
    table.add_column("", justify="right")

    labels = ["Not Relevant", "Marginal", "Relevant", "Highly Relevant"]
    for score in range(4):
        count = score_counts[score]
        pct = count / total_judgments * 100 if total_judgments else 0
        bar = "[green]" + "\u2588" * int(pct / 5) + "[/green]"
        table.add_row(f"{score} {labels[score]}", str(count), f"{pct:.0f}% {bar}")

    console.print(table)


def cmd_run(name: str | None = None, no_autofill: bool = False):
    """Run benchmark against ground truth with auto-fill for unjudged chunks."""
    from .run_benchmark import run_search
    from .evaluate import aggregate_metrics, aggregate_by_category
    from .search_interface import (
        measure_latency,
        clear_cache,
        close_backend,
        get_backend,
    )
    from .judge import judge_single_chunk
    import json
    from pathlib import Path
    import statistics

    if not name:
        console.print("[red]error:[/red] name required")
        console.print("usage: run <name>  (e.g., 'phase1-baseline')")
        sys.exit(1)

    gt_path = _BENCH_DIR / "ground_truth.json"
    if not gt_path.exists():
        console.print("[red]error:[/red] no ground truth")
        console.print("run judge first")
        sys.exit(1)

    report("name", f"{name}")

    ground_truth = json.loads(gt_path.read_text())
    report("queries", str(len(ground_truth["queries"])))
    if no_autofill:
        report("auto-fill", "disabled")

    clear_cache()
    latency_queries = [
        "ARM exception handling",
        "RISC-V trap handling",
        "memory protection unit",
        "process scheduling algorithm",
        "GIC interrupt priority",
    ]

    report("latency", "measuring")
    latency_data = measure_latency(latency_queries)
    latency_times = latency_data["total_times"]

    def _p95(arr):
        s = sorted(arr)
        return s[int(len(s) * 0.95)] if len(s) >= 20 else s[-1]

    latency_p50 = statistics.median(latency_times)
    latency_p95 = _p95(latency_times)
    console.print()
    from .evaluate import (
        compute_metrics,
        compute_diversity_metrics,
        STANDARD_K,
        QueryResult,
    )

    k = 10
    k_values = STANDARD_K
    results = []

    # Track auto-fill stats
    autofill_count = 0
    unjudged_count = 0
    gt_modified = False

    start = time.time()
    with report_progress("evaluating") as progress:
        task = progress.add_task("", total=len(ground_truth["queries"]))

        for q in ground_truth["queries"]:
            hits = run_search(q["query"], limit=max(k_values))

            judgments = {j["chunk_id"]: j["score"] for j in q["judgments"]}
            judgment_list = q["judgments"]

            # Check for unjudged chunks and auto-fill
            for h in hits:
                chunk_id = h["chunk_id"]
                if chunk_id not in judgments:
                    if no_autofill:
                        unjudged_count += 1
                        continue

                    # Auto-fill: judge on the fly
                    backend = get_backend()
                    chunk_info = backend.get_chunk_text(chunk_id)
                    if chunk_info is None:
                        unjudged_count += 1
                        continue

                    doc, text = chunk_info
                    j = judge_single_chunk(q["query"], chunk_id, text, doc)

                    # Add to in-memory ground truth
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

    # Save updated ground truth if auto-fill added judgments
    if gt_modified:
        _save_ground_truth(gt_path, ground_truth)

    close_backend()

    # Separate negative queries from main metrics
    positive_results = [r for r in results if r.category != "negative"]
    negative_results = [r for r in results if r.category == "negative"]

    agg = aggregate_metrics(positive_results) if positive_results else {}

    # Compute false positive rate for negative queries
    neg_fp_rate = {}
    if negative_results:
        for kv in k_values:
            fp_counts = [r.metrics.precision.get(kv, 0) for r in negative_results]
            neg_fp_rate[kv] = sum(fp_counts) / len(fp_counts) if fp_counts else 0

    report("evaluated", f"in {fmt_duration(time.time() - start).strip()}")
    if autofill_count > 0:
        report("auto-fill", f"judged {autofill_count} new chunks")
    if unjudged_count > 0:
        report("unjudged", f"{unjudged_count} chunks (no judgments)")
    console.print()

    blank = "\u2014"
    table = Table(title="Results", show_header=True, header_style="dim")
    table.add_column("Metric")
    for kv in k_values:
        table.add_column(f"@{kv}", justify="right")

    # Latency — single values in @10 column
    table.add_row("Latency P50", *[blank] * (len(k_values) - 1), f"{latency_p50:.0f}ms")
    table.add_row("Latency P95", *[blank] * (len(k_values) - 1), f"{latency_p95:.0f}ms")
    table.add_section()

    # IR metrics at all k cutoffs
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
        row = [label] + [f"{agg.get(metric, {}).get(kv, 0):.3f}" for kv in k_values]
        table.add_row(*row)

    # FP rate at bottom
    if neg_fp_rate:
        table.add_section()
        table.add_row("FP Rate", *[f"{neg_fp_rate.get(kv, 0):.3f}" for kv in k_values])

    console.print(table)

    by_cat = aggregate_by_category(results, k=k)

    if by_cat:
        console.print()
        cat_table = Table(title="By Category", show_header=True, header_style="dim")
        cat_table.add_column("Category")
        cat_table.add_column("nDCG", justify="right")
        cat_table.add_column("MRR", justify="right")
        cat_table.add_column("Precision", justify="right")
        cat_table.add_column("Recall", justify="right")
        for cat, metrics in sorted(by_cat.items()):
            cat_table.add_row(
                cat,
                f"{metrics['ndcg']:.3f}",
                f"{metrics['mrr']:.3f}",
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
            )
        console.print(cat_table)

    output = {
        "name": name,
        "k": k,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
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

    console.print()
    report("saved", f"{json_path.name}")


def cmd_show(run_name: str | None = None):
    """Show benchmark results."""
    import json
    from pathlib import Path

    results_dir = Path(__file__).parent / "results"
    if run_name == "list" or run_name is None:
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
            console.print("no benchmark runs found")
            console.print("run run <name> first")
            return

        table = Table(title="Benchmark Runs", show_header=True, header_style="dim")
        table.add_column("Name")
        table.add_column("Date", style="dim")
        table.add_column("nDCG", justify="right")
        table.add_column("Latency", justify="right", style="dim")

        for run_path in runs[:10]:
            data = json.loads(run_path.read_text())
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

        console.print(table)
        console.print()
        console.print("use show <name> to view details")
        return
    results_path = results_dir / f"{run_name}.json"
    if not results_path.exists():
        console.print(f"[red]error:[/red] run '{run_name}' not found")
        console.print("use show to list available runs")
        sys.exit(1)

    data = json.loads(results_path.read_text())
    k = data["k"]
    m = data.get("metrics", {})

    report("run", f"{data.get('name', run_name)}")
    report("date", data.get("created", "unknown"))
    console.print()
    from .evaluate import STANDARD_K

    def get_val(d, k):
        return d.get(str(k), d.get(k, 0))

    blank = "\u2014"
    lat = data.get("latency_ms", {})
    neg_fp = data.get("negative_fp_rate", {})

    table = Table(title="Results", show_header=True, header_style="dim")
    table.add_column("Metric")
    for kv in STANDARD_K:
        table.add_column(f"@{kv}", justify="right")

    # Latency
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

    # IR metrics
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

    # FP rate
    if neg_fp:
        table.add_section()
        table.add_row("FP Rate", *[f"{get_val(neg_fp, kv):.3f}" for kv in STANDARD_K])

    console.print(table)
    by_cat = data.get("by_category", {})
    if by_cat:
        console.print()
        cat_table = Table(title="By Category", show_header=True, header_style="dim")
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
        console.print(cat_table)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="Sova benchmark suite",
    )
    sub = parser.add_subparsers(dest="command")

    p_judge = sub.add_parser("judge", help="Generate ground truth judgments")
    p_judge.add_argument(
        "--debias",
        action="store_true",
        help="Enable debiasing (re-judge borderline chunks)",
    )

    p_run = sub.add_parser("run", help="Run benchmark against ground truth")
    p_run.add_argument("name", help="Benchmark run name (e.g. 'baseline-v2')")
    p_run.add_argument(
        "--no-autofill",
        action="store_true",
        help="Skip auto-judging unjudged chunks (just report count)",
    )

    p_show = sub.add_parser("show", help="Display benchmark results")
    p_show.add_argument(
        "name", nargs="?", default=None, help="Run name (omit to list all)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "judge":
        cmd_judge(use_debiasing=args.debias)
    elif args.command == "run":
        cmd_run(name=args.name, no_autofill=args.no_autofill)
    elif args.command == "show":
        cmd_show(run_name=args.name)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\ninterrupted")
        sys.exit(130)
