"""Benchmark CLI with Rich UI matching sova style."""

import sys
import time

from rich.console import Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from sova.config import DATA_DIR
from sova.ui import console, fmt_duration, report, report_progress


def cmd_judge(use_debiasing: bool = True):
    """Generate ground truth judgments."""
    from .judge import QUERY_SET, judge_query, JUDGE_MODEL, collect_query_subtopics
    from .search_interface import close_backend
    from sova.config import DB_PATH
    import json

    if not DB_PATH.exists():
        console.print("[red]error:[/red] no database, run sova indexing first")
        sys.exit(1)

    checkpoint_path = DATA_DIR / "ground_truth_partial.json"
    output_path = DATA_DIR / "ground_truth.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing progress
    completed: dict[str, dict] = {}
    if checkpoint_path.exists():
        try:
            data = json.loads(checkpoint_path.read_text())
            for q in data.get("queries", []):
                completed[q["id"]] = q
        except Exception:
            pass

    remaining = [spec for spec in QUERY_SET if spec.id not in completed]

    report("model", JUDGE_MODEL)
    report("mode", "debiasing enabled" if use_debiasing else "debiasing disabled")
    console.print()

    start = time.time()
    if not remaining:
        report("status", "all queries already judged")
    else:
        k = 10

        def save_checkpoint():
            ground_truth = {
                "version": "1.0",
                "created": time.strftime("%Y-%m-%d"),
                "judge_model": JUDGE_MODEL,
                "candidates_per_query": k,
                "use_debiasing": use_debiasing,
                "queries": list(completed.values()),
            }
            checkpoint_path.write_text(json.dumps(ground_truth, indent=2))

        total = len(QUERY_SET)
        done = len(completed)

        progress = Progress(BarColumn(bar_width=30), TimeElapsedColumn())
        task = progress.add_task("", total=total, completed=done)

        def _display():
            return Group(Text(f"queries: {done}/{total}"), progress)

        with Live(_display(), console=console, transient=True) as live:
            for spec in remaining:
                qj = judge_query(spec, k=k, verbose=False, use_debiasing=use_debiasing)

                extracted_subtopics = collect_query_subtopics(qj.judgments)
                all_subtopics = sorted(set(qj.query.subtopics + extracted_subtopics))

                query_data = {
                    "id": qj.query.id,
                    "query": qj.query.query,
                    "category": qj.query.category,
                    "subtopics": all_subtopics,
                    "judgments": [
                        {
                            "chunk_id": j.chunk_id,
                            "doc": j.doc,
                            "score": j.score,
                            "confidence": j.confidence,
                            "subtopics": j.subtopics,
                            "reason": j.reason,
                        }
                        for j in qj.judgments
                    ],
                }
                completed[spec.id] = query_data
                save_checkpoint()
                done += 1
                progress.update(task, completed=done)
                live.update(_display())

        close_backend()
        report(
            "judged", f"{len(remaining)} queries in {fmt_duration(time.time() - start)}"
        )

    # Build final output in query order
    queries_list = [completed[spec.id] for spec in QUERY_SET if spec.id in completed]
    ground_truth = {
        "version": "1.0",
        "created": time.strftime("%Y-%m-%d"),
        "judge_model": JUDGE_MODEL,
        "candidates_per_query": k if remaining else 10,
        "use_debiasing": use_debiasing,
        "queries": queries_list,
    }

    total_judgments = 0
    score_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for q in queries_list:
        for j in q["judgments"]:
            total_judgments += 1
            score_counts[j["score"]] = score_counts.get(j["score"], 0) + 1

    output_path.write_text(json.dumps(ground_truth, indent=2))
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    report("judged", f"{total_judgments} chunks in {fmt_duration(time.time() - start)}")
    report("saved", f"[bold]{output_path.name}[/bold]")
    console.print()
    table = Table(title="Score Distribution", show_header=True, header_style="dim")
    table.add_column("Score")
    table.add_column("Count", justify="right")
    table.add_column("", justify="right")

    labels = ["Not Relevant", "Marginal", "Relevant", "Highly Relevant"]
    for score in range(4):
        count = score_counts[score]
        pct = count / total_judgments * 100 if total_judgments else 0
        bar = "[green]" + "█" * int(pct / 5) + "[/green]"
        table.add_row(f"{score} {labels[score]}", str(count), f"{pct:.0f}% {bar}")

    console.print(table)


def cmd_run(name: str | None = None):
    """Run benchmark against ground truth."""
    from .run_benchmark import run_search
    from .evaluate import aggregate_metrics, aggregate_by_category
    from .search_interface import measure_latency, clear_cache, close_backend
    import json
    from pathlib import Path
    import statistics

    if not name:
        console.print("[red]error:[/red] name required")
        console.print(
            "[dim]usage:[/dim] [bold]run <name>[/bold]  (e.g., 'phase1-baseline')"
        )
        sys.exit(1)

    gt_path = DATA_DIR / "ground_truth.json"
    if not gt_path.exists():
        console.print("[red]error:[/red] no ground truth")
        console.print("run [bold]judge[/bold] first")
        sys.exit(1)

    report("name", f"[bold]{name}[/bold]")

    ground_truth = json.loads(gt_path.read_text())
    report("queries", str(len(ground_truth["queries"])))

    clear_cache()
    latency_queries = [
        "ARM exception handling",
        "RISC-V trap handling",
        "memory protection unit",
        "process scheduling algorithm",
        "GIC interrupt priority",
    ]

    report("latency", "measuring...")
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

    start = time.time()
    with report_progress("evaluating") as progress:
        task = progress.add_task("", total=len(ground_truth["queries"]))

        for q in ground_truth["queries"]:
            hits = run_search(q["query"], limit=max(k_values))

            judgments = {j["chunk_id"]: j["score"] for j in q["judgments"]}
            subtopics = {
                j["chunk_id"]: j.get("subtopics", [])
                for j in q["judgments"]
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

    report("evaluated", f"in {fmt_duration(time.time() - start)}")
    console.print()

    blank = "[dim]—[/dim]"
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
        ("alpha_ndcg", "α-nDCG"),
    ]:
        row = [label] + [f"{agg.get(metric, {}).get(kv, 0):.3f}" for kv in k_values]
        table.add_row(*row)

    # FP rate at bottom
    if neg_fp_rate:
        table.add_section()
        table.add_row("FP Rate", *[f"{neg_fp_rate.get(kv, 0):.3f}" for kv in k_values])

    console.print(table)

    by_cat = aggregate_by_category(results, k=k)

    output = {
        "name": name,
        "k": k,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "latency_ms": {"p50": round(latency_p50, 1), "p95": round(latency_p95, 1)},
        "metrics": agg,
        "negative_fp_rate": neg_fp_rate,
        "by_category": by_cat,
    }

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / f"{name}.json"
    json_path.write_text(json.dumps(output, indent=2))

    console.print()
    report("saved", f"[bold]{json_path.name}[/bold]")


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
            console.print("[dim]no benchmark runs found[/dim]")
            console.print("run [bold]run <name>[/bold] first")
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
        console.print("[dim]use[/dim] show <name> [dim]to view details[/dim]")
        return
    results_path = results_dir / f"{run_name}.json"
    if not results_path.exists():
        console.print(f"[red]error:[/red] run '{run_name}' not found")
        console.print("use [bold]show[/bold] to list available runs")
        sys.exit(1)

    data = json.loads(results_path.read_text())
    k = data["k"]
    m = data.get("metrics", {})

    report("run", f"[bold]{data.get('name', run_name)}[/bold]")
    report("date", data.get("created", "unknown"))
    console.print()
    from .evaluate import STANDARD_K

    def get_val(d, k):
        return d.get(str(k), d.get(k, 0))

    blank = "[dim]—[/dim]"
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
        console.print("[dim]By Category:[/dim]")
        for cat, metrics in sorted(by_cat.items()):
            console.print(
                f"  {cat:20} nDCG={metrics['ndcg']:.3f}  MRR={metrics['mrr']:.3f}"
            )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="Sova benchmark suite",
    )
    sub = parser.add_subparsers(dest="command")

    p_judge = sub.add_parser("judge", help="Generate ground truth judgments")
    p_judge.add_argument(
        "--no-debias", action="store_true", help="Skip debiasing (faster)"
    )

    p_run = sub.add_parser("run", help="Run benchmark against ground truth")
    p_run.add_argument("name", help="Benchmark run name (e.g. 'baseline-v2')")

    p_show = sub.add_parser("show", help="Display benchmark results")
    p_show.add_argument(
        "name", nargs="?", default=None, help="Run name (omit to list all)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "judge":
        cmd_judge(use_debiasing=not args.no_debias)
    elif args.command == "run":
        cmd_run(name=args.name)
    elif args.command == "show":
        cmd_show(run_name=args.name)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted[/dim]")
        sys.exit(130)
