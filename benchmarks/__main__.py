"""Benchmark CLI with Rich UI matching sova style."""

import sys
import time

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from sova.config import DATA_DIR

console = Console()


def _label(name: str) -> str:
    padded = f"{name}:".ljust(8)
    return f"[dim]{padded}[/dim]"


def report(name: str, msg: str) -> None:
    console.print(f"{_label(name)} {msg}")


def report_progress(name: str) -> Progress:
    return Progress(
        TextColumn(_label(name)),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:>6.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:>6.1f}m"
    return f"{seconds / 3600:>6.1f}h"


def cmd_calibrate():
    """Calibrate LLM judge against human labels."""
    from .calibration import (
        generate_calibration_pairs,
        validate_calibration,
        MIN_KAPPA,
    )
    from sova.config import DB_PATH
    import json

    pairs_path = DATA_DIR / "calibration_pairs.json"

    if not pairs_path.exists():
        if not DB_PATH.exists():
            console.print("[red]error:[/red] no database, run sova indexing first")
            sys.exit(1)
        report("step", "1/3 generating calibration pairs")
        generate_calibration_pairs(n=50, output_path=pairs_path, verbose=False)
        report("pairs", f"50 saved to [bold]{pairs_path.name}[/bold]")
        console.print()
        console.print("[yellow]action required[/yellow]")
        console.print(f"  1. open [bold]{pairs_path}[/bold]")
        console.print("  2. fill in 'human_score' (0-3) for each pair")
        console.print("  3. run [bold]calibrate[/bold] again")
        return

    data = json.loads(pairs_path.read_text())
    has_human = sum(1 for p in data["pairs"] if p.get("human_score") is not None)
    has_llm = sum(1 for p in data["pairs"] if p.get("llm_score") is not None)
    total = len(data["pairs"])

    report("pairs", f"{total} loaded")
    report("human", f"{has_human}/{total} labeled")
    report("llm", f"{has_llm}/{total} judged")

    if has_human == 0:
        console.print()
        console.print("[yellow]no human labels found[/yellow]")
        console.print(f"edit [bold]{pairs_path}[/bold] and fill in 'human_score'")
        return

    if has_llm < has_human:
        report("step", "2/3 running LLM judgments")
        start = time.time()

        with report_progress("judging") as progress:
            task = progress.add_task("", total=total - has_llm)

            from .judge import judge_chunk, JUDGE_MODEL

            for pair in data["pairs"]:
                if pair.get("llm_score") is not None:
                    continue
                score, reason, conf, subs = judge_chunk(pair["query"], pair["text"])
                pair["llm_score"] = score
                pair["llm_reason"] = reason
                progress.update(task, advance=1)

        data["llm_model"] = JUDGE_MODEL
        pairs_path.write_text(json.dumps(data, indent=2))
        report(
            "judging", f"{total - has_llm} done in {fmt_duration(time.time() - start)}"
        )

    report("step", "3/3 validating agreement")
    result = validate_calibration(pairs_path, verbose=False)

    console.print()
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    table.add_column("Target", style="dim")

    kappa_style = "green" if result.kappa >= MIN_KAPPA else "red"
    table.add_row(
        "Cohen's kappa",
        f"[{kappa_style}]{result.kappa:.3f}[/{kappa_style}]",
        f">= {MIN_KAPPA}",
    )
    table.add_row("Agreement", f"{result.agreement_rate:.1%}", "")
    table.add_row("Bias", f"{result.bias:+.3f}", "")
    table.add_row("Pairs", str(result.n_pairs), "")
    console.print(table)

    if result.kappa >= MIN_KAPPA:
        console.print("\n[green]pass[/green] LLM judge is calibrated")
    else:
        console.print(
            "\n[red]fail[/red] agreement too low, consider more labels or different model"
        )


def cmd_judge():
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
    if completed:
        report("queries", f"{len(remaining)} remaining ({len(completed)}/{len(QUERY_SET)} done)")
    else:
        report("queries", str(len(QUERY_SET)))
    report("mode", "debiasing enabled")
    console.print()

    start = time.time()
    if not remaining:
        report("status", "all queries already judged")
    else:

        def save_checkpoint():
            ground_truth = {
                "version": "1.0",
                "created": time.strftime("%Y-%m-%d"),
                "judge_model": JUDGE_MODEL,
                "candidates_per_query": 10,
                "use_debiasing": True,
                "queries": list(completed.values()),
            }
            checkpoint_path.write_text(json.dumps(ground_truth, indent=2))

        k = 10
        with report_progress("judging") as progress:
            task = progress.add_task("", total=len(remaining) * k)
            for qi, spec in enumerate(remaining):
                progress.console.print(
                    f"[dim]{spec.id}[/dim] {spec.query[:50]}"
                    f"[dim]({qi + 1 + len(completed)}/{len(QUERY_SET)})[/dim]"
                )
                qj = judge_query(
                    spec, k=k, verbose=False, use_debiasing=True,
                    on_chunk_done=lambda: progress.update(task, advance=1),
                )

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
                progress.update(task, advance=1)

        close_backend()
        report("judged", f"{len(remaining)} queries in {fmt_duration(time.time() - start)}")

    # Build final output in query order
    queries_list = [completed[spec.id] for spec in QUERY_SET if spec.id in completed]
    ground_truth = {
        "version": "1.0",
        "created": time.strftime("%Y-%m-%d"),
        "judge_model": JUDGE_MODEL,
        "candidates_per_query": 10,
        "use_debiasing": True,
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
        console.print("[dim]usage:[/dim] [bold]run <name>[/bold]  (e.g., 'phase1-baseline')")
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
    latency_queries = ["ARM exception handling", "memory barrier", "page table"]

    report("latency", "measuring...")
    latency_data = measure_latency(latency_queries)
    latency_times = latency_data["total_times"]

    latency_avg = statistics.mean(latency_times)
    latency_p50 = statistics.median(latency_times)
    report("latency", f"avg={latency_avg:.0f}ms  p50={latency_p50:.0f}ms")
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
    agg = aggregate_metrics(results)
    report("evaluated", f"in {fmt_duration(time.time() - start)}")
    console.print()
    table = Table(title="Results", show_header=True, header_style="dim")
    table.add_column("Metric")
    for kv in k_values:
        table.add_column(f"@{kv}", justify="right")

    for metric in ["ndcg", "mrr", "precision", "subtopic_recall", "doc_coverage"]:
        label = {"subtopic_recall": "S-Recall", "doc_coverage": "Doc-Cov"}.get(
            metric, metric.upper()
        )
        row = [label] + [f"{agg[metric].get(kv, 0):.3f}" for kv in k_values]
        table.add_row(*row)
    console.print(table)
    by_cat = aggregate_by_category(results, k=k)

    output = {
        "name": name,
        "k": k,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "latency_ms": {"avg": round(latency_avg, 1), "p50": round(latency_p50, 1)},
        "metrics": agg,
        "by_category": by_cat,
    }

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / f"{name}.json"
    json_path.write_text(json.dumps(output, indent=2))
    md_header = "| Metric | " + " | ".join(f"@{kv}" for kv in k_values) + " |"
    md_sep = "|--------|" + "|".join("------" for _ in k_values) + "|"
    md_lines = [
        f"# Benchmark: {name}",
        "",
        f"**Date:** {output['created']}",
        f"**Latency:** {latency_avg:.0f}ms avg, {latency_p50:.0f}ms p50",
        "",
        "## Results",
        "",
        md_header,
        md_sep,
    ]
    for metric in ["ndcg", "mrr", "precision", "subtopic_recall", "doc_coverage"]:
        label = {"subtopic_recall": "S-Recall", "doc_coverage": "Doc-Cov"}.get(
            metric, metric.upper()
        )
        vals = " | ".join(f"{agg[metric].get(kv, 0):.3f}" for kv in k_values)
        md_lines.append(f"| {label} | {vals} |")
    md_lines += [
        "",
        "## By Category",
        "",
        "| Category | nDCG@10 | MRR@10 |",
        "|----------|---------|--------|",
    ]
    for cat, m in sorted(by_cat.items()):
        md_lines.append(f"| {cat} | {m['ndcg']:.3f} | {m['mrr']:.3f} |")

    md_path = results_dir / f"{name}.md"
    md_path.write_text("\n".join(md_lines))

    console.print()
    report("saved", f"[bold]{json_path.name}[/bold]")
    report("report", f"[bold]{md_path.name}[/bold]")


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
            lat = data.get("latency_ms", {}).get("avg", 0)
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
    lat = data.get("latency_ms", {})
    if lat:
        report(
            "latency", f"{lat.get('avg', 0):.0f}ms avg, {lat.get('p50', 0):.0f}ms p50"
        )
    console.print()
    from .evaluate import STANDARD_K

    table = Table(title="Results", show_header=True, header_style="dim")
    table.add_column("Metric")
    for kv in STANDARD_K:
        table.add_column(f"@{kv}", justify="right")

    def get_val(d, k):
        return d.get(str(k), d.get(k, 0))

    for metric in ["ndcg", "mrr", "precision", "subtopic_recall", "doc_coverage"]:
        label = {"subtopic_recall": "S-Recall", "doc_coverage": "Doc-Cov"}.get(
            metric, metric.upper()
        )
        row = [label] + [f"{get_val(m.get(metric, {}), kv):.3f}" for kv in STANDARD_K]
        table.add_row(*row)
    console.print(table)
    by_cat = data.get("by_category", {})
    if by_cat:
        console.print()
        console.print("[dim]By Category:[/dim]")
        for cat, metrics in sorted(by_cat.items()):
            console.print(
                f"  {cat:20} nDCG={metrics['ndcg']:.3f}  MRR={metrics['mrr']:.3f}"
            )


def cmd_latency():
    """Measure query latency."""
    from .search_interface import get_backend, clear_cache, close_backend
    import statistics

    try:
        backend = get_backend()
    except FileNotFoundError:
        console.print("[red]error:[/red] no database, run sova indexing first")
        sys.exit(1)

    queries = [
        "ARM exception handling",
        "RISC-V trap handling",
        "memory protection unit",
        "process scheduling algorithm",
        "GIC interrupt priority",
    ]

    clear_cache()

    embed_times = []
    search_times = []
    total_times = []

    console.print()
    report("queries", str(len(queries)))
    report("cache", "disabled (cold)")
    console.print()

    for q in queries:
        t0 = time.perf_counter()
        emb = backend._embed_query(q)
        t1 = time.perf_counter()
        embed_ms = (t1 - t0) * 1000
        backend.search(q, limit=10, embedding=emb)
        t2 = time.perf_counter()
        search_ms = (t2 - t1) * 1000

        total_ms = (t2 - t0) * 1000
        embed_times.append(embed_ms)
        search_times.append(search_ms)
        total_times.append(total_ms)

        console.print(f"  {q[:35]:35} [dim]{total_ms:6.0f}ms[/dim]")

    close_backend()

    console.print()
    table = Table(show_header=True, header_style="dim", title="Latency (ms)")
    table.add_column("Stage")
    table.add_column("Avg", justify="right")
    table.add_column("P50", justify="right")
    table.add_column("P95", justify="right")

    def p50(arr):
        return statistics.median(arr)

    def p95(arr):
        s = sorted(arr)
        return s[int(len(s) * 0.95)] if len(s) >= 2 else s[-1]

    table.add_row(
        "Embed",
        f"{statistics.mean(embed_times):.0f}",
        f"{p50(embed_times):.0f}",
        f"{p95(embed_times):.0f}",
    )
    table.add_row(
        "Search",
        f"{statistics.mean(search_times):.0f}",
        f"{p50(search_times):.0f}",
        f"{p95(search_times):.0f}",
    )
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{statistics.mean(total_times):.0f}[/bold]",
        f"[bold]{p50(total_times):.0f}[/bold]",
        f"[bold]{p95(total_times):.0f}[/bold]",
    )

    console.print(table)

    target = 500
    avg = statistics.mean(total_times)
    if avg <= target:
        console.print(f"\n[green]target met[/green] {avg:.0f}ms <= {target}ms")
    else:
        console.print(f"\n[yellow]above target[/yellow] {avg:.0f}ms > {target}ms")


def main():
    if len(sys.argv) < 2:
        console.print()
        console.print("[dim]usage:[/dim]")
        console.print(
            "  python -m benchmarks [bold]calibrate[/bold]       validate LLM judge"
        )
        console.print(
            "  python -m benchmarks [bold]judge[/bold]           generate ground truth"
        )
        console.print(
            "  python -m benchmarks [bold]run[/bold] <name>      run benchmark"
        )
        console.print(
            "  python -m benchmarks [bold]show[/bold] \\[name]    display results"
        )
        console.print(
            "  python -m benchmarks [bold]latency[/bold]         measure query latency"
        )
        sys.exit(0)

    cmd = sys.argv[1]
    arg = sys.argv[2] if len(sys.argv) > 2 else None

    if cmd == "calibrate":
        cmd_calibrate()
    elif cmd == "judge":
        cmd_judge()
    elif cmd == "run":
        cmd_run(name=arg)
    elif cmd == "show":
        cmd_show(run_name=arg)
    elif cmd == "latency":
        cmd_latency()
    else:
        console.print(f"[red]error:[/red] unknown command '{cmd}'")
        console.print("use: calibrate, judge, run, show, or latency")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted[/dim]")
        sys.exit(130)
