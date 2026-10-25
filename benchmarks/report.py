"""Score each saved run on its own and list all results in one table.

A result does not refer to another run. To compare runs, read the table.
Models do not start.
"""

from pathlib import Path
from statistics import mean, median

import numpy as np

from benchmarks.corpus import CorpusError, _write_atomic
from benchmarks.evaluate import (
    PROTOCOL,
    architecture_metrics,
    interval,
    passage_metrics,
)
from benchmarks.experiments import EXPERIMENT_DIR, load_experiment
from benchmarks.queries import load_evaluation_queries
from benchmarks.storage import read, write

METRICS = (
    "ndcg@1",
    "ndcg@3",
    "ndcg@5",
    "ndcg@10",
    "mrr@10",
    "precision@10",
    "recall@10",
    "hit@10",
)
STATUSES = ("unjudged", "reference_gap", "changed_query")
SLICES = ("category", "topic", "architecture_scope", "query_style")


def _mean(values):
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def _score_query(value, qrels, tags, scope):
    top = passage_metrics(value["hits"], qrels, 10)
    if top["status"] != "scored":
        return {"status": top["status"]}
    scores = {
        f"ndcg@{k}": passage_metrics(value["hits"], qrels, k)["metrics"]["ndcg"]
        for k in PROTOCOL["cutoffs"]
        if k != 10
    }
    return (
        {"status": "scored"}
        | scores
        | {f"{key}@10": value for key, value in top["metrics"].items()}
        | {"architecture": architecture_metrics(value["hits"], qrels, tags, scope)}
    )


def _aggregate(per_query, queries):
    ids = [q["id"] for q in queries]
    scored = [qid for qid in ids if per_query[qid]["status"] == "scored"]
    return (
        {"queries": len(ids), "scored": len(scored)}
        | {
            status: sorted(qid for qid in ids if per_query[qid]["status"] == status)
            for status in STATUSES
        }
        | {
            metric: _mean(per_query[qid][metric] for qid in scored)
            for metric in METRICS
        }
    )


def _slice(per_query, queries):
    scored = [q["id"] for q in queries if per_query[q["id"]]["status"] == "scored"]
    return {
        "queries": len(queries),
        "scored": len(scored),
        "ndcg@10": _mean(per_query[qid]["ndcg@10"] for qid in scored),
    }


def _description(name, run):
    # Use the current file, so that a corrected description applies to old runs too.
    if (EXPERIMENT_DIR / f"{name}.json").exists():
        return load_experiment(name).description
    return run["description"]


def score_run(run_path, run, suite, queries):
    """Score one run. The queries give the metadata for the slices."""
    by_query = {q["id"]: q for q in queries}
    per_query: dict[str, dict] = {}
    for qid, query in by_query.items():
        value = run["queries"].get(qid)
        # The grades apply only to the query text that the judges saw.
        if value is None or value["query_sha256"] != suite["queries"][qid]:
            per_query[qid] = {"status": "changed_query"}
            continue
        per_query[qid] = _score_query(
            value,
            suite["qrels"].get(qid, {}),
            suite["passage_isa"].get(qid, {}),
            query["architecture_scope"],
        )
    groups = {q["id"]: q["comparison_group"] for q in queries}
    quality = {}
    for language in ("en", "ru", "zh"):
        selected = [q for q in queries if q["language"] == language]
        ndcg = {
            q["id"]: per_query[q["id"]]["ndcg@10"]
            for q in selected
            if per_query[q["id"]]["status"] == "scored"
        }
        bounds = interval(ndcg, groups)
        quality[language] = _aggregate(per_query, selected) | {
            "ndcg@10_interval": {"low": bounds["low"], "high": bounds["high"]}
        }
    english = [q for q in queries if q["language"] == "en"]
    slices = {
        field: {
            value: _slice(per_query, [q for q in english if q[field] == value])
            for value in sorted({q[field] for q in english})
        }
        for field in SLICES
    }
    sources = sorted({ref["source_id"] for q in english for ref in q["source_refs"]})
    slices["source"] = {
        sid: _slice(
            per_query,
            [
                q
                for q in english
                if any(r["source_id"] == sid for r in q["source_refs"])
            ],
        )
        for sid in sources
    }
    isa = [
        per_query[q["id"]]["architecture"]
        for q in english
        if per_query[q["id"]]["status"] == "scored"
    ]
    seconds = {}
    for value in run["queries"].values():
        for mode, values in value["seconds"].items():
            seconds.setdefault(mode, []).extend(values)
    models = run["identity"]["models"]
    return {
        "name": run["name"],
        "description": _description(run["name"], run),
        "completed": run["completed"],
        "run": f"runs/{run_path.name}",
        "settings": run["identity"]["settings"],
        "models": {
            "context": models.get("context", {}).get("request_model"),
            "embedding": models.get("embedding", {}).get("request_model"),
            "vector_dimensions": models.get("embedding", {}).get("embedding_dim"),
        },
        "reference": {"revision": suite["sha256"], "status": suite["status"]},
        "protocol": PROTOCOL,
        "quality": quality,
        "slices": slices,
        "architecture": {
            "wrong_isa_results": _mean(a["wrong_architecture_rate"] for a in isa),
            "queries_with_wrong_isa": _mean(
                float(a["any_wrong_architecture"])
                for a in isa
                if a["any_wrong_architecture"] is not None
            ),
            "both_architectures": _mean(
                float(a["both_architectures"])
                for a in isa
                if a["both_architectures"] is not None
            ),
        },
        "latency": {
            "measurement": run["measurement"],
            "modes": {
                mode: {
                    "p50_ms": median(values) * 1000,
                    "p95_ms": float(np.quantile(values, 0.95)) * 1000,
                    "samples": len(values),
                }
                for mode, values in seconds.items()
            },
        },
        "resources": run["resources"],
        "output_characters_median": median(
            value["characters"] for value in run["queries"].values()
        ),
        "stable_results": all(value["stable"] for value in run["queries"].values()),
        "per_query": {
            qid: row["ndcg@10"] if row["status"] == "scored" else row["status"]
            for qid, row in per_query.items()
        },
    }


def _row(path, result):
    quality = result["quality"]
    warm = result["latency"]["modes"].get("warm", {})
    return {
        "name": result["name"],
        "date": result["completed"][:10],
        "description": result["description"],
        "ndcg@10": {
            "en": {"mean": quality["en"]["ndcg@10"]}
            | quality["en"]["ndcg@10_interval"],
            "ru": quality["ru"]["ndcg@10"],
            "zh": quality["zh"]["ndcg@10"],
        },
        "unjudged_queries": sum(len(row["unjudged"]) for row in quality.values()),
        "wrong_isa_results": result["architecture"]["wrong_isa_results"],
        "warm_p50_ms": warm.get("p50_ms"),
        "measurement": result["latency"]["measurement"],
        "server_rss_bytes": result["resources"]["server_rss_bytes"],
        "result": path,
    }


def _number(value, pattern):
    return "–" if value is None else pattern.format(value)


def experiments_markdown(report):
    lines = [
        "# Experiments",
        "",
        (
            "Generated by `uv run bench report`; do not edit. Reference: "
            f"{report['status']}, revision `{report['suite_revision'][:12]}`. "
            "Two model judges made the grades; no person checked them."
        ),
        "",
        (
            "| Date | Experiment | nDCG@10 EN (95% CI) | RU | ZH | Wrong ISA "
            "| Warm P50 | Model RSS | Unjudged | Description |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    direct = False
    for row in report["experiments"]:
        en = row["ndcg@10"]["en"]
        ci = (
            f" ({en['low']:.3f}–{en['high']:.3f})"
            if en["low"] is not None and en["high"] is not None
            else ""
        )
        speed = _number(row["warm_p50_ms"], "{:.0f} ms")
        if row["measurement"] == "direct_adapter" and row["warm_p50_ms"] is not None:
            speed += "*"
            direct = True
        rss = row["server_rss_bytes"]
        cells = [
            row["date"],
            f"[{row['name']}]({row['result']})",
            _number(en["mean"], "{:.3f}") + ci,
            _number(row["ndcg@10"]["ru"], "{:.3f}"),
            _number(row["ndcg@10"]["zh"], "{:.3f}"),
            _number(row["wrong_isa_results"], "{:.1%}"),
            speed,
            _number(None if rss is None else rss / 2**30, "{:.1f} GiB"),
            str(row["unjudged_queries"]),
            row["description"].replace("|", "\\|"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    if direct:
        lines += [
            "",
            (
                "\\* Direct adapter, not the production CLI. "
                "Do not compare these times with CLI times."
            ),
        ]
    return "\n".join(lines) + "\n"


def build_report(root):
    root = Path(root)
    if (root / "pool.json").exists():
        from benchmarks.build_suite import freeze

        freeze(root)
    if not (root / "suite.json").exists():
        return {
            "status": "preparation_incomplete",
            "next": "bench prepare",
        }
    pointer = read(root / "suite.json")
    suite = read(root / pointer["path"])
    if pointer["revision"] != suite["sha256"]:
        raise CorpusError("Suite revision differs; run bench prepare")
    # Metadata comes from the current banks; the hashes in the suite guard the grades.
    queries = [
        q.model_dump(mode="json")
        for q in load_evaluation_queries()
        if q.id in suite["queries"]
    ]
    rows, unscored = [], []
    for run_path in sorted((root / "runs").glob("*.json")):
        run = read(run_path)
        if run["identity"]["passage_snapshot"] != suite["passage_snapshot"]:
            unscored.append(
                {"run": f"runs/{run_path.name}", "reason": "other passage snapshot"}
            )
            continue
        result = score_run(run_path, run, suite, queries)
        path = (
            Path("results")
            / run["name"]
            / (f"{run['completed'][:10]}-{run_path.stem[:12]}.json")
        )
        write(root / path, result)
        rows.append(_row(str(path), result))
    rows.sort(key=lambda row: (row["date"], row["name"]))
    report = {
        "status": suite["status"],
        "suite_revision": suite["sha256"],
        "gaps": suite["gaps"],
        "pilot": suite["pilot"],
        "experiments": rows,
        "unscored_runs": unscored,
    }
    write(root / "report.json", report)
    _write_atomic(root / "EXPERIMENTS.md", experiments_markdown(report).encode())
    return report
