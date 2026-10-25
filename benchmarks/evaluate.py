"""Calculate passage metrics, wrong-ISA results and confidence intervals."""

import math
from collections import defaultdict

import numpy as np

PROTOCOL: dict = {
    "name": "passage-grades-v2",
    "cutoffs": [1, 3, 5, 10],
    "bootstrap_seed": 260907,
    "bootstrap_samples": 10000,
}


def passage_metrics(ids: list[str], qrels: dict[str, float], k: int = 10) -> dict:
    if any(pid not in qrels for pid in ids[:k]):
        return {"status": "unjudged", "metrics": None}
    seen = set()
    dcg, first, useful = 0.0, 0.0, 0
    for rank, pid in enumerate(ids[:k], 1):
        if pid in seen:
            continue
        seen.add(pid)
        grade = qrels[pid]
        dcg += (2**grade - 1) / math.log2(rank + 1)
        if grade >= 2:
            useful += 1
            if not first:
                first = 1 / rank
    ideal = sum(
        (2**grade - 1) / math.log2(i + 2)
        for i, grade in enumerate(sorted(qrels.values(), reverse=True)[:k])
    )
    if not ideal:
        return {"status": "reference_gap", "metrics": None}
    relevant = sum(grade >= 2 for grade in qrels.values())
    return {
        "status": "scored",
        "metrics": {
            "ndcg": dcg / ideal,
            "mrr": first,
            "precision": useful / k,
            "recall": useful / relevant if relevant else None,
            "hit": float(useful > 0),
        },
    }


def interval(scores: dict[str, float], groups: dict[str, str]) -> dict:
    """Return the mean and its 95% cluster-bootstrap interval for one run."""
    if not scores:
        return {"mean": None, "low": None, "high": None, "queries": 0, "groups": 0}
    clustered = defaultdict(list)
    for qid, value in scores.items():
        clustered[groups[qid]].append(value)
    values = [clustered[key] for key in sorted(clustered)]
    sums = np.array([sum(v) for v in values])
    sizes = np.array([len(v) for v in values])
    rng = np.random.default_rng(PROTOCOL["bootstrap_seed"])
    samples = rng.integers(
        0, len(values), size=(PROTOCOL["bootstrap_samples"], len(values))
    )
    means = sums[samples].sum(axis=1) / sizes[samples].sum(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "mean": sum(scores.values()) / len(scores),
        "low": float(lo),
        "high": float(hi),
        "queries": len(scores),
        "groups": len(values),
    }


ISAS = {"riscv", "loongarch", "x86"}


def architecture_metrics(ids, qrels, passage_tags, scope):
    """Use the architecture tags that both judges agree on, not the source file name.

    For a query about one ISA, a result is wrong when it is only about other ISAs.
    """
    known, wrong, uncertain, useful_tags = 0, 0, 0, set()
    for pid in ids[:10]:
        tags = passage_tags.get(pid)
        if tags is None:
            uncertain += 1
            continue
        known += 1
        covered = set(tags) & ISAS
        wrong += bool(covered) and scope not in covered
        if qrels.get(pid, 0) >= 2:
            useful_tags.update(tags)
    single = scope in ISAS
    return {
        "both_architectures": {"riscv", "loongarch"} <= useful_tags
        if scope == "both" and not uncertain
        else None,
        "wrong_architecture_numerator": wrong,
        "known_results": known,
        "returned_results": min(10, len(ids)),
        "uncertain_results": uncertain,
        "wrong_architecture_rate": wrong / known
        if known and not uncertain and single
        else None,
        "any_wrong_architecture": bool(wrong) if not uncertain and single else None,
    }
