"""Industry-standard IR evaluation metrics.

Implements metrics from TREC, BEIR, and MTEB benchmarks:
- MRR@k (Mean Reciprocal Rank)
- nDCG@k (Normalized Discounted Cumulative Gain)
- MAP@k (Mean Average Precision)
- Precision@k, Recall@k
- Success@k
- Subtopic Recall, α-nDCG, Doc Coverage (diversity)
"""

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path


# Standard cutoffs used in BEIR/MTEB
STANDARD_K = [1, 3, 5, 10]


@dataclass
class Metrics:
    """All metrics for a single query at multiple k values."""

    mrr: dict[int, float] = field(default_factory=dict)
    ndcg: dict[int, float] = field(default_factory=dict)
    map: dict[int, float] = field(default_factory=dict)
    precision: dict[int, float] = field(default_factory=dict)
    recall: dict[int, float] = field(default_factory=dict)
    hit_rate: dict[int, float] = field(default_factory=dict)

    # Diversity metrics
    subtopic_recall: dict[int, float] = field(default_factory=dict)
    alpha_ndcg: dict[int, float] = field(default_factory=dict)
    doc_coverage: dict[int, float] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Evaluation result for a single query."""

    query_id: str
    query: str
    category: str
    metrics: Metrics


def reciprocal_rank(results: list[int], relevant: set[int], k: int) -> float:
    """Reciprocal Rank - 1/position of first relevant result."""
    for i, doc_id in enumerate(results[:k], 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def precision_at_k(results: list[int], relevant: set[int], k: int) -> float:
    """Precision@k - fraction of top-k that are relevant."""
    if k == 0:
        return 0.0
    return sum(1 for d in results[:k] if d in relevant) / k


def recall_at_k(results: list[int], relevant: set[int], k: int) -> float:
    """Recall@k - fraction of relevant docs in top-k."""
    if not relevant:
        return 0.0
    return sum(1 for d in results[:k] if d in relevant) / len(relevant)


def hit_rate_at_k(results: list[int], relevant: set[int], k: int) -> float:
    """Hit Rate@k (Success@k) - 1 if any relevant in top-k, else 0."""
    return 1.0 if any(d in relevant for d in results[:k]) else 0.0


def average_precision(results: list[int], relevant: set[int], k: int) -> float:
    """Average Precision at k - precision at each relevant hit, normalized by k.

    Uses min(len(relevant), k) normalization for truncated evaluation.
    """
    if not relevant:
        return 0.0

    hits = 0
    sum_precisions = 0.0

    for i, doc_id in enumerate(results[:k], 1):
        if doc_id in relevant:
            hits += 1
            sum_precisions += hits / i

    return sum_precisions / min(len(relevant), k)


def dcg_at_k(results: list[int], relevance: dict[int, int], k: int) -> float:
    """Discounted Cumulative Gain with graded relevance."""
    return sum(
        (2 ** relevance.get(doc_id, 0) - 1) / math.log2(i + 2)
        for i, doc_id in enumerate(results[:k])
    )


def ndcg_at_k(results: list[int], relevance: dict[int, int], k: int) -> float:
    """Normalized DCG - DCG / ideal DCG.

    Primary metric for graded relevance (BEIR, MTEB).
    """
    dcg = dcg_at_k(results, relevance, k)
    ideal_order = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_order))
    return dcg / idcg if idcg > 0 else 0.0


def subtopic_recall_at_k(
    results: list[dict], subtopic_map: dict[int, list[str]], k: int
) -> float:
    """Subtopic Recall - coverage of query aspects.

    Primary diversity metric from TREC Web Track.
    """
    all_subtopics = set()
    for subs in subtopic_map.values():
        all_subtopics.update(subs)

    if not all_subtopics:
        return 0.0

    covered: set[str] = set()
    for r in results[:k]:
        doc_id = r.get("chunk_id") or r.get("id")
        if doc_id is not None:
            covered.update(subtopic_map.get(doc_id, []))

    return len(covered) / len(all_subtopics)


def alpha_ndcg_at_k(
    results: list[dict],
    relevance: dict[int, int],
    subtopic_map: dict[int, list[str]],
    k: int,
    alpha: float = 0.5,
) -> float:
    """α-nDCG - nDCG with redundancy penalty.

    Penalizes seeing the same subtopic multiple times.
    From Clarke et al. "Novelty and Diversity" SIGIR 2008.
    """
    seen_subtopics: dict[str, int] = {}
    gain = 0.0

    for i, r in enumerate(results[:k]):
        doc_id = r.get("chunk_id") or r.get("id")
        if doc_id is None:
            continue
        rel = relevance.get(doc_id, 0)
        subtopics = subtopic_map.get(doc_id, [])

        # Novelty-weighted gain
        doc_gain = 0.0
        for st in subtopics:
            count = seen_subtopics.get(st, 0)
            doc_gain += rel * ((1 - alpha) ** count)
            seen_subtopics[st] = count + 1

        if not subtopics:
            doc_gain = rel

        gain += doc_gain / math.log2(i + 2)

    # Compute ideal (greedy selection for max α-nDCG)
    # Simplified: use standard nDCG as upper bound
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal))

    return gain / idcg if idcg > 0 else 0.0


def doc_coverage_at_k(results: list[dict], k: int) -> float:
    """Document Coverage - unique source documents in top-k."""
    docs = [r.get("doc") for r in results[:k] if r.get("doc")]
    unique = len(set(docs))
    return unique / k if k > 0 else 0.0


def compute_metrics(
    results: list[int],
    relevance: dict[int, int],
    k_values: list[int] | None = None,
    threshold: int = 2,
) -> Metrics:
    """Compute all metrics at multiple k cutoffs.

    Args:
        results: Ranked list of document/chunk IDs
        relevance: {doc_id: relevance_score (0-3)}
        k_values: Cutoffs to evaluate (default: STANDARD_K)
        threshold: Score >= threshold counts as binary relevant
    """
    if k_values is None:
        k_values = STANDARD_K

    relevant = {d for d, s in relevance.items() if s >= threshold}
    m = Metrics()

    for k in k_values:
        m.mrr[k] = reciprocal_rank(results, relevant, k)
        m.ndcg[k] = ndcg_at_k(results, relevance, k)
        m.map[k] = average_precision(results, relevant, k)
        m.precision[k] = precision_at_k(results, relevant, k)
        m.recall[k] = recall_at_k(results, relevant, k)
        m.hit_rate[k] = hit_rate_at_k(results, relevant, k)

    return m


def compute_diversity_metrics(
    results: list[dict],
    relevance: dict[int, int],
    subtopic_map: dict[int, list[str]],
    k_values: list[int] | None = None,
) -> Metrics:
    """Compute diversity metrics at multiple k cutoffs."""
    if k_values is None:
        k_values = STANDARD_K

    m = Metrics()
    for k in k_values:
        m.subtopic_recall[k] = subtopic_recall_at_k(results, subtopic_map, k)
        m.alpha_ndcg[k] = alpha_ndcg_at_k(results, relevance, subtopic_map, k)
        m.doc_coverage[k] = doc_coverage_at_k(results, k)

    return m


def aggregate_by_category(
    results: list[QueryResult], k: int = 10
) -> dict[str, dict[str, float]]:
    """Aggregate metrics by query category.

    Returns dict of {category: {metric_name: value}}.
    """
    by_cat: dict[str, list[QueryResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    agg = {}
    for cat, cat_results in by_cat.items():
        n = len(cat_results)
        agg[cat] = {
            "mrr": sum(r.metrics.mrr.get(k, 0) for r in cat_results) / n,
            "ndcg": sum(r.metrics.ndcg.get(k, 0) for r in cat_results) / n,
            "map": sum(r.metrics.map.get(k, 0) for r in cat_results) / n,
            "precision": sum(r.metrics.precision.get(k, 0) for r in cat_results) / n,
            "recall": sum(r.metrics.recall.get(k, 0) for r in cat_results) / n,
            "subtopic_recall": sum(
                r.metrics.subtopic_recall.get(k, 0) for r in cat_results
            )
            / n,
            "doc_coverage": sum(r.metrics.doc_coverage.get(k, 0) for r in cat_results)
            / n,
            "count": n,
        }

    return agg


def aggregate_metrics(
    results: list[QueryResult], k_values: list[int] | None = None
) -> dict[str, dict[int, float]]:
    """Aggregate metrics across queries (mean)."""
    if not results:
        return {}
    if k_values is None:
        k_values = STANDARD_K

    n = len(results)
    agg = {
        "mrr": {},
        "ndcg": {},
        "map": {},
        "precision": {},
        "recall": {},
        "hit_rate": {},
        "subtopic_recall": {},
        "alpha_ndcg": {},
        "doc_coverage": {},
    }

    for k in k_values:
        agg["mrr"][k] = sum(r.metrics.mrr.get(k, 0) for r in results) / n
        agg["ndcg"][k] = sum(r.metrics.ndcg.get(k, 0) for r in results) / n
        agg["map"][k] = sum(r.metrics.map.get(k, 0) for r in results) / n
        agg["precision"][k] = sum(r.metrics.precision.get(k, 0) for r in results) / n
        agg["recall"][k] = sum(r.metrics.recall.get(k, 0) for r in results) / n
        agg["hit_rate"][k] = sum(r.metrics.hit_rate.get(k, 0) for r in results) / n
        agg["subtopic_recall"][k] = (
            sum(r.metrics.subtopic_recall.get(k, 0) for r in results) / n
        )
        agg["alpha_ndcg"][k] = sum(r.metrics.alpha_ndcg.get(k, 0) for r in results) / n
        agg["doc_coverage"][k] = (
            sum(r.metrics.doc_coverage.get(k, 0) for r in results) / n
        )

    return agg


def bootstrap_confidence_interval(
    values: list[float], n_bootstrap: int = 1000, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Bootstrap 95% confidence interval.

    Returns (mean, lower, upper).
    Standard for reporting IR metrics with uncertainty.
    """
    if not values:
        return 0.0, 0.0, 0.0

    n = len(values)
    means = []

    for _ in range(n_bootstrap):
        sample = [values[random.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = 1 - confidence
    lower_idx = int(n_bootstrap * alpha / 2)
    upper_idx = int(n_bootstrap * (1 - alpha / 2))

    return sum(values) / n, means[lower_idx], means[upper_idx]


def paired_t_test(a: list[float], b: list[float]) -> tuple[float, bool]:
    """Paired t-test for statistical significance."""
    if len(a) != len(b) or len(a) < 2:
        return 1.0, False

    n = len(a)
    diffs = [a[i] - b[i] for i in range(n)]
    mean_diff = sum(diffs) / n
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)

    if var_diff == 0:
        return 1.0 if mean_diff == 0 else 0.0, mean_diff != 0

    se = math.sqrt(var_diff / n)
    t_stat = mean_diff / se

    # Approximate two-tailed p-value
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))
    return p, p < 0.05


def wilcoxon_signed_rank(a: list[float], b: list[float]) -> tuple[float, bool]:
    """Wilcoxon signed-rank test (non-parametric).

    Preferred over t-test for IR evaluation (non-normal distributions).
    """
    if len(a) != len(b) or len(a) < 5:
        return 1.0, False

    diffs = [(a[i] - b[i], i) for i in range(len(a)) if a[i] != b[i]]
    if not diffs:
        return 1.0, False

    # Rank by absolute difference
    ranked = sorted(diffs, key=lambda x: abs(x[0]))
    ranks = {}
    for rank, (diff, idx) in enumerate(ranked, 1):
        ranks[idx] = rank

    # Sum of positive and negative ranks
    w_plus = sum(ranks[idx] for diff, idx in diffs if diff > 0)
    w_minus = sum(ranks[idx] for diff, idx in diffs if diff < 0)
    w = min(w_plus, w_minus)

    # Approximate p-value (normal approximation for n > 10)
    n = len(diffs)
    if n < 10:
        return 0.5, False  # Too few samples

    mean_w = n * (n + 1) / 4
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w - mean_w) / std_w
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

    return p, p < 0.05


def format_beir_results(
    name: str, metrics: dict[str, dict[int, float]], k_values: list[int] | None = None
) -> dict:
    """Format results in BEIR leaderboard style."""
    if k_values is None:
        k_values = [1, 3, 5, 10]

    metrics_dict: dict[str, float] = {}
    result: dict[str, str | dict[str, float]] = {"model": name, "metrics": metrics_dict}

    for metric_name, values in metrics.items():
        for k in k_values:
            if k in values:
                key = f"{metric_name}@{k}"
                metrics_dict[key] = round(values[k], 4)

    return result


def format_table(
    configs: dict[str, dict], k: int = 10, metrics: list[str] | None = None
) -> str:
    """Format comparison table in markdown."""
    if metrics is None:
        metrics = ["ndcg", "map", "mrr", "precision", "recall", "hit_rate"]
    header = "| Model |"
    for m in metrics:
        header += f" {m.upper()}@{k} |"
    lines = [header]
    sep = "|-------|"
    for _ in metrics:
        sep += "--------|"
    lines.append(sep)
    for name, data in configs.items():
        row = f"| {name:<20} |"
        for m in metrics:
            val = data.get(m, {}).get(k, 0)
            row += f" {val:.3f}  |"
        lines.append(row)

    return "\n".join(lines)


def format_metrics_table(
    configs: dict[str, dict], k: int = 10, include_diversity: bool = True
) -> str:
    """Format metrics comparison table with relevance and diversity metrics.

    Args:
        configs: {config_name: {metric_name: {k: value}}}
        k: Cutoff for metrics
        include_diversity: Include diversity metrics in output
    """
    rel_metrics = ["mrr", "ndcg", "precision", "recall"]
    div_metrics = ["subtopic_recall", "doc_coverage"] if include_diversity else []
    all_metrics = rel_metrics + div_metrics

    header = "| Config |"
    for m in all_metrics:
        label = m.replace("_", " ").title()
        header += f" {label}@{k} |"
    if "avg_ms" in next(iter(configs.values()), {}):
        header += " Avg ms |"
    lines = [header]
    sep = "|" + "-" * 20 + "|"
    for _ in all_metrics:
        sep += "-" * 12 + "|"
    if "avg_ms" in next(iter(configs.values()), {}):
        sep += "-" * 9 + "|"
    lines.append(sep)
    for name, data in configs.items():
        row = f"| {name:<18} |"
        for m in all_metrics:
            if isinstance(data.get(m), dict):
                val = data.get(m, {}).get(k, 0)
            else:
                val = data.get(m, 0)
            row += f" {val:>10.3f} |"
        if "avg_ms" in data:
            row += f" {data['avg_ms']:>7.1f} |"
        lines.append(row)

    return "\n".join(lines)


def save_results(results: dict, path: Path) -> None:
    """Save results in JSON format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))


def load_ground_truth(path: Path) -> dict:
    """Load ground truth from JSON."""
    return json.loads(path.read_text())


