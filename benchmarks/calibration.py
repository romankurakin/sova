"""LLM-as-Judge calibration against human labels."""

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from sova.config import DATA_DIR

from .judge import QUERY_SET
from .search_interface import get_backend, close_backend

CALIBRATION_SIZE = 50
MIN_KAPPA = 0.6
MAX_BIAS = 0.3


@dataclass
class CalibrationResult:
    kappa: float
    bias: float
    adjustment: float
    agreement_rate: float
    n_pairs: int
    llm_mean: float
    human_mean: float


def cohen_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    """Cohen's kappa for inter-rater agreement."""
    if len(labels_a) != len(labels_b) or not labels_a:
        return 0.0

    n = len(labels_a)
    categories = sorted(set(labels_a) | set(labels_b))
    freq_a = {c: labels_a.count(c) for c in categories}
    freq_b = {c: labels_b.count(c) for c in categories}
    agreements = sum(1 for a, b in zip(labels_a, labels_b) if a == b)

    p_o = agreements / n
    p_e = sum(freq_a.get(c, 0) * freq_b.get(c, 0) for c in categories) / (n * n)

    if p_e == 1:
        return 1.0 if p_o == 1 else 0.0
    return (p_o - p_e) / (1 - p_e)


def generate_calibration_pairs(
    n: int = CALIBRATION_SIZE,
    output_path: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Generate query-chunk pairs for human calibration."""
    if output_path is None:
        output_path = DATA_DIR / "calibration_pairs.json"

    backend = get_backend()

    # Sample queries from each category
    queries_by_cat = {}
    for q in QUERY_SET:
        queries_by_cat.setdefault(q.category, []).append(q)

    selected_queries = []
    per_cat = max(1, n // len(queries_by_cat))
    for queries in queries_by_cat.values():
        selected_queries.extend(random.sample(queries, min(per_cat, len(queries))))

    pairs = []
    pairs_per_query = max(1, n // len(selected_queries))

    for qi, q in enumerate(selected_queries):
        if verbose:
            print(f"  [{qi + 1}/{len(selected_queries)}] {q.query[:40]}...")

        hits = backend.search(q.query, limit=20)

        if not hits:
            continue

        # Sample from different rank positions
        positions = (
            [0, len(hits) // 2, len(hits) - 1]
            if len(hits) >= 3
            else list(range(len(hits)))
        )

        for pos in positions[:pairs_per_query]:
            hit = hits[pos]
            pairs.append(
                {
                    "id": f"cal_{len(pairs):03d}",
                    "query_id": q.id,
                    "query": q.query,
                    "category": q.category,
                    "chunk_id": hit.chunk_id,
                    "doc": hit.doc,
                    "text": hit.text[:800],
                    "rank_position": pos,
                    "human_score": None,
                    "llm_score": None,
                    "llm_reason": "",
                }
            )
            if len(pairs) >= n:
                break
        if len(pairs) >= n:
            break

    close_backend()
    random.shuffle(pairs)

    result = {
        "version": "1.0",
        "created": time.strftime("%Y-%m-%d"),
        "instructions": "Score each chunk 0-3. 3=Highly Relevant, 2=Relevant, 1=Marginal, 0=Not Relevant",
        "pairs": pairs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    return result


def validate_calibration(
    calibration_path: Path,
    output_path: Path | None = None,
    verbose: bool = True,
) -> CalibrationResult:
    """Validate LLM judgments against human labels."""
    if output_path is None:
        output_path = DATA_DIR / "calibration_results.json"

    data = json.loads(calibration_path.read_text())

    human_scores = []
    llm_scores = []

    for pair in data["pairs"]:
        if pair.get("human_score") is not None and pair.get("llm_score") is not None:
            human_scores.append(pair["human_score"])
            llm_scores.append(pair["llm_score"])

    if not human_scores:
        raise ValueError("No pairs with both human and LLM scores found")

    n = len(human_scores)
    kappa = cohen_kappa(human_scores, llm_scores)
    human_mean = sum(human_scores) / n
    llm_mean = sum(llm_scores) / n
    bias = llm_mean - human_mean
    agreements = sum(1 for h, llm in zip(human_scores, llm_scores) if h == llm)
    agreement_rate = agreements / n
    adjustment = -bias if abs(bias) > MAX_BIAS else 0

    result = CalibrationResult(
        kappa=kappa,
        bias=bias,
        adjustment=adjustment,
        agreement_rate=agreement_rate,
        n_pairs=n,
        llm_mean=llm_mean,
        human_mean=human_mean,
    )

    output_data = {
        "version": "1.0",
        "created": time.strftime("%Y-%m-%d"),
        "n_pairs": n,
        "metrics": {
            "cohen_kappa": round(kappa, 3),
            "bias": round(bias, 3),
            "exact_agreement": round(agreement_rate, 3),
        },
        "status": "PASS" if kappa >= MIN_KAPPA else "FAIL",
    }

    output_path.write_text(json.dumps(output_data, indent=2))
    return result
