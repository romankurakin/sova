"""Hand-calculated graded metrics, ISA diagnostics and confidence intervals."""

import pytest

from benchmarks.evaluate import architecture_metrics, interval, passage_metrics


def ndcg(ids, qrels, k=10):
    return passage_metrics(ids, qrels, k)["metrics"]["ndcg"]


def test_fractional_grades_duplicates_unjudged_and_no_zero_reference():
    assert ndcg(["a"], {"a": 1.5}, k=1) == 1
    duplicated = passage_metrics(["a", "a"], {"a": 3.0, "b": 3.0}, k=2)
    assert duplicated["metrics"]["precision"] == 0.5
    assert passage_metrics(["b"], {"a": 3.0})["status"] == "unjudged"
    assert ndcg([], {"a": 3.0}) == 0
    assert passage_metrics(["a"], {"a": 0.0})["status"] == "reference_gap"
    assert passage_metrics(["a"], {"a": 2.5})["metrics"]["recall"] == 1
    assert passage_metrics(["a"], {"a": 1.0})["metrics"]["recall"] is None
    assert passage_metrics(["a"], {"a": 1.0})["metrics"]["hit"] == 0
    assert passage_metrics(["b", "a"], {"a": 2.0, "b": 0.0})["metrics"]["hit"] == 1


def test_rank_positions_follow_the_log_discount():
    qrels = {"a": 3.0, "b": 0.0}
    assert ndcg(["a", "b"], qrels) == 1
    assert ndcg(["b", "a"], qrels) == pytest.approx(1 / 1.584962500721156)
    assert passage_metrics(["b", "a"], qrels)["metrics"]["mrr"] == 0.5


def test_wrong_architecture_follows_agreed_tags_and_query_scope():
    tags = {"a": ["riscv"], "b": ["loongarch"], "c": ["other"], "d": None}
    single = architecture_metrics(["a", "b", "c"], {"a": 3.0, "b": 0.0}, tags, "riscv")
    assert single["wrong_architecture_rate"] == 1 / 3
    assert single["any_wrong_architecture"] is True
    both = architecture_metrics(["a", "b"], {"a": 3.0, "b": 2.0}, tags, "both")
    assert both["both_architectures"] is True
    assert both["wrong_architecture_rate"] is None
    unknown = architecture_metrics(["a", "d"], {"a": 3.0, "d": 3.0}, tags, "riscv")
    assert unknown["uncertain_results"] == 1
    assert unknown["wrong_architecture_rate"] is None


def test_cluster_bootstrap_preserves_query_weights():
    result = interval(
        {"a": 0.1, "b": 0.1, "c": -0.1}, {"a": "pair", "b": "pair", "c": "single"}
    )
    assert result["mean"] == pytest.approx(0.1 / 3)
    assert result["low"] == pytest.approx(-0.1)
    assert result["high"] == pytest.approx(0.1)
