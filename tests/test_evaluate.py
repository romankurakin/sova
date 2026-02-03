"""Tests for benchmarks.evaluate module."""

import math

import pytest

from benchmarks.evaluate import (
    aggregate_by_category,
    aggregate_metrics,
    alpha_ndcg_at_k,
    average_precision,
    compute_diversity_metrics,
    compute_metrics,
    dcg_at_k,
    doc_coverage_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    subtopic_recall_at_k,
    Metrics,
    QueryResult,
)


class TestReciprocalRank:
    def test_first_result_relevant(self):
        assert reciprocal_rank([1, 2, 3], {1}, 10) == 1.0

    def test_second_result_relevant(self):
        assert reciprocal_rank([1, 2, 3], {2}, 10) == 0.5

    def test_no_relevant(self):
        assert reciprocal_rank([1, 2, 3], {99}, 10) == 0.0

    def test_k_cutoff(self):
        # Relevant doc at position 3, but k=2
        assert reciprocal_rank([1, 2, 3], {3}, 2) == 0.0

    def test_empty_results(self):
        assert reciprocal_rank([], {1}, 10) == 0.0


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k([1, 2, 3], {1, 2, 3}, 3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k([1, 2, 3], {99}, 3) == 0.0

    def test_half_relevant(self):
        assert precision_at_k([1, 2, 3, 4], {1, 3}, 4) == 0.5

    def test_k_zero(self):
        assert precision_at_k([1, 2], {1}, 0) == 0.0

    def test_k_smaller_than_results(self):
        assert precision_at_k([1, 2, 3], {1, 2, 3}, 2) == 1.0


class TestRecallAtK:
    def test_all_found(self):
        assert recall_at_k([1, 2, 3], {1, 2}, 3) == 1.0

    def test_partial_found(self):
        assert recall_at_k([1, 2, 3], {1, 2, 99}, 3) == pytest.approx(2 / 3)

    def test_none_found(self):
        assert recall_at_k([1, 2, 3], {99}, 3) == 0.0

    def test_empty_relevant(self):
        assert recall_at_k([1, 2], set(), 2) == 0.0


class TestHitRateAtK:
    def test_hit(self):
        assert hit_rate_at_k([1, 2, 3], {2}, 3) == 1.0

    def test_miss(self):
        assert hit_rate_at_k([1, 2, 3], {99}, 3) == 0.0

    def test_k_cutoff_miss(self):
        assert hit_rate_at_k([1, 2, 3], {3}, 2) == 0.0


class TestAveragePrecision:
    def test_perfect_ranking(self):
        # All relevant at top
        assert average_precision([1, 2, 3, 4], {1, 2}, 4) == 1.0

    def test_no_relevant(self):
        assert average_precision([1, 2, 3], set(), 3) == 0.0

    def test_one_relevant_at_position_2(self):
        # AP = (1/2) / min(1, 3) = 0.5
        assert average_precision([10, 1, 20], {1}, 3) == 0.5

    def test_empty_results(self):
        assert average_precision([], {1}, 10) == 0.0


class TestDCGAndNDCG:
    def test_dcg_single_result(self):
        # DCG = (2^3 - 1) / log2(2) = 7.0
        assert dcg_at_k([1], {1: 3}, 1) == pytest.approx(7.0)

    def test_dcg_two_results(self):
        # First: (2^3-1)/log2(2) = 7.0, Second: (2^1-1)/log2(3) = 0.63
        dcg = dcg_at_k([1, 2], {1: 3, 2: 1}, 2)
        assert dcg == pytest.approx(7.0 + 1.0 / math.log2(3))

    def test_ndcg_perfect(self):
        # Already in ideal order
        assert ndcg_at_k([1, 2], {1: 3, 2: 1}, 2) == pytest.approx(1.0)

    def test_ndcg_reversed(self):
        # Worst order — should be < 1
        score = ndcg_at_k([2, 1], {1: 3, 2: 1}, 2)
        assert 0 < score < 1.0

    def test_ndcg_no_relevant(self):
        assert ndcg_at_k([1, 2], {}, 2) == 0.0

    def test_ndcg_k_cutoff(self):
        score = ndcg_at_k([1, 2, 3], {3: 3}, 1)
        # Relevant doc at position 3 but k=1
        assert score == 0.0


class TestSubtopicRecall:
    def test_full_coverage(self):
        results = [{"chunk_id": 1}, {"chunk_id": 2}]
        subtopic_map = {1: ["a", "b"], 2: ["c"]}
        assert subtopic_recall_at_k(results, subtopic_map, 2) == 1.0

    def test_partial_coverage(self):
        results = [{"chunk_id": 1}]
        subtopic_map = {1: ["a"], 2: ["b"]}
        assert subtopic_recall_at_k(results, subtopic_map, 1) == 0.5

    def test_no_subtopics(self):
        assert subtopic_recall_at_k([{"chunk_id": 1}], {}, 1) == 0.0

    def test_k_cutoff(self):
        results = [{"chunk_id": 1}, {"chunk_id": 2}]
        subtopic_map = {2: ["a"]}
        # k=1, only first result checked, which has no subtopics
        assert subtopic_recall_at_k(results, subtopic_map, 1) == 0.0


class TestAlphaNDCG:
    def test_no_relevance(self):
        results = [{"chunk_id": 1}]
        assert alpha_ndcg_at_k(results, {}, {}, 1) == 0.0

    def test_with_subtopics(self):
        results = [{"chunk_id": 1}, {"chunk_id": 2}]
        relevance = {1: 3, 2: 3}
        subtopic_map = {1: ["a"], 2: ["b"]}
        score = alpha_ndcg_at_k(results, relevance, subtopic_map, 2)
        assert score > 0

    def test_redundant_subtopics_penalized(self):
        results = [{"chunk_id": 1}, {"chunk_id": 2}]
        relevance = {1: 3, 2: 3}
        # Same subtopic repeated — should be penalized
        same_subs = {1: ["a"], 2: ["a"]}
        diff_subs = {1: ["a"], 2: ["b"]}
        score_same = alpha_ndcg_at_k(results, relevance, same_subs, 2)
        score_diff = alpha_ndcg_at_k(results, relevance, diff_subs, 2)
        assert score_diff >= score_same


class TestDocCoverage:
    def test_all_different_docs(self):
        results = [{"doc": "a"}, {"doc": "b"}, {"doc": "c"}]
        assert doc_coverage_at_k(results, 3) == 1.0

    def test_all_same_doc(self):
        results = [{"doc": "a"}, {"doc": "a"}, {"doc": "a"}]
        assert doc_coverage_at_k(results, 3) == pytest.approx(1 / 3)

    def test_k_zero(self):
        assert doc_coverage_at_k([], 0) == 0.0


class TestComputeMetrics:
    def test_returns_all_metric_types(self):
        results = [1, 2, 3]
        relevance = {1: 3, 2: 1, 3: 0}
        m = compute_metrics(results, relevance, k_values=[3])
        assert 3 in m.mrr
        assert 3 in m.ndcg
        assert 3 in m.precision
        assert 3 in m.recall
        assert 3 in m.map
        assert 3 in m.hit_rate

    def test_default_k_values(self):
        m = compute_metrics([1], {1: 3})
        # Should use STANDARD_K = [1, 3, 5, 10]
        assert 1 in m.mrr
        assert 10 in m.mrr

    def test_threshold(self):
        results = [1, 2]
        relevance = {1: 1, 2: 3}
        # Default threshold=2, so only doc 2 is relevant
        m = compute_metrics(results, relevance, k_values=[2])
        # Doc 2 is at position 2, so precision = 1/2
        assert m.precision[2] == 0.5


class TestComputeDiversityMetrics:
    def test_returns_diversity_fields(self):
        results = [{"chunk_id": 1, "doc": "a"}]
        m = compute_diversity_metrics(results, {1: 3}, {1: ["x"]}, k_values=[1])
        assert 1 in m.subtopic_recall
        assert 1 in m.alpha_ndcg
        assert 1 in m.doc_coverage


class TestAggregateMetrics:
    def test_empty_results(self):
        assert aggregate_metrics([]) == {}

    def test_single_result(self):
        m = Metrics()
        m.mrr = {10: 1.0}
        m.ndcg = {10: 0.5}
        m.map = {10: 0.5}
        m.precision = {10: 0.5}
        m.recall = {10: 1.0}
        m.hit_rate = {10: 1.0}
        m.subtopic_recall = {10: 0.8}
        m.alpha_ndcg = {10: 0.6}
        m.doc_coverage = {10: 0.5}
        qr = QueryResult("q1", "test", "exact_lookup", m)
        agg = aggregate_metrics([qr], k_values=[10])
        assert agg["mrr"][10] == 1.0
        assert agg["ndcg"][10] == 0.5

    def test_averages_two_results(self):
        m1 = Metrics()
        m1.mrr = {10: 1.0}
        m1.ndcg = m1.map = m1.precision = m1.recall = m1.hit_rate = {10: 0.0}
        m1.subtopic_recall = m1.alpha_ndcg = m1.doc_coverage = {10: 0.0}
        m2 = Metrics()
        m2.mrr = {10: 0.5}
        m2.ndcg = m2.map = m2.precision = m2.recall = m2.hit_rate = {10: 0.0}
        m2.subtopic_recall = m2.alpha_ndcg = m2.doc_coverage = {10: 0.0}
        qrs = [
            QueryResult("q1", "a", "cat", m1),
            QueryResult("q2", "b", "cat", m2),
        ]
        agg = aggregate_metrics(qrs, k_values=[10])
        assert agg["mrr"][10] == pytest.approx(0.75)


class TestAggregateByCategory:
    def test_groups_by_category(self):
        m1 = Metrics()
        m1.mrr = m1.ndcg = m1.map = m1.precision = m1.recall = {10: 1.0}
        m1.subtopic_recall = m1.doc_coverage = {10: 0.5}
        m2 = Metrics()
        m2.mrr = m2.ndcg = m2.map = m2.precision = m2.recall = {10: 0.5}
        m2.subtopic_recall = m2.doc_coverage = {10: 0.3}
        qrs = [
            QueryResult("q1", "a", "exact", m1),
            QueryResult("q2", "b", "conceptual", m2),
        ]
        by_cat = aggregate_by_category(qrs, k=10)
        assert "exact" in by_cat
        assert "conceptual" in by_cat
        assert by_cat["exact"]["mrr"] == 1.0
        assert by_cat["conceptual"]["mrr"] == 0.5
