"""Tests for score-decay diversity."""

from sova.diversity import score_decay_diversify


def _make_results(doc_scores: list[tuple[str, float]]) -> list[dict]:
    return [{"doc": doc, "final_score": score} for doc, score in doc_scores]


class TestScoreDecayDiversify:
    def test_empty(self):
        assert score_decay_diversify([], limit=5) == []

    def test_single_result(self):
        results = _make_results([("a", 10)])
        assert len(score_decay_diversify(results, limit=5)) == 1

    def test_limit_respected(self):
        results = _make_results([("a", 10), ("b", 9), ("c", 8), ("d", 7)])
        assert len(score_decay_diversify(results, limit=2)) == 2

    def test_no_decay_on_first_per_doc(self):
        """First chunk from each doc should keep original ordering."""
        results = _make_results([("a", 10), ("b", 9), ("c", 8)])
        filtered = score_decay_diversify(results, limit=3, decay=0.5)
        docs = [r["doc"] for r in filtered]
        assert docs == ["a", "b", "c"]

    def test_repeated_doc_gets_penalized(self):
        """Second chunk from same doc should drop below a close competitor."""
        results = _make_results([("a", 10), ("a", 9.5), ("b", 9)])
        filtered = score_decay_diversify(results, limit=3, decay=0.7)
        # a's 2nd chunk: 9.5 * 0.7 = 6.65, b's 1st: 9.0
        # Order should be: a(10), b(9), a(6.65)
        docs = [r["doc"] for r in filtered]
        assert docs == ["a", "b", "a"]

    def test_dominant_doc_survives(self):
        """If one doc is much higher scored, multiple chunks survive."""
        results = _make_results([("a", 20), ("a", 18), ("a", 16), ("b", 5)])
        filtered = score_decay_diversify(results, limit=4, decay=0.8)
        # a chunks: 20, 18*0.8=14.4, 16*0.64=10.24; b: 5
        # All a's above b, so a dominates
        docs = [r["doc"] for r in filtered]
        assert docs == ["a", "a", "a", "b"]

    def test_decay_zero_is_one_per_doc(self):
        """decay=0 means only first chunk per doc survives (score * 0^1 = 0)."""
        results = _make_results([("a", 10), ("a", 9), ("b", 8), ("b", 7)])
        filtered = score_decay_diversify(results, limit=4, decay=0.0)
        docs = [r["doc"] for r in filtered]
        # a(10), b(8), a(0), b(0)
        assert docs[:2] == ["a", "b"]

    def test_decay_one_is_no_diversity(self):
        """decay=1.0 means no penalty — pure relevance."""
        results = _make_results([("a", 10), ("a", 9), ("a", 8), ("b", 7)])
        filtered = score_decay_diversify(results, limit=4, decay=1.0)
        docs = [r["doc"] for r in filtered]
        assert docs == ["a", "a", "a", "b"]

    def test_uses_rerank_score_when_available(self):
        """Should prefer rerank_score over final_score."""
        results = [
            {"doc": "a", "final_score": 5, "rerank_score": 10},
            {"doc": "b", "final_score": 9, "rerank_score": 2},
        ]
        filtered = score_decay_diversify(results, limit=2, decay=0.8)
        assert filtered[0]["doc"] == "a"

    def test_single_doc_corpus(self):
        results = _make_results([("only", 10), ("only", 9), ("only", 8)])
        filtered = score_decay_diversify(results, limit=3, decay=0.8)
        assert len(filtered) == 3
        assert all(r["doc"] == "only" for r in filtered)
