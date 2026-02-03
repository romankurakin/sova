"""Tests for benchmarks.judge module."""

from benchmarks.judge import (
    Judgment,
    collect_query_subtopics,
    QUERY_SET,
)


class TestCollectQuerySubtopics:
    def test_empty_judgments(self):
        assert collect_query_subtopics([]) == []

    def test_ignores_low_scores(self):
        judgments = [
            Judgment(chunk_id=1, doc="d", score=0, reason="", subtopics=["a"]),
            Judgment(chunk_id=2, doc="d", score=1, reason="", subtopics=["b"]),
        ]
        assert collect_query_subtopics(judgments) == []

    def test_collects_from_relevant(self):
        judgments = [
            Judgment(chunk_id=1, doc="d", score=2, reason="", subtopics=["a", "b"]),
            Judgment(chunk_id=2, doc="d", score=3, reason="", subtopics=["b", "c"]),
        ]
        result = collect_query_subtopics(judgments)
        assert result == ["a", "b", "c"]  # sorted, deduplicated

    def test_empty_subtopics_ignored(self):
        judgments = [
            Judgment(chunk_id=1, doc="d", score=3, reason="", subtopics=[]),
        ]
        assert collect_query_subtopics(judgments) == []


class TestQuerySet:
    def test_all_ids_unique(self):
        ids = [q.id for q in QUERY_SET]
        assert len(ids) == len(set(ids))

    def test_has_all_categories(self):
        categories = {q.category for q in QUERY_SET}
        assert "exact_lookup" in categories
        assert "conceptual" in categories
        assert "cross_doc" in categories
        assert "natural" in categories
        assert "negative" in categories

    def test_negative_queries_have_no_subtopics(self):
        for q in QUERY_SET:
            if q.category == "negative":
                assert q.subtopics == [], f"{q.id} should have no subtopics"
