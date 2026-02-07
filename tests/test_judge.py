"""Tests for benchmarks.judge module."""

from unittest.mock import patch
import pytest

from benchmarks.judge import (
    Judgment,
    JudgeError,
    collect_query_subtopics,
    judge_chunk,
    _is_permanent_error,
    _is_cloud_model,
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


class TestJudgeFailFast:
    """Judge must raise immediately on permanent errors, not accumulate garbage."""

    def test_model_not_found_raises_judge_error(self):
        exc = Exception("model 'kimi-k2.5:cloud' not found (status code: 404)")
        with patch("benchmarks.judge.ollama.chat", side_effect=exc):
            with pytest.raises(JudgeError, match="not found"):
                judge_chunk("test query", "some chunk text")

    def test_404_is_permanent(self):
        assert _is_permanent_error(Exception("model 'foo' not found (status code: 404)"))

    def test_timeout_is_not_permanent(self):
        assert not _is_permanent_error(Exception("connection timed out"))

    def test_transient_error_retries_then_returns_error(self):
        exc = Exception("connection reset by peer")
        with patch("benchmarks.judge.ollama.chat", side_effect=exc):
            with patch("benchmarks.judge.time.sleep"):
                score, reason, conf, subs = judge_chunk(
                    "test query", "some text", max_retries=1
                )
        assert score == 0
        assert reason.startswith("error:")
        assert conf == 0.0

    def test_cloud_model_detection(self):
        assert _is_cloud_model("kimi-k2.5:cloud")
        assert _is_cloud_model("deepseek-r1:cloud")
        assert not _is_cloud_model("gemma3:27b")
        assert not _is_cloud_model("qwen3:30b")


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
