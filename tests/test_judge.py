"""Tests for benchmarks.judge module."""

from unittest.mock import patch, MagicMock
import pytest

from benchmarks.judge import (
    Judgment,
    JudgeError,
    JudgeRateLimitError,
    QuerySpec,
    collect_query_subtopics,
    judge_chunk,
    judge_query,
    _is_permanent_error,
    _is_rate_limit,
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


class TestErrorDetection:
    def test_404_is_permanent(self):
        assert _is_permanent_error(
            Exception("model 'foo' not found (status code: 404)")
        )

    def test_timeout_is_not_permanent(self):
        assert not _is_permanent_error(Exception("connection timed out"))

    def test_429_is_rate_limit(self):
        assert _is_rate_limit(Exception("status code: 429"))

    def test_usage_limit_is_rate_limit(self):
        assert _is_rate_limit(
            Exception("you've reached your session usage limit, please wait")
        )

    def test_rate_limit_phrase(self):
        assert _is_rate_limit(Exception("rate limit exceeded"))

    def test_timeout_is_not_rate_limit(self):
        assert not _is_rate_limit(Exception("connection timed out"))

    def test_cloud_model_detection(self):
        assert _is_cloud_model("kimi-k2.5:cloud")
        assert _is_cloud_model("deepseek-r1:cloud")
        assert not _is_cloud_model("gemma3:27b")
        assert not _is_cloud_model("qwen3:30b")


class TestJudgeChunkErrorHandling:
    """judge_chunk must raise on errors, never return score 0 for failures."""

    def test_model_not_found_raises_judge_error(self):
        exc = Exception("model 'kimi-k2.5:cloud' not found (status code: 404)")
        with patch("benchmarks.judge._call_judge", side_effect=exc):
            with pytest.raises(JudgeError, match="not found"):
                judge_chunk("test query", "some chunk text")

    def test_rate_limit_raises_after_retries(self):
        exc = Exception("session usage limit (status code: 429)")
        with patch("benchmarks.judge._call_judge", side_effect=exc):
            with patch("benchmarks.judge.time.sleep"):
                with pytest.raises(JudgeRateLimitError):
                    judge_chunk("test query", "some text")

    def test_transient_error_raises_judge_error(self):
        exc = Exception("connection reset by peer")
        with patch("benchmarks.judge._call_judge", side_effect=exc):
            with patch("benchmarks.judge.time.sleep"):
                with pytest.raises(JudgeError, match="failed after"):
                    judge_chunk("test query", "some text", max_retries=1)

    def test_rate_limit_uses_backoff(self):
        exc = Exception("status code: 429")
        with patch("benchmarks.judge._call_judge", side_effect=exc):
            with patch("benchmarks.judge.time.sleep") as mock_sleep:
                with pytest.raises(JudgeRateLimitError):
                    judge_chunk("test query", "some text", max_retries=0)
                # Should have called sleep with exponential backoff
                assert mock_sleep.call_count >= 1


class TestJudgeQueryPropagatesErrors:
    """judge_query must propagate errors, not skip or swallow them."""

    def _make_spec(self):
        return QuerySpec("t01", "test query", "exact_lookup", [])

    def test_rate_limit_propagates(self):
        with patch(
            "benchmarks.judge.collect_pool",
            return_value=[
                {"chunk_id": 1, "doc": "d", "text": "chunk text", "section_id": None}
            ],
        ):
            with patch(
                "benchmarks.judge.judge_chunk",
                side_effect=JudgeRateLimitError("429"),
            ):
                with pytest.raises(JudgeRateLimitError):
                    judge_query(self._make_spec(), k_per_strategy=10)

    def test_judge_error_propagates(self):
        with patch(
            "benchmarks.judge.collect_pool",
            return_value=[
                {"chunk_id": 1, "doc": "d", "text": "chunk text", "section_id": None}
            ],
        ):
            with patch(
                "benchmarks.judge.judge_chunk",
                side_effect=JudgeError("model not found"),
            ):
                with pytest.raises(JudgeError):
                    judge_query(self._make_spec(), k_per_strategy=10)

    def test_on_chunk_judged_called_per_success(self):
        mock_response = (2, "relevant", 0.9, ["topic"])
        callback = MagicMock()

        with patch(
            "benchmarks.judge.collect_pool",
            return_value=[
                {"chunk_id": 1, "doc": "d", "text": "t1", "section_id": None},
                {"chunk_id": 2, "doc": "d", "text": "t2", "section_id": None},
            ],
        ):
            with patch("benchmarks.judge.judge_chunk", return_value=mock_response):
                qj = judge_query(
                    self._make_spec(),
                    k_per_strategy=10,
                    on_chunk_judged=callback,
                )
        assert callback.call_count == 2
        assert len(qj.judgments) == 2

    def test_callback_fires_before_error(self):
        """If chunk 1 succeeds and chunk 2 errors, callback fires once."""
        callback = MagicMock()

        with patch(
            "benchmarks.judge.collect_pool",
            return_value=[
                {"chunk_id": 1, "doc": "d", "text": "t1", "section_id": None},
                {"chunk_id": 2, "doc": "d", "text": "t2", "section_id": None},
            ],
        ):
            with patch(
                "benchmarks.judge.judge_chunk",
                side_effect=[
                    (2, "good", 0.9, ["topic"]),
                    JudgeRateLimitError("429"),
                ],
            ):
                with pytest.raises(JudgeRateLimitError):
                    judge_query(
                        self._make_spec(),
                        k_per_strategy=10,
                        on_chunk_judged=callback,
                    )
        # First chunk was judged and callback fired before error on second
        assert callback.call_count == 1


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
