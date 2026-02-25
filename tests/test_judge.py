"""Tests for benchmarks.judge module."""

import io
from email.message import Message
import urllib.error
from unittest.mock import patch, MagicMock
import pytest

import benchmarks.judge as judge_module
from benchmarks.judge import (
    Judgment,
    JudgeError,
    QuerySpec,
    collect_query_subtopics,
    judge_chunk,
    judge_query,
    _is_permanent_error,
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


class TestJudgeChunkErrorHandling:
    """judge_chunk must raise on errors, never return score 0 for failures."""

    def test_model_not_found_raises_judge_error(self):
        exc = Exception(
            "model 'ministral-3-14b-instruct-2512' not found (status code: 404)"
        )
        with patch("benchmarks.judge._call_judge", side_effect=exc):
            with pytest.raises(JudgeError, match="not found"):
                judge_chunk("test query", "some chunk text")

    def test_transient_error_raises_judge_error(self):
        exc = Exception("connection reset by peer")
        with patch("benchmarks.judge._call_judge", side_effect=exc):
            with patch("benchmarks.judge.time.sleep"):
                with pytest.raises(JudgeError, match="failed after"):
                    judge_chunk("test query", "some text", max_retries=1)


class TestJudgeQueryPropagatesErrors:
    """judge_query must propagate errors, not skip or swallow them."""

    def _make_spec(self):
        return QuerySpec("t01", "test query", "exact_lookup", [])

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
                    JudgeError("server error"),
                ],
            ):
                with pytest.raises(JudgeError):
                    judge_query(
                        self._make_spec(),
                        k_per_strategy=10,
                        on_chunk_judged=callback,
                    )
        # First chunk was judged and callback fired before error on second
        assert callback.call_count == 1


class TestJudgeRuntimeAndParsing:
    def test_call_judge_uses_llama_server(self, monkeypatch):
        calls: list[tuple[str, dict, float]] = []

        def _fake_post(url: str, payload: dict, timeout: float = 60.0):
            calls.append((url, payload, timeout))
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"score": 2, "confidence": 0.9, "subtopics": [], "reason": "ok"}'
                        }
                    }
                ]
            }

        monkeypatch.setattr(judge_module, "_post_json", _fake_post)
        result = judge_module._call_judge("test prompt")
        assert result.score == 2
        url, payload, timeout = calls[0]
        assert url.endswith("/v1/chat/completions")
        assert payload["model"] == judge_module.JUDGE_MODEL
        assert payload["messages"][0]["content"] == "test prompt"
        assert payload["response_format"] == judge_module._JUDGE_RESPONSE_FORMAT
        assert timeout == 60.0

    def test_should_use_debiasing_defaults_true(self, monkeypatch):
        monkeypatch.delenv("SOVA_BENCH_USE_DEBIASING", raising=False)
        assert judge_module.should_use_debiasing() is True

    def test_should_use_debiasing_respects_env(self, monkeypatch):
        monkeypatch.setenv("SOVA_BENCH_USE_DEBIASING", "false")
        assert judge_module.should_use_debiasing() is False
        monkeypatch.setenv("SOVA_BENCH_USE_DEBIASING", "true")
        assert judge_module.should_use_debiasing() is True

    def test_python_literal_response_is_accepted(self):
        result = judge_module._parse_judgment_response(
            "{'score': 1, 'confidence': 0.6, 'subtopics': ['trap'], 'reason': 'partial'}"
        )
        assert result.score == 1
        assert result.subtopics == ["trap"]

    def test_json_with_trailing_commas_and_comments_is_accepted(self):
        payload = """
        ```json
        {
          "score": 2,
          "confidence": 0.7,
          "subtopics": ["mtvec", "handler",],
          // inline comment
          "reason": "mostly relevant",
        }
        ```
        """
        result = judge_module._parse_judgment_response(payload)
        assert result.score == 2
        assert result.reason == "mostly relevant"
        assert result.subtopics == ["mtvec", "handler"]

    def test_regex_recovery_for_non_json_output(self):
        payload = """
        score: 1
        confidence: 0.42
        subtopics: [trap entry, save/restore]
        reason: partially relevant due to context only
        """
        result = judge_module._parse_judgment_response(payload)
        assert result.score == 1
        assert result.confidence == pytest.approx(0.42)
        assert "partially relevant" in result.reason
        assert result.subtopics == ["trap entry", "save/restore"]

    def test_judge_query_error_includes_query_and_chunk_context(self):
        spec = QuerySpec("t99", "failing query", "conceptual", [])
        with patch(
            "benchmarks.judge.collect_pool",
            return_value=[
                {"chunk_id": 77, "doc": "doc-a", "text": "t", "section_id": None}
            ],
        ):
            with patch(
                "benchmarks.judge.judge_chunk",
                side_effect=JudgeError("bad model output"),
            ):
                with pytest.raises(JudgeError, match=r"query t99 chunk 77"):
                    judge_query(spec, k_per_strategy=10)


class TestPostJsonErrors:
    def test_http_error_includes_response_body_detail(self, monkeypatch):
        err = urllib.error.HTTPError(
            url="http://localhost:11434/api/chat",
            code=404,
            msg="Not Found",
            hdrs=Message(),
            fp=io.BytesIO(b'{"error":"model \\"glm-5:cloud\\" not found"}'),
        )

        def _raise_http_error(*_args, **_kwargs):
            raise err

        monkeypatch.setattr("urllib.request.urlopen", _raise_http_error)

        with pytest.raises(RuntimeError) as exc:
            judge_module._post_json("http://localhost:11434/api/chat", {})
        text = str(exc.value)
        assert "HTTP Error 404: Not Found" in text
        assert "glm-5:cloud" in text


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
