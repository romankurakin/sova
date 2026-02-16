"""Tests for llama_client module."""

import json
from unittest.mock import MagicMock, patch


def _mock_urlopen(response_body: dict, status: int = 200):
    """Create a mock for urllib.request.urlopen that returns JSON."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(response_body).encode()
    mock_resp.status = status
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _mock_ensure_server_for(*down_ports: str):
    """Return a side_effect for _ensure_server that fails for given ports."""

    def side_effect(url, timeout=120.0):
        for port in down_ports:
            if port in url:
                return False
        return True

    return side_effect


class TestCheckServers:
    def test_all_healthy(self):
        from sova.llama_client import check_servers

        with patch("sova.llama_client._ensure_server", return_value=True):
            ok, msg = check_servers()
            assert ok is True
            assert msg == "ready"

    def test_embedding_down(self):
        from sova.llama_client import check_servers

        with patch(
            "sova.llama_client._ensure_server",
            side_effect=_mock_ensure_server_for("8081"),
        ):
            ok, msg = check_servers()
            assert ok is False
            assert "embedding" in msg

    def test_chat_down(self):
        from sova.llama_client import check_servers

        with patch(
            "sova.llama_client._ensure_server",
            side_effect=_mock_ensure_server_for("8083"),
        ):
            ok, msg = check_servers()
            assert ok is False
            assert "chat" in msg

    def test_reranker_down_is_warning(self):
        from sova.llama_client import check_servers

        with patch(
            "sova.llama_client._ensure_server",
            side_effect=_mock_ensure_server_for("8082"),
        ):
            ok, msg = check_servers()
            assert ok is True
            assert "reranker" in msg


class TestGetQueryEmbedding:
    def test_adds_instruction_prefix(self):
        from sova.llama_client import get_query_embedding

        captured = {}

        def urlopen_side_effect(req, timeout=None):
            captured["body"] = json.loads(req.data)
            return _mock_urlopen({"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]})

        with (
            patch("sova.llama_client._ensure_server", return_value=True),
            patch(
                "sova.llama_client.urllib.request.urlopen",
                side_effect=urlopen_side_effect,
            ),
        ):
            get_query_embedding("test query")
            assert "Instruct:" in captured["body"]["input"]
            assert "Query: test query" in captured["body"]["input"]

    def test_returns_float_list(self):
        from sova.llama_client import get_query_embedding

        def urlopen_side_effect(req, timeout=None):
            return _mock_urlopen({"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]})

        with (
            patch("sova.llama_client._ensure_server", return_value=True),
            patch(
                "sova.llama_client.urllib.request.urlopen",
                side_effect=urlopen_side_effect,
            ),
        ):
            result = get_query_embedding("test")
            assert isinstance(result, list)
            assert all(isinstance(v, float) for v in result)


class TestGetEmbeddingsBatch:
    def test_returns_list_of_embeddings(self):
        from sova.llama_client import get_embeddings_batch

        def urlopen_side_effect(req, timeout=None):
            return _mock_urlopen(
                {
                    "data": [
                        {"index": 0, "embedding": [0.1, 0.2]},
                        {"index": 1, "embedding": [0.3, 0.4]},
                    ]
                }
            )

        with (
            patch("sova.llama_client._ensure_server", return_value=True),
            patch(
                "sova.llama_client.urllib.request.urlopen",
                side_effect=urlopen_side_effect,
            ),
        ):
            result = get_embeddings_batch(["text1", "text2"])
            assert len(result) == 2
            assert all(isinstance(emb, list) for emb in result)

    def test_no_instruction_prefix(self):
        from sova.llama_client import get_embeddings_batch

        captured = {}

        def urlopen_side_effect(req, timeout=None):
            captured["body"] = json.loads(req.data)
            return _mock_urlopen({"data": [{"index": 0, "embedding": [0.1, 0.2]}]})

        with (
            patch("sova.llama_client._ensure_server", return_value=True),
            patch(
                "sova.llama_client.urllib.request.urlopen",
                side_effect=urlopen_side_effect,
            ),
        ):
            get_embeddings_batch(["test text"])
            assert "Instruct:" not in str(captured["body"]["input"])

    def test_preserves_order(self):
        from sova.llama_client import get_embeddings_batch

        def urlopen_side_effect(req, timeout=None):
            # Return in reverse order to test sorting
            return _mock_urlopen(
                {
                    "data": [
                        {"index": 1, "embedding": [0.3, 0.4]},
                        {"index": 0, "embedding": [0.1, 0.2]},
                    ]
                }
            )

        with (
            patch("sova.llama_client._ensure_server", return_value=True),
            patch(
                "sova.llama_client.urllib.request.urlopen",
                side_effect=urlopen_side_effect,
            ),
        ):
            result = get_embeddings_batch(["text1", "text2"])
            assert result[0] == [0.1, 0.2]
            assert result[1] == [0.3, 0.4]


class TestGenerateContext:
    def test_returns_stripped_string(self):
        from sova.llama_client import generate_context

        def urlopen_side_effect(req, timeout=None):
            return _mock_urlopen(
                {"choices": [{"message": {"content": "  This chunk covers auth.  "}}]}
            )

        with patch(
            "sova.llama_client.urllib.request.urlopen", side_effect=urlopen_side_effect
        ):
            result = generate_context("doc1", "Auth", "chunk text here")
            assert result == "This chunk covers auth."

    def test_prompt_contains_doc_and_section(self):
        from sova.llama_client import generate_context

        captured = {}

        def urlopen_side_effect(req, timeout=None):
            captured["body"] = json.loads(req.data)
            return _mock_urlopen({"choices": [{"message": {"content": "Context."}}]})

        with patch(
            "sova.llama_client.urllib.request.urlopen", side_effect=urlopen_side_effect
        ):
            generate_context("my-doc", "Introduction", "some text")
            prompt = captured["body"]["messages"][0]["content"]
            assert "my-doc" in prompt
            assert "Introduction" in prompt

    def test_none_section_uses_placeholder(self):
        from sova.llama_client import generate_context

        captured = {}

        def urlopen_side_effect(req, timeout=None):
            captured["body"] = json.loads(req.data)
            return _mock_urlopen({"choices": [{"message": {"content": "Context."}}]})

        with patch(
            "sova.llama_client.urllib.request.urlopen", side_effect=urlopen_side_effect
        ):
            generate_context("doc1", None, "text")
            prompt = captured["body"]["messages"][0]["content"]
            assert "(no section)" in prompt

    def test_surrounding_text_included(self):
        from sova.llama_client import generate_context

        captured = {}

        def urlopen_side_effect(req, timeout=None):
            captured["body"] = json.loads(req.data)
            return _mock_urlopen({"choices": [{"message": {"content": "Context."}}]})

        with patch(
            "sova.llama_client.urllib.request.urlopen", side_effect=urlopen_side_effect
        ):
            generate_context("doc1", "Sec", "main", "prev text", "next text")
            prompt = captured["body"]["messages"][0]["content"]
            assert "prev text" in prompt
            assert "next text" in prompt

    def test_empty_surrounding_uses_placeholders(self):
        from sova.llama_client import generate_context

        captured = {}

        def urlopen_side_effect(req, timeout=None):
            captured["body"] = json.loads(req.data)
            return _mock_urlopen({"choices": [{"message": {"content": "Context."}}]})

        with patch(
            "sova.llama_client.urllib.request.urlopen", side_effect=urlopen_side_effect
        ):
            generate_context("doc1", "Sec", "text", "", "")
            prompt = captured["body"]["messages"][0]["content"]
            assert "(start of document)" in prompt
            assert "(end of document)" in prompt

    def test_uses_context_model(self):
        from sova.llama_client import generate_context

        captured = {}

        def urlopen_side_effect(req, timeout=None):
            captured["body"] = json.loads(req.data)
            return _mock_urlopen({"choices": [{"message": {"content": "Context."}}]})

        with patch(
            "sova.llama_client.urllib.request.urlopen", side_effect=urlopen_side_effect
        ):
            generate_context("doc1", "Sec", "text")
            assert captured["body"]["model"] == "ministral-3-14b-instruct-2512"


class TestRerank:
    def test_success(self):
        from sova.llama_client import rerank

        def urlopen_side_effect(req, timeout=None):
            return _mock_urlopen(
                {
                    "results": [
                        {"index": 0, "relevance_score": 0.99},
                        {"index": 1, "relevance_score": 0.42},
                    ]
                }
            )

        with (
            patch("sova.llama_client._ensure_server", return_value=True),
            patch(
                "sova.llama_client.urllib.request.urlopen",
                side_effect=urlopen_side_effect,
            ),
        ):
            result = rerank("query", ["doc1", "doc2"], top_n=2)
            assert result is not None
            assert len(result) == 2
            assert result[0]["relevance_score"] == 0.99

    def test_connection_failure_returns_none(self):
        from sova.llama_client import rerank

        with patch("sova.llama_client._ensure_server", return_value=False):
            result = rerank("query", ["doc1", "doc2"])
            assert result is None

    def test_timeout_returns_none(self):
        from sova.llama_client import rerank

        def urlopen_side_effect(req, timeout=None):
            raise TimeoutError("timed out")

        with (
            patch("sova.llama_client._ensure_server", return_value=True),
            patch(
                "sova.llama_client.urllib.request.urlopen",
                side_effect=urlopen_side_effect,
            ),
        ):
            result = rerank("query", ["doc1"])
            assert result is None
