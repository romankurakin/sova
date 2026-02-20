"""Tests for llama_client module."""

import io
import json
import os
import pytest
import urllib.error
from unittest.mock import MagicMock, patch


def _mock_urlopen(response_body: dict, status: int = 200):
    """Create a mock for urllib.request.urlopen that returns JSON."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(response_body).encode()
    mock_resp.status = status
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _mock_urlopen_for_health(*up_ports: str):
    """Return a side_effect for urlopen that returns healthy for given ports only."""
    import urllib.error

    def side_effect(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        for port in up_ports:
            if f":{port}" in url:
                return _mock_urlopen({"status": "ok"})
        raise urllib.error.URLError("connection refused")

    return side_effect


class TestCheckServers:
    def _run_check(self, *up_ports: str, mode: str = "search"):
        from sova.llama_client import check_servers

        with (
            patch(
                "sova.llama_client.urllib.request.urlopen",
                side_effect=_mock_urlopen_for_health(*up_ports),
            ),
            patch("sova.llama_client.get_memory_hard_cap_gib", return_value=100.0),
            patch("sova.llama_client._plist_exists", return_value=False),
            patch("sova.llama_client._touch_activity"),
        ):
            return check_servers(mode=mode)

    def test_index_context_healthy(self):
        ok, msg = self._run_check("8083", mode="index_context")
        assert ok is True
        assert msg == "ready"

    def test_index_embed_healthy(self):
        ok, msg = self._run_check("8081", mode="index_embed")
        assert ok is True
        assert msg == "ready"

    def test_search_all_healthy(self):
        ok, msg = self._run_check("8081", "8082", mode="search")
        assert ok is True
        assert msg == "ready"

    def test_unknown_mode_raises(self):
        from sova.llama_client import check_servers

        with pytest.raises(ValueError, match="unknown server mode"):
            check_servers(mode="index")

    def test_chat_down(self):
        ok, msg = self._run_check(mode="index_context")
        assert ok is False
        assert "chat" in msg

    def test_embedding_down_for_index(self):
        ok, msg = self._run_check(mode="index_embed")
        assert ok is False
        assert "embedding" in msg

    def test_embedding_down_for_search(self):
        ok, msg = self._run_check("8082", mode="search")
        assert ok is False
        assert "embedding" in msg

    def test_reranker_down_no_plist_is_error(self):
        """Search requires reranker; missing service should fail."""
        ok, msg = self._run_check("8081", mode="search")
        assert ok is False
        assert "reranker" in msg

    def test_reranker_down_with_plist_is_error(self):
        """Search requires reranker; startup timeout should fail."""
        from sova.llama_client import check_servers

        with (
            patch(
                "sova.llama_client.urllib.request.urlopen",
                side_effect=_mock_urlopen_for_health("8081"),
            ),
            patch("sova.llama_client.get_memory_hard_cap_gib", return_value=100.0),
            patch("sova.llama_client._plist_exists", return_value=True),
            patch("sova.llama_client.subprocess.run"),
            patch("sova.llama_client._touch_activity"),
            patch("sova.llama_client.time.monotonic", side_effect=[0, 0, 301]),
            patch("sova.llama_client.time.sleep"),
        ):
            ok, msg = check_servers(mode="search")
            assert ok is False
            assert "reranker" in msg

    def test_reranker_ubatch_too_small_fails_preflight(self):
        from sova.llama_client import check_servers

        with (
            patch(
                "sova.llama_client.urllib.request.urlopen",
                side_effect=_mock_urlopen_for_health("8081", "8082"),
            ),
            patch("sova.llama_client.get_memory_hard_cap_gib", return_value=100.0),
            patch("sova.llama_client._plist_exists", return_value=True),
            patch("sova.llama_client._configured_ubatch_size", return_value=512),
            patch("sova.llama_client._touch_activity"),
        ):
            ok, msg = check_servers(mode="search")

        assert ok is False
        assert "--ubatch-size" in msg

    def test_admission_rejects_required_service(self):
        from sova.llama_client import check_servers

        with (
            patch("sova.llama_client.get_memory_hard_cap_gib", return_value=2.0),
            patch(
                "sova.llama_client.urllib.request.urlopen",
                side_effect=_mock_urlopen_for_health(),
            ),
            patch("sova.llama_client._plist_exists", return_value=False),
        ):
            ok, msg = check_servers(mode="index_context")

        # index_context uses one required model and should not be blocked by
        # conservative estimate-only preflight.
        assert ok is False
        assert "not reachable" in msg

    def test_index_phase_skips_strict_required_admission(self):
        from sova.llama_client import _admit_services_for_mode

        with patch("sova.llama_client.get_memory_hard_cap_gib", return_value=2.0):
            admitted, note = _admit_services_for_mode(
                "index_context",
                [("chat", "http://localhost:8083", True)],
            )

        assert admitted == [("chat", "http://localhost:8083", True)]
        assert note is None

    def test_search_skips_strict_required_admission_for_embedding(self):
        from sova.llama_client import _admit_services_for_mode

        with patch("sova.llama_client.get_memory_hard_cap_gib", return_value=2.0):
            admitted, note = _admit_services_for_mode(
                "search",
                [("embedding", "http://localhost:8081", True)],
            )

        assert admitted == [("embedding", "http://localhost:8081", True)]
        assert note is None

    def test_search_skips_strict_required_admission_for_reranker(self):
        from sova.llama_client import _admit_services_for_mode

        with patch("sova.llama_client.get_memory_hard_cap_gib", return_value=2.0):
            admitted, note = _admit_services_for_mode(
                "search",
                [
                    ("embedding", "http://localhost:8081", True),
                    ("reranker", "http://localhost:8082", True),
                ],
            )

        assert admitted == [
            ("embedding", "http://localhost:8081", True),
            ("reranker", "http://localhost:8082", True),
        ]
        assert note is None


class TestPostJson:
    def test_touches_activity_for_known_service_url(self):
        from sova.llama_client import _post_json

        with (
            patch(
                "sova.llama_client.urllib.request.urlopen",
                return_value=_mock_urlopen({"ok": True}),
            ),
            patch("sova.llama_client._touch_activity") as mock_touch,
        ):
            out = _post_json("http://localhost:8083/v1/chat/completions", {"x": 1})

        assert out == {"ok": True}
        mock_touch.assert_called_once_with("com.sova.chat")


class TestServiceRuntimeStatus:
    def test_reports_running_and_not_installed(self):
        from sova.llama_client import get_services_runtime_status

        def pid_for_port(port):
            if port == 8081:
                return 1234
            return None

        def rss_for_pid(pid):
            if pid == 1234:
                return 512.0
            return None

        with (
            patch("sova.llama_client._pid_for_port", side_effect=pid_for_port),
            patch("sova.llama_client._rss_mib_for_pid", side_effect=rss_for_pid),
            patch(
                "sova.llama_client._health_ok",
                side_effect=lambda url: ":8081" in url,
            ),
            patch(
                "sova.llama_client._plist_exists",
                side_effect=lambda label: label != "com.sova.reranker",
            ),
        ):
            rows = get_services_runtime_status()

        assert rows[0]["name"] == "embedding"
        assert rows[0]["state"] == "running"
        assert rows[0]["pid"] == 1234
        assert rows[0]["rss_mib"] == 512.0
        assert rows[1]["name"] == "reranker"
        assert rows[1]["state"] == "not installed"

    def test_reports_starting_when_pid_exists_but_health_is_down(self):
        from sova.llama_client import get_services_runtime_status

        with (
            patch(
                "sova.llama_client._pid_for_port",
                side_effect=lambda port: 2222 if port == 8082 else None,
            ),
            patch("sova.llama_client._rss_mib_for_pid", return_value=256.0),
            patch("sova.llama_client._health_ok", return_value=False),
            patch("sova.llama_client._plist_exists", return_value=True),
        ):
            rows = get_services_runtime_status()

        reranker = next(r for r in rows if r["name"] == "reranker")
        assert reranker["state"] == "starting"
        assert reranker["pid"] == 2222


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
            prompt = captured["body"]["input"][0]
            assert "Instruct:" in prompt
            assert "Query: test query" in prompt

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


def test_server_status_download_progress_is_bucketed(tmp_path, monkeypatch):
    from sova import llama_client

    monkeypatch.setattr(llama_client, "_LLAMA_CACHE", tmp_path)
    monkeypatch.setitem(llama_client._CACHE_FILES, "com.sova.chat", "chat.gguf")
    dl_path = tmp_path / "chat.gguf.downloadInProgress"
    dl_path.touch()

    # 1.64 GiB should be shown as 1.5 GiB (0.5 GiB step).
    with dl_path.open("wb") as f:
        f.truncate(int(1.64 * (1024**3)))
    assert llama_client._server_status("com.sova.chat") == "downloading (1.5 GB)"

    # 2.01 GiB should step up to 2.0 GiB.
    with dl_path.open("wb") as f:
        f.truncate(int(2.01 * (1024**3)))
    assert llama_client._server_status("com.sova.chat") == "downloading (2.0 GB)"


class TestGetEmbeddingsBatch:
    def test_returns_list_of_embeddings(self):
        from sova.llama_client import get_embeddings_batch

        def embed_side_effect(batch, timeout=None):
            return [[float(len(text)), float(i)] for i, text in enumerate(batch)]

        with (
            patch("sova.llama_client._ensure_server", return_value=True),
            patch(
                "sova.llama_client._embed_inputs_via_server",
                side_effect=embed_side_effect,
            ),
        ):
            result = get_embeddings_batch(["text1", "text2"])
            assert len(result) == 2
            assert all(isinstance(emb, list) for emb in result)

    def test_empty_input_returns_empty(self):
        from sova.llama_client import get_embeddings_batch

        assert get_embeddings_batch([]) == []

    def test_preserves_order(self):
        from sova.llama_client import get_embeddings_batch

        def embed_side_effect(batch, timeout=None):
            out = []
            for text in batch:
                idx = int(text.removeprefix("text"))
                out.append([float(idx)])
            return out

        with (
            patch("sova.llama_client._ensure_server", return_value=True),
            patch(
                "sova.llama_client._embed_inputs_via_server",
                side_effect=embed_side_effect,
            ),
            patch("sova.llama_client._EMBED_BATCH_SIZE", 2),
        ):
            texts = [f"text{i}" for i in range(1, 11)]
            result = get_embeddings_batch(texts)
            assert result == [[float(i)] for i in range(1, 11)]

    def test_fails_fast_on_embedding_error(self):
        from sova.llama_client import ServerError, get_embeddings_batch

        def embed_side_effect(batch, timeout=None):
            if any(text == "boom" for text in batch):
                raise ServerError("crashed")
            return [[0.1] for _ in batch]

        with (
            patch("sova.llama_client._ensure_server", return_value=True),
            patch(
                "sova.llama_client._embed_inputs_via_server",
                side_effect=embed_side_effect,
            ),
            patch("sova.llama_client._EMBED_BATCH_SIZE", 2),
        ):
            with pytest.raises(ServerError, match="embedding server failed"):
                get_embeddings_batch(["ok1", "ok2", "boom", "ok3"])

    def test_on_batch_callback(self):
        from sova.llama_client import get_embeddings_batch

        seen: list[tuple[list[int], list[list[float]]]] = []

        def embed_side_effect(batch, timeout=None):
            return [[float(i)] for i, _ in enumerate(batch)]

        def on_batch(indices, embeddings, _metrics):
            seen.append((indices, embeddings))

        with (
            patch("sova.llama_client._ensure_server", return_value=True),
            patch(
                "sova.llama_client._embed_inputs_via_server",
                side_effect=embed_side_effect,
            ),
            patch("sova.llama_client._EMBED_BATCH_SIZE", 2),
        ):
            get_embeddings_batch(["a", "b", "c", "d"], on_batch=on_batch)

        flattened = [idx for batch_indices, _ in seen for idx in batch_indices]
        assert sorted(flattened) == [0, 1, 2, 3]


class TestGenerateContext:
    def _mock_with_health(self, response_body, captured=None):
        """Create urlopen side_effect that handles both /health and API calls."""

        def side_effect(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "/health" in url:
                return _mock_urlopen({"status": "ok"})
            if captured is not None and hasattr(req, "data"):
                captured["body"] = json.loads(req.data)
            return _mock_urlopen(response_body)

        return side_effect

    def test_returns_stripped_string(self):
        from sova.llama_client import generate_context

        with patch(
            "sova.llama_client.urllib.request.urlopen",
            side_effect=self._mock_with_health(
                {"choices": [{"message": {"content": "  This chunk covers auth.  "}}]}
            ),
        ):
            result = generate_context("doc1", "Auth", "chunk text here")
            assert result == "This chunk covers auth."

    def test_prompt_contains_doc_and_section(self):
        from sova.llama_client import generate_context

        captured = {}

        with patch(
            "sova.llama_client.urllib.request.urlopen",
            side_effect=self._mock_with_health(
                {"choices": [{"message": {"content": "Context."}}]}, captured
            ),
        ):
            generate_context("my-doc", "Introduction", "some text")
            prompt = captured["body"]["messages"][1]["content"]
            assert "my-doc" in prompt
            assert "Introduction" in prompt

    def test_none_section_uses_placeholder(self):
        from sova.llama_client import generate_context

        captured = {}

        with patch(
            "sova.llama_client.urllib.request.urlopen",
            side_effect=self._mock_with_health(
                {"choices": [{"message": {"content": "Context."}}]}, captured
            ),
        ):
            generate_context("doc1", None, "text")
            prompt = captured["body"]["messages"][1]["content"]
            assert "(no section)" in prompt

    def test_surrounding_text_included(self):
        from sova.llama_client import generate_context

        captured = {}

        with patch(
            "sova.llama_client.urllib.request.urlopen",
            side_effect=self._mock_with_health(
                {"choices": [{"message": {"content": "Context."}}]}, captured
            ),
        ):
            generate_context("doc1", "Sec", "main", "prev text", "next text")
            prompt = captured["body"]["messages"][1]["content"]
            assert "prev text" in prompt
            assert "next text" in prompt

    def test_empty_surrounding_uses_placeholders(self):
        from sova.llama_client import generate_context

        captured = {}

        with patch(
            "sova.llama_client.urllib.request.urlopen",
            side_effect=self._mock_with_health(
                {"choices": [{"message": {"content": "Context."}}]}, captured
            ),
        ):
            generate_context("doc1", "Sec", "text", "", "")
            prompt = captured["body"]["messages"][1]["content"]
            assert "(start of document)" in prompt
            assert "(end of document)" in prompt

    def test_uses_context_model(self):
        from sova.llama_client import generate_context

        captured = {}

        with patch(
            "sova.llama_client.urllib.request.urlopen",
            side_effect=self._mock_with_health(
                {"choices": [{"message": {"content": "Context."}}]}, captured
            ),
        ):
            generate_context("doc1", "Sec", "text")
            assert captured["body"]["model"] == "ministral-3-14b-instruct-2512"
            assert captured["body"]["temperature"] == 0.0
            assert captured["body"]["max_tokens"] == 96
            assert captured["body"]["messages"][0]["role"] == "system"

    def test_parse_error_falls_back_to_completion_endpoint(self):
        from sova.llama_client import generate_context

        def side_effect(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "/health" in url:
                return _mock_urlopen({"status": "ok"})
            if "/v1/chat/completions" in url:
                raise urllib.error.HTTPError(
                    url=url,
                    code=500,
                    msg="Internal Server Error",
                    hdrs=None,
                    fp=io.BytesIO(
                        b'{"error":{"code":500,"message":"Failed to parse input at pos 0","type":"server_error"}}'
                    ),
                )
            if "/completion" in url:
                return _mock_urlopen({"content": "Fallback plain sentence."})
            raise AssertionError(f"unexpected url: {url}")

        with patch(
            "sova.llama_client.urllib.request.urlopen",
            side_effect=side_effect,
        ):
            result = generate_context("doc1", "Sec", "text")

        assert result == "Fallback plain sentence."


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

    def test_connection_failure_raises(self):
        from sova.llama_client import ServerError, rerank

        with patch("sova.llama_client._ensure_server", return_value=False):
            with pytest.raises(ServerError, match="reranker server not reachable"):
                rerank("query", ["doc1", "doc2"])

    def test_timeout_raises(self):
        from sova.llama_client import ServerError, rerank

        def urlopen_side_effect(req, timeout=None):
            raise TimeoutError("timed out")

        with (
            patch("sova.llama_client._ensure_server", return_value=True),
            patch(
                "sova.llama_client.urllib.request.urlopen",
                side_effect=urlopen_side_effect,
            ),
        ):
            with pytest.raises(ServerError, match="server timeout"):
                rerank("query", ["doc1"])

    def test_invalid_response_raises(self):
        from sova.llama_client import ServerError, rerank

        def urlopen_side_effect(req, timeout=None):
            return _mock_urlopen({"results": "not a list"})

        with (
            patch("sova.llama_client._ensure_server", return_value=True),
            patch(
                "sova.llama_client.urllib.request.urlopen",
                side_effect=urlopen_side_effect,
            ),
        ):
            with pytest.raises(ServerError, match="invalid rerank response"):
                rerank("query", ["doc1"])

    def test_sends_full_documents_in_request(self):
        from sova.llama_client import rerank

        captured = {}

        def urlopen_side_effect(req, timeout=None):
            captured["body"] = json.loads(req.data.decode())
            return _mock_urlopen({"results": [{"index": 0, "relevance_score": 0.5}]})

        with (
            patch("sova.llama_client._ensure_server", return_value=True),
            patch(
                "sova.llama_client.urllib.request.urlopen",
                side_effect=urlopen_side_effect,
            ),
        ):
            rerank("query", ["x" * 8000], top_n=1)

        sent_doc = captured["body"]["documents"][0]
        assert len(sent_doc) == 8000

    def test_returns_actionable_error_for_physical_batch_limit(self):
        from sova.llama_client import ServerError, rerank

        with (
            patch("sova.llama_client._ensure_server", return_value=True),
            patch(
                "sova.llama_client._post_json",
                side_effect=ServerError(
                    "server error 500: input (567 tokens) is too large to process. "
                    "increase the physical batch size (current batch size: 512)"
                ),
            ),
        ):
            with pytest.raises(ServerError, match="increase com.sova.reranker --ubatch-size"):
                rerank("query", ["doc1"], top_n=1)


class TestStopServer:
    def test_stops_known_service(self):
        from sova.llama_client import stop_server

        mock_run = MagicMock()
        # First call: launchctl stop; second call: launchctl list (already stopped)
        mock_run.side_effect = [
            MagicMock(),  # stop
            MagicMock(returncode=0, stdout="- 0\tcom.sova.embedding"),  # list: stopped
        ]

        with (
            patch("sova.llama_client.subprocess.run", mock_run),
            patch("sova.llama_client.time.sleep"),
            patch("sova.llama_client._ACTIVITY_DIR") as mock_dir,
        ):
            mock_file = MagicMock()
            mock_dir.__truediv__ = MagicMock(return_value=mock_file)
            stop_server("http://localhost:8081")

        assert mock_run.call_count == 2
        mock_file.unlink.assert_any_call(missing_ok=True)
        assert mock_file.unlink.call_count >= 1

    def test_noop_for_unknown_url(self):
        from sova.llama_client import stop_server

        with patch("sova.llama_client.subprocess.run") as mock_run:
            stop_server("http://localhost:9999")
            mock_run.assert_not_called()

    def test_propagates_keyboard_interrupt_by_default(self):
        from sova.llama_client import stop_server

        with (
            patch(
                "sova.llama_client.subprocess.run",
                side_effect=KeyboardInterrupt(),
            ),
            patch("sova.llama_client._ACTIVITY_DIR") as mock_dir,
        ):
            mock_file = MagicMock()
            mock_dir.__truediv__ = MagicMock(return_value=mock_file)
            with pytest.raises(KeyboardInterrupt):
                stop_server("http://localhost:8081")
            mock_file.unlink.assert_any_call(missing_ok=True)
            assert mock_file.unlink.call_count >= 1

    def test_suppresses_keyboard_interrupt_when_requested(self):
        from sova.llama_client import stop_server

        with (
            patch(
                "sova.llama_client.subprocess.run",
                side_effect=KeyboardInterrupt(),
            ),
            patch("sova.llama_client._ACTIVITY_DIR") as mock_dir,
        ):
            mock_file = MagicMock()
            mock_dir.__truediv__ = MagicMock(return_value=mock_file)
            stop_server("http://localhost:8081", suppress_interrupt=True)
            mock_file.unlink.assert_any_call(missing_ok=True)
            assert mock_file.unlink.call_count >= 1

    def test_suppresses_keyboard_interrupt_from_cleanup_when_requested(self):
        from sova.llama_client import stop_server

        mock_run = MagicMock()
        mock_run.side_effect = [
            MagicMock(),  # launchctl stop
            MagicMock(returncode=0, stdout="- 0\tcom.sova.embedding"),  # already stopped
        ]

        with (
            patch("sova.llama_client.subprocess.run", mock_run),
            patch("sova.llama_client.time.sleep"),
            patch("sova.llama_client._ACTIVITY_DIR") as mock_dir,
        ):
            mock_file = MagicMock()
            mock_file.unlink.side_effect = KeyboardInterrupt()
            mock_dir.__truediv__ = MagicMock(return_value=mock_file)
            stop_server("http://localhost:8081", suppress_interrupt=True)

    def test_propagates_keyboard_interrupt_from_cleanup_by_default(self):
        from sova.llama_client import stop_server

        mock_run = MagicMock()
        mock_run.side_effect = [
            MagicMock(),  # launchctl stop
            MagicMock(returncode=0, stdout="- 0\tcom.sova.embedding"),  # already stopped
        ]

        with (
            patch("sova.llama_client.subprocess.run", mock_run),
            patch("sova.llama_client.time.sleep"),
            patch("sova.llama_client._ACTIVITY_DIR") as mock_dir,
        ):
            mock_file = MagicMock()
            mock_file.unlink.side_effect = KeyboardInterrupt()
            mock_dir.__truediv__ = MagicMock(return_value=mock_file)
            with pytest.raises(KeyboardInterrupt):
                stop_server("http://localhost:8081")


class TestCleanupIdleServices:
    def test_stops_search_pair_together_with_shared_activity(self, tmp_path):
        from sova.llama_client import cleanup_idle_services

        now = 10_000.0
        labels = ["com.sova.search", "com.sova.embedding", "com.sova.reranker"]
        for label in labels:
            path = tmp_path / label
            path.write_bytes(b"")
            os.utime(path, (now - 1_000.0, now - 1_000.0))

        with (
            patch("sova.llama_client._ACTIVITY_DIR", tmp_path),
            patch("sova.llama_client._IDLE_TIMEOUT", 900),
            patch("sova.llama_client.time.time", return_value=now),
            patch("sova.llama_client.subprocess.run") as mock_run,
        ):
            cleanup_idle_services()

        stopped = {
            call.args[0][2]
            for call in mock_run.call_args_list
            if len(call.args) == 1 and len(call.args[0]) >= 3
        }
        assert "com.sova.embedding" in stopped
        assert "com.sova.reranker" in stopped
        for label in labels:
            assert not (tmp_path / label).exists()

    def test_keeps_search_pair_when_shared_activity_is_fresh(self, tmp_path):
        from sova.llama_client import cleanup_idle_services

        now = 10_000.0
        search = tmp_path / "com.sova.search"
        embedding = tmp_path / "com.sova.embedding"
        reranker = tmp_path / "com.sova.reranker"
        for path in (search, embedding, reranker):
            path.write_bytes(b"")
        os.utime(search, (now - 100.0, now - 100.0))
        os.utime(embedding, (now - 2_000.0, now - 2_000.0))
        os.utime(reranker, (now - 2_000.0, now - 2_000.0))

        with (
            patch("sova.llama_client._ACTIVITY_DIR", tmp_path),
            patch("sova.llama_client._IDLE_TIMEOUT", 900),
            patch("sova.llama_client.time.time", return_value=now),
            patch("sova.llama_client.subprocess.run") as mock_run,
        ):
            cleanup_idle_services()

        mock_run.assert_not_called()
        assert search.exists()
        assert embedding.exists()
        assert reranker.exists()

class TestRunEmbeddingCanary:
    def test_sends_canary_requests(self):
        from sova.llama_client import run_embedding_canary, _EMBED_CANARY_REQUESTS

        call_count = 0

        def embed_side_effect(texts, timeout=None):
            nonlocal call_count
            call_count += 1
            return [[0.1] * 3]

        with (
            patch("sova.llama_client._ensure_server", return_value=True),
            patch(
                "sova.llama_client._embed_inputs_via_server",
                side_effect=embed_side_effect,
            ),
        ):
            run_embedding_canary()

        assert call_count == _EMBED_CANARY_REQUESTS

    def test_raises_when_server_unreachable(self):
        from sova.llama_client import ServerError, run_embedding_canary

        with patch("sova.llama_client._ensure_server", return_value=False):
            with pytest.raises(ServerError, match="embedding server not reachable"):
                run_embedding_canary()

    def test_supports_custom_request_count(self):
        from sova.llama_client import run_embedding_canary

        call_count = 0

        def embed_side_effect(texts, timeout=None):
            nonlocal call_count
            call_count += 1
            return [[0.1] * 3]

        with (
            patch("sova.llama_client._ensure_server", return_value=True),
            patch(
                "sova.llama_client._embed_inputs_via_server",
                side_effect=embed_side_effect,
            ),
        ):
            run_embedding_canary(requests=3)

        assert call_count == 3
