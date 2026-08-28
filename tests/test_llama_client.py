"""Tests for llama_client module."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest


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
    def _run_check(
        self,
        *up_ports: str,
        mode: str = "search",
    ):
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
        ok, msg = self._run_check("8081", mode="search")
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
        ok, msg = self._run_check(mode="search")
        assert ok is False
        assert "embedding" in msg

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

        # index_context uses one required model and should not be blocked by.
        # conservative estimate-only preflight.
        assert ok is False
        assert "not reachable" in msg

    def test_index_phase_skips_strict_required_admission(self):
        from sova.llama_client import _admit_services_for_mode

        with patch("sova.llama_client.get_memory_hard_cap_gib", return_value=2.0):
            admitted, note = _admit_services_for_mode(
                "index_context",
                [("chat", "http://127.0.0.1:8083", True)],
            )

        assert admitted == [("chat", "http://127.0.0.1:8083", True)]
        assert note is None

    def test_search_skips_strict_required_admission_for_embedding(self):
        from sova.llama_client import _admit_services_for_mode

        with patch("sova.llama_client.get_memory_hard_cap_gib", return_value=2.0):
            admitted, note = _admit_services_for_mode(
                "search",
                [("embedding", "http://127.0.0.1:8081", True)],
            )

        assert admitted == [("embedding", "http://127.0.0.1:8081", True)]
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
            out = _post_json("http://127.0.0.1:8083/v1/chat/completions", {"x": 1})

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
                return_value=True,
            ),
        ):
            rows = get_services_runtime_status()

        assert rows[0]["name"] == "embedding"
        assert rows[0]["state"] == "running"
        assert rows[0]["pid"] == 1234
        assert rows[0]["rss_mib"] == 512.0
        assert rows[1]["name"] == "chat"
        assert rows[1]["state"] == "stopped"

    def test_reports_starting_when_pid_exists_but_health_is_down(self):
        from sova.llama_client import get_services_runtime_status

        with (
            patch(
                "sova.llama_client._pid_for_port",
                side_effect=lambda port: 2222 if port == 8083 else None,
            ),
            patch("sova.llama_client._rss_mib_for_pid", return_value=256.0),
            patch("sova.llama_client._health_ok", return_value=False),
            patch("sova.llama_client._plist_exists", return_value=True),
        ):
            rows = get_services_runtime_status()

        chat = next(r for r in rows if r["name"] == "chat")
        assert chat["state"] == "starting"
        assert chat["pid"] == 2222


class TestGetQueryEmbedding:
    def test_adds_instruction_prefix(self):
        from sova.llama_client import get_query_embedding

        captured = {}

        def urlopen_side_effect(req, timeout=None):
            captured["body"] = json.loads(req.data)
            return _mock_urlopen({"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]})

        with (
            patch("sova.llama_client._ensure_server", return_value=True),
            patch("sova.llama_client.EMBEDDING_DIM", 3),
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
            patch("sova.llama_client.EMBEDDING_DIM", 3),
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

    monkeypatch.setattr(llama_client, "_HF_HUB_CACHE", tmp_path / "hf")
    repo, _ = llama_client._MODEL_SPECS["com.sova.chat"]
    blobs = tmp_path / "hf" / ("models--" + repo.replace("/", "--")) / "blobs"
    blobs.mkdir(parents=True)
    dl_path = blobs / "chat.downloadInProgress"

    # 1.64 GiB should be shown as 1.5 GiB (0.5 GiB step).
    with dl_path.open("wb") as f:
        f.truncate(int(1.64 * (1024**3)))
    assert llama_client._server_status("com.sova.chat") == "downloading (1.5 GB)"

    # 2.01 GiB should step up to 2.0 GiB.
    with dl_path.open("wb") as f:
        f.truncate(int(2.01 * (1024**3)))
    assert llama_client._server_status("com.sova.chat") == "downloading (2.0 GB)"


def _hf_model_layout(root, repo: str, filename: str, *, complete: bool):
    """Create a Hugging Face hub cache layout for a model download."""
    model_dir = root / ("models--" + repo.replace("/", "--"))
    blobs = model_dir / "blobs"
    snapshots = model_dir / "snapshots" / "rev0"
    blobs.mkdir(parents=True)
    snapshots.mkdir(parents=True)
    if complete:
        blob = blobs / "sha256abc"
        blob.write_bytes(b"gguf")
        (snapshots / filename).symlink_to(blob)
    else:
        (blobs / "sha256abc.downloadInProgress").write_bytes(b"gg")
    return model_dir


def test_is_model_cached_detects_hf_hub_layout(tmp_path, monkeypatch):
    from sova import llama_client

    monkeypatch.setattr(llama_client, "_HF_HUB_CACHE", tmp_path / "hf")

    repo, filename = llama_client._MODEL_SPECS["com.sova.embedding"]
    assert llama_client.is_model_cached("com.sova.embedding") is False

    _hf_model_layout(tmp_path / "hf", repo, filename or "model.gguf", complete=True)
    assert llama_client.is_model_cached("com.sova.embedding") is True
    assert llama_client._server_status("com.sova.embedding") == "loading"


def test_is_model_cached_false_while_hf_download_in_progress(tmp_path, monkeypatch):
    from sova import llama_client

    monkeypatch.setattr(llama_client, "_HF_HUB_CACHE", tmp_path / "hf")

    repo, filename = llama_client._MODEL_SPECS["com.sova.embedding"]
    _hf_model_layout(tmp_path / "hf", repo, filename or "model.gguf", complete=False)

    assert llama_client.is_model_cached("com.sova.embedding") is False
    assert llama_client._server_status("com.sova.embedding").startswith("downloading")


def test_embedding_token_budget_uses_dynamic_margin():
    from sova.llama_client import _embedding_token_budget

    with patch("sova.llama_client._configured_ctx_size", return_value=4096):
        # 2% would be 82, but margin has a 128-token minimum.
        assert _embedding_token_budget() == 3968

    with patch("sova.llama_client._configured_ctx_size", return_value=12288):
        # Computed value is capped by stable embedding budget.
        assert _embedding_token_budget() == 4096

    with patch("sova.llama_client._configured_ctx_size", return_value=20000):
        # Computed value is capped by stable embedding budget.
        assert _embedding_token_budget() == 4096


def test_token_counts_batch_splits_one_server_response(monkeypatch):
    from sova import llama_client

    tokenized = {
        "<|endoftext|>": [99],
        "hello<|endoftext|>привет<|endoftext|>": [1, 99, 2, 3, 99],
    }
    calls: list[str] = []

    def tokenize(text: str) -> list[int]:
        calls.append(text)
        return tokenized[text]

    llama_client._TOKENIZE_SEPARATOR_IDS.clear()
    monkeypatch.setattr(llama_client, "_token_ids_via_server", tokenize)

    assert llama_client.get_token_counts_batch(["hello", "привет", ""]) == [1, 2, 0]
    assert calls == [
        "<|endoftext|>",
        "hello<|endoftext|>привет<|endoftext|>",
    ]


def test_token_counts_batch_bounds_request_size(monkeypatch):
    from sova import llama_client

    groups: list[list[str]] = []
    monkeypatch.setattr(llama_client, "_EMBED_TOKENIZE_BATCH_CHARS", 8)
    monkeypatch.setattr(
        llama_client,
        "_token_counts_group",
        lambda texts: groups.append(list(texts)) or [len(text) for text in texts],
    )

    assert llama_client.get_token_counts_batch(["aaaa", "bbbb", "cc"]) == [4, 4, 2]
    assert groups == [["aaaa"], ["bbbb"], ["cc"]]


def test_token_count_request_disables_bos_and_parses_batch_markers(monkeypatch):
    from sova import llama_client

    captured: dict = {}

    def post(_url, payload, timeout):
        captured.update(payload)
        assert timeout == llama_client._EMBED_TOKENIZE_TIMEOUT_S
        return {"tokens": [1, 2]}

    monkeypatch.setattr(llama_client, "_post_json", post)

    assert llama_client._token_count_via_server("hello") == 2
    assert captured == {
        "content": "hello",
        "add_special": False,
        "parse_special": True,
    }


class TestGetEmbeddingsBatch:
    def test_returns_list_of_embeddings(self):
        from sova.llama_client import get_embeddings_batch

        def embed_side_effect(batch, timeout=None):
            return [[float(len(text)), float(i)] for i, text in enumerate(batch)]

        with (
            patch("sova.llama_client._ensure_server", return_value=True),
            patch(
                "sova.llama_client._prepare_embedding_text",
                side_effect=lambda text, token_budget=None: text,
            ),
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
                "sova.llama_client._prepare_embedding_text",
                side_effect=lambda text, token_budget=None: text,
            ),
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
                "sova.llama_client._prepare_embedding_text",
                side_effect=lambda text, token_budget=None: text,
            ),
            patch(
                "sova.llama_client._embed_inputs_via_server",
                side_effect=embed_side_effect,
            ),
            patch("sova.llama_client._EMBED_BATCH_SIZE", 2),
            pytest.raises(ServerError, match="embedding server failed"),
        ):
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
                "sova.llama_client._prepare_embedding_text",
                side_effect=lambda text, token_budget=None: text,
            ),
            patch(
                "sova.llama_client._embed_inputs_via_server",
                side_effect=embed_side_effect,
            ),
            patch("sova.llama_client._EMBED_BATCH_SIZE", 2),
        ):
            get_embeddings_batch(["a", "b", "c", "d"], on_batch=on_batch)

        flattened = [idx for batch_indices, _ in seen for idx in batch_indices]
        assert sorted(flattened) == [0, 1, 2, 3]

    def test_token_budget_trims_body_and_keeps_header(self):
        from sova.llama_client import get_embeddings_batch

        captured_batches: list[list[str]] = []

        def embed_side_effect(batch, timeout=None):
            captured_batches.append(list(batch))
            return [[0.1] for _ in batch]

        long_text = "[doc | section]\n\n" + ("abcdefghij " * 20)
        with (
            patch("sova.llama_client._ensure_server", return_value=True),
            patch("sova.llama_client._embedding_token_budget", return_value=48),
            patch(
                "sova.llama_client._token_count_via_server",
                side_effect=lambda text: len(text),
            ),
            patch(
                "sova.llama_client._embed_inputs_via_server",
                side_effect=embed_side_effect,
            ),
        ):
            get_embeddings_batch(["short", long_text])

        assert captured_batches
        sent = captured_batches[0][1]
        assert sent.startswith("[doc | section]\n\n")
        assert len(sent) <= 48

    def test_recovers_remote_close_with_single_requests(self):
        from sova.llama_client import get_embeddings_batch

        seen_batches: list[list[str]] = []

        def embed_side_effect(batch, timeout=None):
            seen_batches.append(list(batch))
            if len(batch) > 1:
                raise RuntimeError("Remote end closed connection without response")
            return [[float(len(batch[0]))]]

        with (
            patch("sova.llama_client._ensure_server", return_value=True),
            patch(
                "sova.llama_client._prepare_embedding_text",
                side_effect=lambda text, token_budget=None: text,
            ),
            patch(
                "sova.llama_client._embed_inputs_via_server",
                side_effect=embed_side_effect,
            ),
            patch("sova.llama_client._EMBED_BATCH_SIZE", 3),
        ):
            result = get_embeddings_batch(["a", "bb", "ccc"])

        assert [len(batch) for batch in seen_batches] == [3, 1, 1, 1]
        assert len(result) == 3

    def test_fails_fast_on_preflight_prepare_error(self):
        from sova.llama_client import ServerError, get_embeddings_batch

        def prepare_side_effect(text, token_budget=None):
            if text == "boom":
                raise RuntimeError("tokenize down")
            return text

        with (
            patch("sova.llama_client._ensure_server", return_value=True),
            patch(
                "sova.llama_client._prepare_embedding_text",
                side_effect=prepare_side_effect,
            ),
            patch(
                "sova.llama_client._embed_inputs_via_server",
                side_effect=lambda batch, timeout=None: [[0.1] for _ in batch],
            ),
            patch("sova.llama_client._EMBED_BATCH_SIZE", 2),
            pytest.raises(ServerError, match="embedding preflight failed"),
        ):
            get_embeddings_batch(["ok1", "ok2", "boom", "ok3"])

    def test_compacts_long_header_path_when_needed(self):
        from sova.llama_client import _prepare_embedding_text

        text = "[doc | alpha | beta | gamma]\n\nbody body body"
        with patch(
            "sova.llama_client._token_count_via_server",
            side_effect=lambda content: len(content),
        ):
            prepared = _prepare_embedding_text(text, token_budget=28)

        assert prepared.startswith("[doc | beta | gamma]\n\n")
        assert len(prepared) <= 28


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
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"context":"Authentication rules govern account access."}'
                            }
                        }
                    ]
                }
            ),
        ):
            result = generate_context("doc1", "Auth", "chunk text here")
            assert result == "Authentication rules govern account access."

    def test_prompt_contains_doc_and_section(self):
        from sova.llama_client import generate_context

        captured = {}

        with patch(
            "sova.llama_client.urllib.request.urlopen",
            side_effect=self._mock_with_health(
                {"choices": [{"message": {"content": "A complete context sentence."}}]},
                captured,
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
                {"choices": [{"message": {"content": "A complete context sentence."}}]},
                captured,
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
                {"choices": [{"message": {"content": "A complete context sentence."}}]},
                captured,
            ),
        ):
            generate_context("doc1", "Sec", "main", "prev text", "next text")
            prompt = captured["body"]["messages"][1]["content"]
            assert "prev text" in prompt
            assert "next text" in prompt

    def test_target_passage_is_not_truncated(self):
        from sova.llama_client import generate_context

        captured = {}
        target = "start " + ("content " * 300) + "TARGET_END_SENTINEL"

        with patch(
            "sova.llama_client.urllib.request.urlopen",
            side_effect=self._mock_with_health(
                {"choices": [{"message": {"content": "A complete context sentence."}}]},
                captured,
            ),
        ):
            generate_context("doc1", "Chapter > Section", target)

        prompt = captured["body"]["messages"][1]["content"]
        assert "TARGET_END_SENTINEL" in prompt
        assert "Chapter > Section" in prompt

    def test_empty_surrounding_uses_placeholders(self):
        from sova.llama_client import generate_context

        captured = {}

        with patch(
            "sova.llama_client.urllib.request.urlopen",
            side_effect=self._mock_with_health(
                {"choices": [{"message": {"content": "A complete context sentence."}}]},
                captured,
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
                {"choices": [{"message": {"content": "A complete context sentence."}}]},
                captured,
            ),
        ):
            generate_context("doc1", "Sec", "text")
            assert captured["body"]["model"] == "qwen3.8-27b"
            assert captured["body"]["temperature"] == 0.0
            assert captured["body"]["max_tokens"] == 192
            assert captured["body"]["reasoning_effort"] == "low"
            assert captured["body"]["response_format"]["type"] == "json_schema"
            assert captured["body"]["messages"][0]["role"] == "system"

    def test_context_validation_keeps_only_structural_safeguards(self):
        from sova.llama_client import ServerError, _validate_context_response

        with pytest.raises(ServerError, match="empty"):
            _validate_context_response("   ")
        assert (
            _validate_context_response(
                "In the RISC-V specification, sbi_nacl_sync_sret handles values < 100."
            )
            == "In the RISC-V specification, sbi_nacl_sync_sret handles values < 100."
        )


class TestStopServer:
    def test_stops_known_service(self):
        from sova.llama_client import stop_server

        mock_run = MagicMock()
        # First call: launchctl stop; second call: launchctl list (already stopped).
        mock_run.side_effect = [
            MagicMock(),  # stop.
            MagicMock(returncode=0, stdout="- 0\tcom.sova.embedding"),  # list: stopped.
        ]

        with (
            patch("sova.llama_client.subprocess.run", mock_run),
            patch("sova.llama_client.time.sleep"),
            patch("sova.llama_client._ACTIVITY_DIR") as mock_dir,
        ):
            mock_file = MagicMock()
            mock_dir.__truediv__ = MagicMock(return_value=mock_file)
            stop_server("http://127.0.0.1:8081")

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
                stop_server("http://127.0.0.1:8081")
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
            stop_server("http://127.0.0.1:8081", suppress_interrupt=True)
            mock_file.unlink.assert_any_call(missing_ok=True)
            assert mock_file.unlink.call_count >= 1

    def test_suppresses_keyboard_interrupt_from_cleanup_when_requested(self):
        from sova.llama_client import stop_server

        mock_run = MagicMock()
        mock_run.side_effect = [
            MagicMock(),  # launchctl stop.
            MagicMock(
                returncode=0, stdout="- 0\tcom.sova.embedding"
            ),  # already stopped.
        ]

        with (
            patch("sova.llama_client.subprocess.run", mock_run),
            patch("sova.llama_client.time.sleep"),
            patch("sova.llama_client._ACTIVITY_DIR") as mock_dir,
        ):
            mock_file = MagicMock()
            mock_file.unlink.side_effect = KeyboardInterrupt()
            mock_dir.__truediv__ = MagicMock(return_value=mock_file)
            stop_server("http://127.0.0.1:8081", suppress_interrupt=True)

    def test_propagates_keyboard_interrupt_from_cleanup_by_default(self):
        from sova.llama_client import stop_server

        mock_run = MagicMock()
        mock_run.side_effect = [
            MagicMock(),  # launchctl stop.
            MagicMock(
                returncode=0, stdout="- 0\tcom.sova.embedding"
            ),  # already stopped.
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
                stop_server("http://127.0.0.1:8081")


class TestCleanupIdleServices:
    def test_stops_each_idle_service(self, tmp_path):
        from sova.llama_client import cleanup_idle_services

        now = 10_000.0
        labels = ["com.sova.embedding", "com.sova.chat"]
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
        assert "com.sova.chat" in stopped
        for label in labels:
            assert not (tmp_path / label).exists()

    def test_keeps_fresh_service(self, tmp_path):
        from sova.llama_client import cleanup_idle_services

        now = 10_000.0
        embedding = tmp_path / "com.sova.embedding"
        embedding.write_bytes(b"")
        os.utime(embedding, (now - 100.0, now - 100.0))

        with (
            patch("sova.llama_client._ACTIVITY_DIR", tmp_path),
            patch("sova.llama_client._IDLE_TIMEOUT", 900),
            patch("sova.llama_client.time.time", return_value=now),
            patch("sova.llama_client.subprocess.run") as mock_run,
        ):
            cleanup_idle_services()

        mock_run.assert_not_called()
        assert embedding.exists()


class TestRunEmbeddingCanary:
    def test_sends_canary_requests(self):
        from sova.llama_client import _EMBED_CANARY_REQUESTS, run_embedding_canary

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

        with (
            patch("sova.llama_client._ensure_server", return_value=False),
            pytest.raises(ServerError, match="embedding server not reachable"),
        ):
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
