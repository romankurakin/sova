"""Tests for ollama_client module."""

from unittest.mock import MagicMock, patch

import ollama as ollama_lib


class TestCheckOllama:
    def test_success_model_present(self):
        from sova.ollama_client import check_ollama

        with patch("sova.ollama_client.ollama") as mock_ollama:
            mock_embed = MagicMock()
            mock_embed.model = "qwen3-embedding:4b"
            mock_context = MagicMock()
            mock_context.model = "gemma3:12b"
            mock_list = MagicMock()
            mock_list.models = [mock_embed, mock_context]
            mock_ollama.list.return_value = mock_list

            ok, msg = check_ollama()

            assert ok is True
            assert msg == "ready"
            mock_ollama.pull.assert_not_called()

    def test_success_model_missing_pulls(self):
        from sova.ollama_client import check_ollama

        with patch("sova.ollama_client.ollama") as mock_ollama:
            mock_model = MagicMock()
            mock_model.model = "some-other-model:latest"
            mock_list = MagicMock()
            mock_list.models = [mock_model]
            mock_ollama.list.return_value = mock_list

            ok, msg = check_ollama()

            assert ok is True
            # Both embedding and context models should be pulled
            assert mock_ollama.pull.call_count == 2

    def test_response_error(self):
        from sova.ollama_client import check_ollama

        with patch("sova.ollama_client.ollama") as mock_ollama:
            mock_ollama.ResponseError = ollama_lib.ResponseError
            mock_ollama.list.side_effect = ollama_lib.ResponseError("model not found")

            ok, msg = check_ollama()

            assert ok is False
            assert "model not found" in msg

    def test_connection_error(self):
        from sova.ollama_client import check_ollama

        with patch("sova.ollama_client.ollama") as mock_ollama:
            mock_ollama.ResponseError = ollama_lib.ResponseError
            mock_ollama.list.side_effect = ConnectionError("refused")

            ok, msg = check_ollama()

            assert ok is False
            assert "not running" in msg


class TestGetQueryEmbedding:
    def test_adds_instruction_prefix(self):
        from sova.ollama_client import get_query_embedding

        with patch("sova.ollama_client.ollama") as mock_ollama:
            mock_response = MagicMock()
            mock_response.embeddings = [[0.1, 0.2, 0.3]]
            mock_ollama.embed.return_value = mock_response

            get_query_embedding("test query")

            call_args = mock_ollama.embed.call_args
            prompt = call_args[1]["input"]
            assert "Instruct:" in prompt
            assert "Query: test query" in prompt

    def test_returns_float_list(self):
        from sova.ollama_client import get_query_embedding

        with patch("sova.ollama_client.ollama") as mock_ollama:
            mock_response = MagicMock()
            mock_response.embeddings = [[0.1, 0.2, 0.3]]
            mock_ollama.embed.return_value = mock_response

            result = get_query_embedding("test")

            assert isinstance(result, list)
            assert all(isinstance(v, float) for v in result)


class TestGetEmbeddingsBatch:
    def test_returns_list_of_embeddings(self):
        from sova.ollama_client import get_embeddings_batch

        with patch("sova.ollama_client.ollama") as mock_ollama:
            mock_response = MagicMock()
            mock_response.embeddings = [[0.1, 0.2], [0.3, 0.4]]
            mock_ollama.embed.return_value = mock_response

            result = get_embeddings_batch(["text1", "text2"])

            assert len(result) == 2
            assert all(isinstance(emb, bytes) for emb in result)

    def test_no_instruction_prefix(self):
        from sova.ollama_client import get_embeddings_batch

        with patch("sova.ollama_client.ollama") as mock_ollama:
            mock_response = MagicMock()
            mock_response.embeddings = [[0.1, 0.2]]
            mock_ollama.embed.return_value = mock_response

            get_embeddings_batch(["test text"])

            call_args = mock_ollama.embed.call_args
            prompt = call_args[1]["input"]
            # Document embeddings don't have instruction prefix
            assert "Instruct:" not in str(prompt)


class TestGenerateContext:
    def test_returns_stripped_string(self):
        from sova.ollama_client import generate_context

        with patch("sova.ollama_client.ollama") as mock_ollama:
            mock_message = MagicMock()
            mock_message.content = "  This chunk covers authentication flows.  "
            mock_response = MagicMock()
            mock_response.message = mock_message
            mock_ollama.chat.return_value = mock_response

            result = generate_context("doc1", "Auth", "chunk text here")

            assert result == "This chunk covers authentication flows."

    def test_prompt_contains_doc_and_section(self):
        from sova.ollama_client import generate_context

        with patch("sova.ollama_client.ollama") as mock_ollama:
            mock_message = MagicMock()
            mock_message.content = "Context."
            mock_response = MagicMock()
            mock_response.message = mock_message
            mock_ollama.chat.return_value = mock_response

            generate_context("my-doc", "Introduction", "some text")

            call_args = mock_ollama.chat.call_args
            prompt = call_args[1]["messages"][0]["content"]
            assert "my-doc" in prompt
            assert "Introduction" in prompt

    def test_none_section_uses_placeholder(self):
        from sova.ollama_client import generate_context

        with patch("sova.ollama_client.ollama") as mock_ollama:
            mock_message = MagicMock()
            mock_message.content = "Context."
            mock_response = MagicMock()
            mock_response.message = mock_message
            mock_ollama.chat.return_value = mock_response

            generate_context("doc1", None, "text")

            call_args = mock_ollama.chat.call_args
            prompt = call_args[1]["messages"][0]["content"]
            assert "(no section)" in prompt

    def test_surrounding_text_included(self):
        from sova.ollama_client import generate_context

        with patch("sova.ollama_client.ollama") as mock_ollama:
            mock_message = MagicMock()
            mock_message.content = "Context."
            mock_response = MagicMock()
            mock_response.message = mock_message
            mock_ollama.chat.return_value = mock_response

            generate_context("doc1", "Sec", "main", "prev text", "next text")

            call_args = mock_ollama.chat.call_args
            prompt = call_args[1]["messages"][0]["content"]
            assert "prev text" in prompt
            assert "next text" in prompt

    def test_empty_surrounding_uses_placeholders(self):
        from sova.ollama_client import generate_context

        with patch("sova.ollama_client.ollama") as mock_ollama:
            mock_message = MagicMock()
            mock_message.content = "Context."
            mock_response = MagicMock()
            mock_response.message = mock_message
            mock_ollama.chat.return_value = mock_response

            generate_context("doc1", "Sec", "text", "", "")

            call_args = mock_ollama.chat.call_args
            prompt = call_args[1]["messages"][0]["content"]
            assert "(start of document)" in prompt
            assert "(end of document)" in prompt

    def test_uses_context_model(self):
        from sova.ollama_client import generate_context

        with patch("sova.ollama_client.ollama") as mock_ollama:
            mock_message = MagicMock()
            mock_message.content = "Context."
            mock_response = MagicMock()
            mock_response.message = mock_message
            mock_ollama.chat.return_value = mock_response

            generate_context("doc1", "Sec", "text")

            call_args = mock_ollama.chat.call_args
            assert call_args[1]["model"] == "gemma3:12b"
