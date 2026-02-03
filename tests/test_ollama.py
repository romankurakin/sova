"""Tests for ollama_client module."""

from unittest.mock import MagicMock, patch


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
            assert all(isinstance(emb, list) for emb in result)

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
