"""Tests for ollama_client module."""

from unittest.mock import MagicMock, patch


class TestDetectDomain:
    def test_returns_domain_from_llm(self):
        from sova.ollama_client import detect_domain

        with patch("sova.ollama_client.ollama") as mock_ollama:
            mock_response = MagicMock()
            mock_response.message.content = "Computer Architecture"
            mock_ollama.chat.return_value = mock_response

            result = detect_domain("riscv-privileged", ["Trap Handling", "Interrupts"])

            assert result == "Computer Architecture"
            mock_ollama.chat.assert_called_once()

    def test_falls_back_to_doc_name_on_error(self):
        from sova.ollama_client import detect_domain

        with patch("sova.ollama_client.ollama") as mock_ollama:
            import ollama

            mock_ollama.ResponseError = ollama.ResponseError
            mock_ollama.chat.side_effect = ollama.ResponseError("error")

            result = detect_domain("my-document", ["Section 1", "Section 2"])

            assert result == "my-document"

    def test_falls_back_to_doc_name_on_empty_response(self):
        from sova.ollama_client import detect_domain

        with patch("sova.ollama_client.ollama") as mock_ollama:
            mock_response = MagicMock()
            mock_response.message.content = ""
            mock_ollama.chat.return_value = mock_response

            result = detect_domain("my-document", ["Section 1"])

            assert result == "my-document"

    def test_cleans_multiline_response(self):
        from sova.ollama_client import detect_domain

        with patch("sova.ollama_client.ollama") as mock_ollama:
            mock_response = MagicMock()
            mock_response.message.content = "Operating Systems\nThis is extra text"
            mock_ollama.chat.return_value = mock_response

            result = detect_domain("os-book", ["Processes", "Memory"])

            assert result == "Operating Systems"

    def test_truncates_long_domain(self):
        from sova.ollama_client import detect_domain

        with patch("sova.ollama_client.ollama") as mock_ollama:
            mock_response = MagicMock()
            mock_response.message.content = "A" * 100  # Very long response
            mock_ollama.chat.return_value = mock_response

            result = detect_domain("doc", ["Section"])

            assert len(result) <= 60

    def test_uses_all_section_titles_in_prompt(self):
        from sova.ollama_client import detect_domain

        with patch("sova.ollama_client.ollama") as mock_ollama:
            mock_response = MagicMock()
            mock_response.message.content = "Test Domain"
            mock_ollama.chat.return_value = mock_response

            titles = ["Chapter 1", "Chapter 2", "Chapter 3"]
            detect_domain("doc", titles)

            # Check that all titles are in the prompt
            call_args = mock_ollama.chat.call_args
            prompt = call_args[1]["messages"][0]["content"]
            for title in titles:
                assert title in prompt


class TestExpandQuery:
    def test_returns_list_of_terms(self):
        from sova.ollama_client import expand_query

        with patch("sova.ollama_client.ollama") as mock_ollama:
            mock_response = MagicMock()
            mock_response.message.content = "term1\nterm2\nterm3"
            mock_ollama.chat.return_value = mock_response

            result = expand_query("test query")

            assert isinstance(result, list)
            assert len(result) == 3
            assert "term1" in result

    def test_returns_empty_list_on_error(self):
        from sova.ollama_client import expand_query

        with patch("sova.ollama_client.ollama") as mock_ollama:
            import ollama

            mock_ollama.ResponseError = ollama.ResponseError
            mock_ollama.chat.side_effect = ollama.ResponseError("error")

            result = expand_query("test query")

            assert result == []

    def test_limits_to_five_terms(self):
        from sova.ollama_client import expand_query

        with patch("sova.ollama_client.ollama") as mock_ollama:
            mock_response = MagicMock()
            mock_response.message.content = "\n".join([f"term{i}" for i in range(10)])
            mock_ollama.chat.return_value = mock_response

            result = expand_query("test query")

            assert len(result) <= 5

    def test_filters_short_terms(self):
        from sova.ollama_client import expand_query

        with patch("sova.ollama_client.ollama") as mock_ollama:
            mock_response = MagicMock()
            mock_response.message.content = "a\nab\nabc\nabcd"
            mock_ollama.chat.return_value = mock_response

            result = expand_query("test")

            # Only terms with len >= 2 should be included
            assert "a" not in result
            assert "ab" in result
