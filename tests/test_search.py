"""Tests for search module."""

import pytest

from sova.search import (
    compute_candidates,
    is_index_like,
    rrf_fusion,
    search_fts,
    text_density,
)


class TestTextDensity:
    def test_empty_string(self):
        assert text_density("") == 0.0

    def test_all_letters(self):
        assert text_density("abcdef") == 1.0

    def test_all_numbers(self):
        assert text_density("123456") == 0.0

    def test_mixed_content(self):
        # "hello 123" = 5 letters / 9 chars
        density = text_density("hello 123")
        assert 0.5 < density < 0.6

    def test_with_punctuation(self):
        # "hi!" = 2 letters / 3 chars
        assert text_density("hi!") == pytest.approx(2 / 3)


class TestIsIndexLike:
    def test_normal_text(self):
        text = "This is a normal paragraph with regular content. " * 20
        assert is_index_like(text) is False

    def test_toc_header(self):
        text = "Table of Contents\n1. Chapter 1....10\n2. Chapter 2....25"
        assert is_index_like(text) is True

    def test_low_density_text(self):
        # Simulating index-like content with lots of numbers and dots
        text = "1.1 ... 10\n1.2 ... 15\n1.3 ... 20\n" * 50
        assert is_index_like(text) is True

    def test_code_block(self):
        # Code has letters, should pass density check
        text = "def function():\n    return value\n" * 30
        assert is_index_like(text) is False


class TestRRFFusion:
    def test_empty_lists(self):
        assert rrf_fusion([]) == {}
        assert rrf_fusion([[]]) == {}

    def test_single_list(self):
        ranked = [(1, 0.9), (2, 0.8), (3, 0.7)]
        scores = rrf_fusion([ranked])
        # First item: 1/(60+1), second: 1/(60+2), third: 1/(60+3)
        assert scores[1] > scores[2] > scores[3]

    def test_two_lists_same_order(self):
        list1 = [(1, 0.9), (2, 0.8)]
        list2 = [(1, 0.95), (2, 0.85)]
        scores = rrf_fusion([list1, list2])
        # Item 1 appears first in both, should have highest score
        assert scores[1] > scores[2]

    def test_two_lists_different_order(self):
        list1 = [(1, 0.9), (2, 0.8)]
        list2 = [(2, 0.95), (1, 0.85)]
        scores = rrf_fusion([list1, list2])
        # Both items appear once at rank 1 and once at rank 2
        assert scores[1] == pytest.approx(scores[2])

    def test_custom_k(self):
        ranked = [(1, 0.9), (2, 0.8)]
        scores_k60 = rrf_fusion([ranked], k=60)
        scores_k10 = rrf_fusion([ranked], k=10)
        # With smaller k, rank differences matter more
        ratio_k60 = scores_k60[1] / scores_k60[2]
        ratio_k10 = scores_k10[1] / scores_k10[2]
        assert ratio_k10 > ratio_k60

    def test_unique_items(self):
        list1 = [(1, 0.9), (2, 0.8)]
        list2 = [(3, 0.95), (4, 0.85)]
        scores = rrf_fusion([list1, list2])
        assert len(scores) == 4
        assert all(item_id in scores for item_id in [1, 2, 3, 4])

    def test_overlapping_items(self):
        list1 = [(1, 0.9), (2, 0.8), (3, 0.7)]
        list2 = [(2, 0.95), (3, 0.85), (4, 0.75)]
        scores = rrf_fusion([list1, list2])
        # Item 2 and 3 appear in both lists
        assert scores[2] > scores[1]  # 2 appears in both, 1 only in first
        assert scores[2] > scores[4]  # 2 appears in both, 4 only in second

    def test_non_positive_k_is_clamped(self):
        ranked = [(1, 0.9)]
        scores = rrf_fusion([ranked], k=0)
        assert scores[1] == pytest.approx(0.5)


class TestComputeCandidates:
    def test_small_corpus(self):
        # With few chunks, should return at least base_candidates
        result = compute_candidates(10, 5)
        assert result >= 20  # limit * 4

    def test_large_corpus(self):
        result = compute_candidates(100_000, 10)
        # Should cap at 1500
        assert result <= 1500

    def test_scales_with_limit(self):
        small = compute_candidates(1000, 5)
        large = compute_candidates(1000, 50)
        assert large >= small

    def test_minimum_floor(self):
        result = compute_candidates(50, 5)
        assert result >= 50  # at least limit * 4 = 20, but also at least min(50, 150)

    def test_zero_chunks(self):
        result = compute_candidates(0, 10)
        assert result >= 40  # at least limit * 4


class TestSearchFtsSingleChar:
    """Test that single-char tokens are filtered out."""

    @staticmethod
    def _make_fts_db():
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.executescript("""
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY, text TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                text, content='chunks', content_rowid='id',
                tokenize='porter unicode61'
            );
            INSERT INTO chunks (id, text) VALUES (1, 'hello world example');
            INSERT INTO chunks_fts (rowid, text) VALUES (1, 'hello world example');
        """)
        conn.commit()
        return conn

    def test_single_char_query_returns_empty(self):
        conn = self._make_fts_db()
        # All single-char tokens get dropped
        results = search_fts(conn, "a b c", 5)
        assert results == []
        conn.close()

    def test_mixed_query_ignores_short_tokens(self):
        conn = self._make_fts_db()
        results = search_fts(conn, "a hello b", 5)
        assert len(results) > 0
        conn.close()

    def test_empty_query(self):
        conn = self._make_fts_db()
        results = search_fts(conn, "", 5)
        assert results == []
        conn.close()
