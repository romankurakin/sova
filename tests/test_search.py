"""Tests for search module."""

import sqlite3

import pytest

from sova.search import (
    _exact_match_bonuses,
    compute_candidates,
    fuse_and_rank,
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
        # "hello 123" = 5 letters / 9 chars.
        density = text_density("hello 123")
        assert 0.5 < density < 0.6

    def test_with_punctuation(self):
        # "hi!" = 2 letters / 3 chars.
        assert text_density("hi!") == pytest.approx(2 / 3)


class TestIsIndexLike:
    def test_normal_text(self):
        text = "This is a normal paragraph with regular content. " * 20
        assert is_index_like(text) is False

    def test_toc_header(self):
        text = "Table of Contents\n1. Chapter 1....10\n2. Chapter 2....25"
        assert is_index_like(text) is True

    def test_low_density_text(self):
        # Simulating index-like content with lots of numbers and dots.
        text = "1.1 ... 10\n1.2 ... 15\n1.3 ... 20\n" * 50
        assert is_index_like(text) is True

    def test_code_block(self):
        # Code has letters, should pass density check.
        text = "def function():\n    return value\n" * 30
        assert is_index_like(text) is False


class TestRRFFusion:
    def test_empty_lists(self):
        assert rrf_fusion([]) == {}
        assert rrf_fusion([[]]) == {}

    def test_single_list(self):
        ranked = [(1, 0.9), (2, 0.8), (3, 0.7)]
        scores = rrf_fusion([ranked])
        # First item: 1/(60+1), second: 1/(60+2), third: 1/(60+3).
        assert scores[1] > scores[2] > scores[3]

    def test_two_lists_same_order(self):
        list1 = [(1, 0.9), (2, 0.8)]
        list2 = [(1, 0.95), (2, 0.85)]
        scores = rrf_fusion([list1, list2])
        # Item 1 appears first in both, should have highest score.
        assert scores[1] > scores[2]

    def test_two_lists_different_order(self):
        list1 = [(1, 0.9), (2, 0.8)]
        list2 = [(2, 0.95), (1, 0.85)]
        scores = rrf_fusion([list1, list2])
        # Both items appear once at rank 1 and once at rank 2.
        assert scores[1] == pytest.approx(scores[2])

    def test_custom_k(self):
        ranked = [(1, 0.9), (2, 0.8)]
        scores_k60 = rrf_fusion([ranked], k=60)
        scores_k10 = rrf_fusion([ranked], k=10)
        # With smaller k, rank differences matter more.
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
        # Item 2 and 3 appear in both lists.
        assert scores[2] > scores[1]  # 2 appears in both, 1 only in first.
        assert scores[2] > scores[4]  # 2 appears in both, 4 only in second.

    def test_non_positive_k_is_clamped(self):
        ranked = [(1, 0.9)]
        scores = rrf_fusion([ranked], k=0)
        assert scores[1] == pytest.approx(0.5)


class TestComputeCandidates:
    def test_small_corpus(self):
        result = compute_candidates(10, 5)
        assert result == 10

    def test_large_corpus(self):
        result = compute_candidates(100_000, 10)
        # Should cap at 1500.
        assert result <= 1500

    def test_scales_with_limit(self):
        small = compute_candidates(1000, 5)
        large = compute_candidates(1000, 50)
        assert large >= small

    def test_minimum_floor(self):
        result = compute_candidates(50, 5)
        assert result >= 50  # at least limit * 4 = 20, but also at least min(50, 150).

    def test_zero_chunks(self):
        result = compute_candidates(0, 10)
        assert result == 0


@pytest.mark.parametrize(
    ("text", "query"),
    [
        ("Ошибка подключения", "ОШИБКА ПОДКЛЮЧЕНИЯ"),
        ("Überprüfung fehlgeschlagen", "ÜBERPRÜFUNG FEHLGESCHLAGEN"),
        ("Ошибка HTTP подключения", "ошибка HTTP подключения"),
    ],
)
def test_unicode_fts_and_exact_bonuses(text, query):
    conn = TestSearchFtsSingleChar._make_fts_db()
    conn.execute("INSERT INTO chunks VALUES (2, ?)", (text,))
    conn.execute("INSERT INTO chunks_fts(rowid, text) VALUES (2, ?)", (text,))
    assert [row[0] for row in search_fts(conn, query, 10)] == [2]
    assert _exact_match_bonuses(conn, [1, 2], query, 0.3, 0.15) == {
        2: pytest.approx(0.45)
    }
    conn.close()


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
        # All single-char tokens get dropped.
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


class TestFuseAndRank:
    @staticmethod
    def _make_db(n_chunks: int = 2):
        conn = sqlite3.connect(":memory:")
        conn.executescript("""
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                path TEXT
            );
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,
                doc_id INTEGER NOT NULL,
                section_id INTEGER,
                start_line INTEGER,
                end_line INTEGER,
                text TEXT NOT NULL,
                is_index INTEGER NOT NULL DEFAULT 0
            );
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                text, content='chunks', content_rowid='id',
                tokenize='porter unicode61'
            );
            INSERT INTO documents (id, name, path) VALUES
                (1, 'doc-a', '/tmp/doc-a.md');
        """)
        chunk_rows = [
            (
                i,
                1,
                i,
                i * 10,
                i * 10 + 5,
                f"timer chunk {i} kernel scheduling",
                0,
            )
            for i in range(1, n_chunks + 1)
        ]
        conn.executemany(
            "INSERT INTO chunks (id, doc_id, section_id, start_line, end_line, text, is_index)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            chunk_rows,
        )
        conn.executemany(
            "INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)",
            [(row[0], row[5]) for row in chunk_rows],
        )
        conn.commit()
        return conn

    def test_fuses_and_diversifies_without_a_second_model(self):
        conn = self._make_db()
        try:
            results, _, _ = fuse_and_rank(
                conn,
                vector_results=[(1, 0.9), (2, 0.8)],
                query_text="timer",
                limit=2,
            )
            assert len(results) == 2
            assert all(
                result["sort_score"] == result["final_score"] for result in results
            )
        finally:
            conn.close()


@pytest.mark.parametrize(
    "query",
    [
        "What does PLIC do?",
        "Что делает PLIC?",
        "PLIC неизвестное_слово",
    ],
)
def test_fts_keeps_identifier_match_despite_absent_query_words(query):
    conn = TestSearchFtsSingleChar._make_fts_db()
    conn.execute("INSERT INTO chunks VALUES (2, 'PLIC handles external interrupts')")
    conn.execute(
        "INSERT INTO chunks_fts(rowid, text) VALUES (2, 'PLIC handles external interrupts')"
    )
    assert [r[0] for r in search_fts(conn, query, 10)] == [2]
    conn.close()


def test_fts_bm25_prefers_multiple_matching_terms():
    conn = TestSearchFtsSingleChar._make_fts_db()
    for chunk_id, text in [
        (2, "virtual memory paging"),
        (3, "virtual storage device"),
        (4, "memory allocation policy"),
    ]:
        conn.execute("INSERT INTO chunks VALUES (?, ?)", (chunk_id, text))
        conn.execute(
            "INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)", (chunk_id, text)
        )
    results = search_fts(conn, "virtual memory explanation", 10)
    assert results[0][0] == 2
    assert {r[0] for r in results} == {2, 3, 4}
    assert search_fts(conn, "VIRTUAL virtual memory explanation", 10) == results
    assert search_fts(conn, "virtual memory explanation", 1) == results[:1]
    conn.close()


def test_fts_treats_user_operators_as_literal_terms():
    conn = TestSearchFtsSingleChar._make_fts_db()
    assert [r[0] for r in search_fts(conn, "missing NOT hello", 10)] == [1]
    assert search_fts(conn, '"missing" OR * NEAR ()', 10) == []
    conn.close()


def test_fts_does_not_hide_a_broken_index():
    conn = sqlite3.connect(":memory:")
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        search_fts(conn, "timer", 10)
    conn.close()


def test_fts_only_hybrid_still_returns_hydrated_results():
    conn = TestFuseAndRank._make_db()
    results, n_vector, n_fts = fuse_and_rank(conn, [], "timer", 2)
    assert len(results) == 2
    assert n_vector == 0
    assert n_fts == 2
    assert all(r["fts_hit"] and r["text"] and r["path"] for r in results)
    assert fuse_and_rank(conn, [], "nonexistent", 2) == ([], 0, 0)
    conn.close()


def test_vector_only_fusion_keeps_rrf_scale():
    from sova import config

    conn = TestFuseAndRank._make_db()
    results, n_vector, n_fts = fuse_and_rank(
        conn, [(1, 0.9), (2, 0.8)], "nonexistent", 2
    )
    assert (n_vector, n_fts) == (2, 0)
    assert results[0]["rrf_score"] == pytest.approx(1 / (config.SEARCH_RRF_K + 1))
    assert results[0]["final_score"] == pytest.approx(
        config.SEARCH_RRF_WEIGHT / (config.SEARCH_RRF_K + 1)
    )
    conn.close()
