"""Tests for extract module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from sova.extract import chunk_text, find_docs, find_section, parse_sections


class TestParseSections:
    def test_empty_input(self):
        assert parse_sections([]) == []

    def test_single_header(self):
        lines = ["# Introduction", "Some content here"]
        sections = parse_sections(lines)
        assert len(sections) == 1
        assert sections[0]["title"] == "Introduction"
        assert sections[0]["level"] == 1
        assert sections[0]["start_line"] == 1
        assert sections[0]["end_line"] == 2

    def test_multiple_headers(self):
        lines = [
            "# Chapter 1",
            "Content",
            "## Section 1.1",
            "More content",
            "## Section 1.2",
            "Even more",
        ]
        sections = parse_sections(lines)
        assert len(sections) == 3
        assert sections[0]["title"] == "Chapter 1"
        assert sections[0]["level"] == 1
        assert sections[1]["title"] == "Section 1.1"
        assert sections[1]["level"] == 2
        assert sections[2]["title"] == "Section 1.2"
        assert sections[2]["level"] == 2

    def test_header_levels(self):
        lines = ["# H1", "## H2", "### H3", "#### H4", "##### H5", "###### H6"]
        sections = parse_sections(lines)
        assert [s["level"] for s in sections] == [1, 2, 3, 4, 5, 6]

    def test_no_headers(self):
        lines = ["Just some text", "More text", "No headers here"]
        assert parse_sections(lines) == []

    def test_title_truncation(self):
        long_title = "A" * 250
        lines = [f"# {long_title}"]
        sections = parse_sections(lines)
        assert len(sections[0]["title"]) == 200


class TestChunkText:
    def test_empty_input(self):
        assert chunk_text([]) == []

    def test_small_text(self):
        lines = ["Short text"] * 5
        # Too few words, should return empty
        chunks = chunk_text(lines)
        assert chunks == []

    def test_single_chunk(self):
        lines = ["Word " * 100] * 2  # ~200 words
        chunks = chunk_text(lines)
        assert len(chunks) >= 1
        assert chunks[0]["start_line"] == 1

    def test_chunk_at_header(self):
        lines = ["Word " * 60] + ["# New Section"] + ["More words " * 60]
        chunks = chunk_text(lines)
        # Should split at header
        assert len(chunks) >= 1

    def test_chunk_word_count(self):
        lines = ["Hello world"] * 50
        chunks = chunk_text(lines)
        for chunk in chunks:
            assert chunk["word_count"] > 0
            assert "text" in chunk

    def test_respects_target_words(self):
        # Chunks split at blank lines, so include them
        lines = (["Word " * 100] * 5 + [""]) * 4  # 2000 words with breaks
        chunks = chunk_text(lines, target_words=500)
        # Should create multiple chunks at blank line boundaries
        assert len(chunks) >= 2


class TestFindSection:
    def test_empty_sections(self):
        assert find_section([], 10) is None

    def test_line_in_section(self):
        sections = [
            {"start_line": 1, "end_line": 10},
            {"start_line": 11, "end_line": 20},
        ]
        assert find_section(sections, 5) == 0
        assert find_section(sections, 15) == 1

    def test_line_at_boundary(self):
        sections = [
            {"start_line": 1, "end_line": 10},
            {"start_line": 11, "end_line": 20},
        ]
        assert find_section(sections, 1) == 0
        assert find_section(sections, 10) == 0
        assert find_section(sections, 11) == 1

    def test_line_outside_sections(self):
        sections = [{"start_line": 10, "end_line": 20}]
        assert find_section(sections, 5) is None
        assert find_section(sections, 25) is None


class TestFindDocs:
    def test_empty_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir) / "docs"
            data_dir = Path(tmpdir) / "data"
            docs_dir.mkdir()
            data_dir.mkdir()
            with (
                patch("sova.extract.DOCS_DIR", docs_dir),
                patch("sova.extract.DATA_DIR", data_dir),
            ):
                docs = find_docs()
                assert docs == []

    def test_pdf_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir) / "docs"
            data_dir = Path(tmpdir) / "data"
            docs_dir.mkdir()
            data_dir.mkdir()
            (docs_dir / "paper.pdf").write_bytes(b"%PDF-fake")
            with (
                patch("sova.extract.DOCS_DIR", docs_dir),
                patch("sova.extract.DATA_DIR", data_dir),
            ):
                docs = find_docs()
                assert len(docs) == 1
                assert docs[0]["name"] == "paper"
                assert docs[0]["pdf"] is not None
                assert docs[0]["md"] is None

    def test_md_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir) / "docs"
            data_dir = Path(tmpdir) / "data"
            docs_dir.mkdir()
            data_dir.mkdir()
            (data_dir / "notes.md").write_text("# Notes")
            with (
                patch("sova.extract.DOCS_DIR", docs_dir),
                patch("sova.extract.DATA_DIR", data_dir),
            ):
                docs = find_docs()
                assert len(docs) == 1
                assert docs[0]["name"] == "notes"
                assert docs[0]["pdf"] is None
                assert docs[0]["md"] is not None

    def test_pdf_with_extracted_md(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir) / "docs"
            data_dir = Path(tmpdir) / "data"
            docs_dir.mkdir()
            data_dir.mkdir()
            (docs_dir / "paper.pdf").write_bytes(b"%PDF-fake")
            (data_dir / "paper.md").write_text("# Paper")
            with (
                patch("sova.extract.DOCS_DIR", docs_dir),
                patch("sova.extract.DATA_DIR", data_dir),
            ):
                docs = find_docs()
                # PDF and its extracted MD should merge into one entry
                assert len(docs) == 1
                assert docs[0]["name"] == "paper"
                assert docs[0]["pdf"] is not None
                assert docs[0]["md"] is not None

    def test_sorted_by_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir) / "docs"
            data_dir = Path(tmpdir) / "data"
            docs_dir.mkdir()
            data_dir.mkdir()
            (docs_dir / "small.pdf").write_bytes(b"x")
            (docs_dir / "big.pdf").write_bytes(b"x" * 1000)
            with (
                patch("sova.extract.DOCS_DIR", docs_dir),
                patch("sova.extract.DATA_DIR", data_dir),
            ):
                docs = find_docs()
                assert len(docs) == 2
                assert docs[0]["size"] <= docs[1]["size"]
