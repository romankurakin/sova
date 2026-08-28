"""Tests for extract module."""

import re
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from sova.extract import (
    chunk_text as _chunk_text,
)
from sova.extract import (
    extract_pdf,
    find_docs,
    find_section,
    parse_sections,
)


def _token_counts(texts: list[str]) -> list[int]:
    return [len(re.findall(r"\w+|[^\w\s]", text)) for text in texts]


def chunk_text(lines: list[str], target_tokens: int = 768) -> list[dict]:
    return _chunk_text(
        lines,
        target_tokens=target_tokens,
        count_tokens_batch=_token_counts,
    )


@pytest.mark.filterwarnings(
    "ignore:builtin type .* has no __module__ attribute:DeprecationWarning"
)
def test_extract_pdf_suppresses_parser_messages(monkeypatch, capsys, tmp_path):
    import pymupdf

    fake = types.SimpleNamespace()

    def to_markdown(*_args, **_kwargs):
        print("raw converter output")
        pymupdf.message("=== Document parser messages ===")
        pymupdf.message("Using Tesseract for OCR processing.")
        return "# Extracted"

    fake.to_markdown = to_markdown
    monkeypatch.setitem(__import__("sys").modules, "pymupdf4llm", fake)
    monkeypatch.setattr("sova.extract.importlib.import_module", lambda _name: None)

    assert extract_pdf(tmp_path / "source.pdf") == "# Extracted"
    assert capsys.readouterr() == ("", "")


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

    def test_optional_closing_hashes_are_supported(self):
        sections = parse_sections(["## Release notes ##", "content"])
        assert sections[0]["title"] == "Release notes"

    def test_hash_inside_a_real_title_is_supported(self):
        sections = parse_sections(["# C# Language Guide", "content"])
        assert sections[0]["title"] == "C# Language Guide"

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
        chunks = chunk_text(lines)
        assert len(chunks) == 1
        assert chunks[0]["text"].count("Short text") == 5

    def test_single_chunk(self):
        lines = ["Word " * 100] * 2  # ~200 words.
        chunks = chunk_text(lines)
        assert len(chunks) >= 1
        assert chunks[0]["start_line"] == 1

    def test_chunk_at_header(self):
        lines = ["Word " * 60] + ["# New Section"] + ["More words " * 60]
        chunks = chunk_text(lines)
        # Should split at header.
        assert len(chunks) >= 1

    def test_chunk_word_count(self):
        lines = ["Hello world"] * 50
        chunks = chunk_text(lines)
        for chunk in chunks:
            assert chunk["word_count"] > 0
            assert "text" in chunk

    def test_respects_target_tokens(self):
        # Chunks split at blank lines, so include them.
        lines = (["Word " * 100] * 5 + [""]) * 4  # 2000 words with breaks.
        chunks = chunk_text(lines, target_tokens=500)
        # Should create multiple chunks at blank line boundaries.
        assert len(chunks) >= 2

    def test_indivisible_long_line_does_not_pull_following_lines_over_budget(self):
        lines = ["x" * 100, "short line"]

        chunks = _chunk_text(
            lines,
            target_tokens=20,
            count_tokens_batch=lambda texts: [len(text) for text in texts],
        )

        assert [(chunk["start_line"], chunk["end_line"]) for chunk in chunks] == [
            (1, 1),
            (2, 2),
        ]

    def test_heading_starts_its_own_structural_chunk(self):
        lines = [
            "Tail of the previous section.",
            "",
            "## Medium any code model",
            "The medany model addresses data relative to the program counter.",
        ]
        chunks = chunk_text(lines, target_tokens=10_000)
        assert len(chunks) == 2
        assert chunks[0]["text"] == "Tail of the previous section."
        assert chunks[1]["text"].startswith("## Medium any code model")

    def test_deep_field_labels_are_soft_boundaries(self):
        lines = [
            "##### Instruction",
            "intro",
            "###### Purpose",
            "purpose text",
            "###### Attributes",
            "attribute text",
        ]

        chunks = chunk_text(lines, target_tokens=10_000)

        assert len(chunks) == 1
        assert "###### Purpose" in chunks[0]["text"]
        assert "###### Attributes" in chunks[0]["text"]

    def test_heading_is_not_orphaned_from_a_large_first_paragraph(self):
        lines = ["## Large section", "", "detail " * 100]
        chunks = chunk_text(lines, target_tokens=20)
        assert len(chunks) == 1
        assert chunks[0]["text"].startswith("## Large section\n\n")

    def test_outline_only_parent_is_carried_by_child_path_not_indexed(self):
        lines = ["# Parent", "", "## Child", "", "Useful child content."]
        chunks = chunk_text(lines)
        assert len(chunks) == 1
        assert chunks[0]["text"].startswith("## Child")
        sections = parse_sections(lines)
        child = find_section(sections, chunks[0]["start_line"])
        assert child is not None
        assert sections[child]["path"] == "Parent > Child"

    def test_assembler_comments_are_not_headings(self):
        lines = [
            "# Smallest negative number: lui a0, 0x80000 # a0 = 0xffffffff80000000 addi a0, a0, -0x800",
            "",
            "#### **5.2. Medium any code model**",
            "The medium any code model uses PC-relative addressing.",
            "# Calculate address .Ltmp2: auipc a0, %pcrel_hi(symbol) addi a0, a0, %pcrel_lo(.Ltmp2)",
        ]
        sections = parse_sections(lines)
        assert [section["title"] for section in sections] == [
            "5.2. Medium any code model"
        ]
        chunks = chunk_text(lines, target_tokens=10_000)
        assert len(chunks) == 2
        assert "Smallest negative" in chunks[0]["text"]
        assert "Calculate address" in chunks[1]["text"]

    def test_fenced_code_headings_are_not_sections(self):
        lines = ["# API", "```asm", "# comment", "```", "## Details", "text"]
        sections = parse_sections(lines)
        assert [section["title"] for section in sections] == ["API", "Details"]

    def test_decorative_parser_icons_are_not_sections(self):
        lines = ["# Chapter", "# ", "## **Attributes**", "text"]
        sections = parse_sections(lines)
        assert [section["path"] for section in sections] == [
            "Chapter",
            "Chapter > Attributes",
        ]

    def test_section_paths_follow_heading_hierarchy(self):
        lines = ["# Chapter", "## Section", "### Detail", "## Sibling"]
        sections = parse_sections(lines)
        assert [section["path"] for section in sections] == [
            "Chapter",
            "Chapter > Section",
            "Chapter > Section > Detail",
            "Chapter > Sibling",
        ]

    def test_uses_the_injected_token_counter(self):
        calls: list[list[str]] = []

        def counter(texts: list[str]) -> list[int]:
            calls.append(texts)
            return [len(text) for text in texts]

        chunks = _chunk_text(["hello"], count_tokens_batch=counter)

        assert chunks[0]["text"] == "hello"
        assert calls


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
                patch("sova.extract.get_docs_dir", return_value=docs_dir),
                patch("sova.extract.get_data_dir", return_value=data_dir),
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
                patch("sova.extract.get_docs_dir", return_value=docs_dir),
                patch("sova.extract.get_data_dir", return_value=data_dir),
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
            (docs_dir / "notes.md").write_text("# Notes")
            with (
                patch("sova.extract.get_docs_dir", return_value=docs_dir),
                patch("sova.extract.get_data_dir", return_value=data_dir),
            ):
                docs = find_docs()
                assert len(docs) == 1
                assert docs[0]["name"] == "notes"
                assert docs[0]["pdf"] is None
                assert docs[0]["md"] is not None

    def test_generated_markdown_without_source_is_not_a_document(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir) / "docs"
            data_dir = Path(tmpdir) / "data"
            docs_dir.mkdir()
            data_dir.mkdir()
            (data_dir / "deleted-source.md").write_text("# Stale")
            with (
                patch("sova.extract.get_docs_dir", return_value=docs_dir),
                patch("sova.extract.get_data_dir", return_value=data_dir),
            ):
                assert find_docs() == []

    def test_pdf_with_extracted_md(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir) / "docs"
            data_dir = Path(tmpdir) / "data"
            docs_dir.mkdir()
            data_dir.mkdir()
            (docs_dir / "paper.pdf").write_bytes(b"%PDF-fake")
            (data_dir / "paper.md").write_text("# Paper")
            with (
                patch("sova.extract.get_docs_dir", return_value=docs_dir),
                patch("sova.extract.get_data_dir", return_value=data_dir),
            ):
                docs = find_docs()
                # PDF and its extracted MD should merge into one entry.
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
                patch("sova.extract.get_docs_dir", return_value=docs_dir),
                patch("sova.extract.get_data_dir", return_value=data_dir),
            ):
                docs = find_docs()
                assert len(docs) == 2
                assert docs[0]["size"] <= docs[1]["size"]
