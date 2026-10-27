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
        assert _kwargs["page_chunks"] is True
        assert _kwargs["table_output"] == "html"
        return [
            {
                "text": "# Extracted",
                "metadata": {"page_number": 1},
                "page_boxes": [
                    {
                        "class": "section-header",
                        "pos": [0, 11],
                        "bbox": [0, 0, 100, 100],
                    }
                ],
            }
        ]

    fake.to_markdown = to_markdown
    monkeypatch.setitem(__import__("sys").modules, "pymupdf4llm", fake)
    monkeypatch.setattr("sova.extract.importlib.import_module", lambda _name: None)

    path = tmp_path / "source.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.save(path)
    assert extract_pdf(path) == "# Extracted"
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
    def test_chunk_spanning_soft_headings_uses_common_parent(self):
        lines = [
            "# Parent",
            "",
            " ".join(["intro"] * 60),
            "",
            "###### First field",
            "",
            "First field details.",
            "",
            "###### Second field",
            "",
            "Second field details.",
        ]
        sections = parse_sections(lines)
        chunks = chunk_text(lines, target_tokens=48)
        spanning = next(chunk for chunk in chunks if chunk["start_line"] == 5)

        assert spanning["end_line"] == 11
        assert find_section(sections, 5) == 1
        assert find_section(sections, 5, 11) == 0
        assert find_section(sections, 7, 8) == 1

    @pytest.mark.parametrize(
        ("lines", "start", "end", "expected"),
        [
            (["# Root", "## Child", "text"], 1, 3, 0),
            (["# Root", "## A", "### Detail", "## B", "text"], 3, 5, 0),
            (["# Same", "text", "# Same", "text"], 1, 4, None),
            (["# Root", "###### Same", "text", "###### Same", "text"], 2, 5, 0),
            (["###### A", "text", "###### B", "text"], 1, 4, None),
            (["preface", "# Root", "text"], 1, 3, None),
            (["# Root", "text"], 1, 3, None),
        ],
    )
    def test_span_hierarchy(self, lines, start, end, expected):
        assert find_section(parse_sections(lines), start, end) == expected

    def test_inline_heading_markup_does_not_determine_section_identity(self):
        # Synthetic PDF-style markup must not determine the chunk's ancestry.
        lines = [
            "##### **3.1. Parent (** **`control` ) Register**",
            "###### **3.1.1. First Field in** **`control` Register**",
            "###### **3.1.2. Second Field in** **`control` Register**",
        ]
        sections = parse_sections(lines)
        assert find_section(sections, 2, 3) == 0

    def test_reversed_span_is_rejected(self):
        with pytest.raises(ValueError, match="ends before"):
            find_section(parse_sections(["# Root", "text"]), 2, 1)

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


def test_flat_numbered_instruction_fields_keep_their_owner():
    from sova.extract import context_neighbors, context_scope

    lines = [
        "## Chapter 32. Vector instructions",
        "#### 32.3. Instructions",
        "##### 32.3.21. compression",
        "##### Operation",
        "Compression operation.",
        "##### Included in",
        "Vector extension.",
        "##### 32.3.22. schedule",
        "##### Synopsis",
        "Message schedule.",
    ]
    sections = parse_sections(lines)
    assert sections[4]["path"].endswith("32.3.21. compression > Included in")
    chunks = chunk_text(lines, target_tokens=1000)
    assert len(chunks) == 2
    assert "32.3.21. compression" in chunks[0]["text"]
    assert "32.3.22. schedule" in chunks[1]["text"]
    chunks = [c | {"context_scope": context_scope(sections, c)} for c in chunks]
    assert context_neighbors(chunks, 0) == ("", "")
    assert context_neighbors(chunks, 1) == ("", "")


def test_standalone_sibling_heading_is_not_silently_discarded():
    chunks = chunk_text(["## Empty topic", "", "## Other topic", "Body"])
    assert chunks[0]["text"] == "## Empty topic"


def test_table_continuations_repeat_headers_with_exact_source_ranges():
    from sova.extract import source_ranges

    lines = ["## Registers", "", "| Register | Width |", "| --- | --- |"]
    lines += [f"| r{i} | {i + 1} |" for i in range(12)]
    chunks = chunk_text(lines, target_tokens=40)
    assert len(chunks) > 2
    seen = []
    for chunk in chunks:
        assert "| Register | Width |" in chunk["text"]
        reconstructed = "\n".join(
            "\n".join(lines[a - 1 : b]) for a, b in source_ranges(chunk)
        )
        assert reconstructed.split() == chunk["text"].split()
        assert _token_counts([chunk["text"]])[0] <= 40
        seen.extend(re.findall(r"\br\d+\b", chunk["text"]))
    assert seen == [f"r{i}" for i in range(12)]


def test_oversized_table_row_keeps_header_and_all_cell_text():
    lines = [
        "| Name | Meaning |",
        "| --- | --- |",
        "| long | " + "detail " * 100 + "|",
        "| small | value |",
    ]
    chunks = chunk_text(lines, target_tokens=30)
    assert len(chunks) == 2
    assert chunks[0]["text"].count("detail") == 100
    assert all("| Name | Meaning |" in chunk["text"] for chunk in chunks)


def test_pipe_syntax_in_code_does_not_create_a_table():
    lines = ["```", "| Name |", "| --- |", "| row |", "```"]
    assert all("source_ranges" not in c for c in chunk_text(lines, 8))


def test_empty_table_headers_and_optional_outer_pipes_remain_structural():
    from sova.extract import _table_cells

    assert _table_cells("||Name||") == ["", "Name", ""]
    assert _table_cells(r"| A\|B | C |") == [r"A\|B", "C"]
    for header, separator, row in [
        ("|||", "|---|---|", "|r{0}|description number {0}|"),
        ("Name | Meaning", "--- | ---", "r{0} | description number {0}"),
    ]:
        chunks = chunk_text(
            [header, separator, *[row.format(i) for i in range(10)]], target_tokens=25
        )
        assert len(chunks) > 1
        assert all("source_ranges" in chunk for chunk in chunks)
        assert all(header in chunk["text"] for chunk in chunks)
        assert [
            value for c in chunks for value in re.findall(r"\br\d+\b", c["text"])
        ] == [f"r{i}" for i in range(10)]


_PYMUPDF_SWIG_WARNING = pytest.mark.filterwarnings(
    "ignore:builtin type .* has no __module__ attribute:DeprecationWarning"
)


def _chunk(page: int, text: str) -> dict:
    boxes = [{"class": "text", "pos": [0, len(text)], "bbox": [0, 0, 10, 10]}]
    return {
        "text": text,
        "metadata": {"page_number": page},
        "page_boxes": boxes if text else [],
    }


def _pdf(path: Path, pages: list[str]) -> Path:
    """Write pages that hold text, a dark image like a scan, or nothing."""
    import pymupdf

    with pymupdf.open() as pdf:
        for kind in pages:
            page = pdf.new_page(width=200, height=100)
            if kind == "text":
                page.insert_text((10, 50), "Digital text")
            elif kind == "scan":
                pix = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 40, 20), 0)
                pix.clear_with(255)
                pix.set_rect(pymupdf.IRect(5, 5, 35, 15), (0,))
                page.insert_image(page.rect, pixmap=pix)
        pdf.save(path)
    return path


class _FakeIndra:
    def __init__(self, rotations=(0,)):
        self.rotations = list(rotations)
        self.calls: list[tuple] = []

    def __call__(self, gray, turn=True):
        self.calls.append((gray.ndim, str(gray.dtype), turn))
        rotation = self.rotations.pop(0) if turn and self.rotations else 0
        line = types.SimpleNamespace(
            box=[10, 10, 150, 40], text="Scanned words", conf=0.9
        )
        return types.SimpleNamespace(rotation=rotation, lines=[line])


def _use_fake_indra(monkeypatch, engine) -> list:
    loads = []
    monkeypatch.setattr("indra_ocr.model_dir", lambda: "/models")
    monkeypatch.setattr("indra_ocr.OrtOCR", lambda path: loads.append(path) or engine)
    return loads


def _use_fake_extractor(monkeypatch, to_markdown) -> None:
    monkeypatch.setitem(
        __import__("sys").modules,
        "pymupdf4llm",
        types.SimpleNamespace(to_markdown=to_markdown),
    )
    monkeypatch.setattr("sova.extract.importlib.import_module", lambda _name: None)


@_PYMUPDF_SWIG_WARNING
def test_indra_lines_become_an_invisible_text_layer(monkeypatch, tmp_path):
    import pymupdf
    from pymupdf4llm.ocr.exec_ocr_interface import ocr_text

    from sova.extract import _IndraOCR

    engine = _FakeIndra()
    loads = _use_fake_indra(monkeypatch, engine)
    ocr = _IndraOCR()
    with pymupdf.open(_pdf(tmp_path / "scan.pdf", ["scan"])) as pdf:
        ocr(pdf[0], dpi=72, language="eng", keep_ocr_text=False)
        spans = [
            span
            for block in pdf[0].get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line["spans"]
        ]
    # Layout must see OCR text the way it sees Tesseract's: as an OCR layer.
    assert [span["text"] for span in spans] == ["Scanned words"]
    assert all(ocr_text(span) for span in spans)
    assert loads == ["/models"]
    assert engine.calls == [(2, "uint8", True)]
    assert ocr.pages == {1}


def test_quarter_turned_page_is_reread_in_image_coordinates(monkeypatch):
    import numpy as np

    from sova.extract import _IndraOCR

    engine = _FakeIndra(rotations=[90])
    _use_fake_indra(monkeypatch, engine)
    lines = _IndraOCR().lines(np.zeros((50, 200, 3), dtype=np.uint8))
    assert [call[2] for call in engine.calls] == [True, False]
    assert lines == [([10, 10, 150, 40], "Scanned words")]


@_PYMUPDF_SWIG_WARNING
def test_indra_loads_only_when_a_page_needs_ocr(monkeypatch, tmp_path):
    from sova.extract import extract_pdf_document

    seen = {}

    def to_markdown(_pdf, **kwargs):
        seen.update(kwargs)
        return [_chunk(1, "# Digital")]

    monkeypatch.setattr("indra_ocr.OrtOCR", lambda _path: pytest.fail("loaded"))
    _use_fake_extractor(monkeypatch, to_markdown)
    document = extract_pdf_document(_pdf(tmp_path / "digital.pdf", ["text"]))
    assert callable(seen["ocr_function"])
    assert document.markdown == "# Digital"
    assert document.structure["ocr"]["pages"] == []
    assert document.structure["diagnostics"] == []


@_PYMUPDF_SWIG_WARNING
def test_scan_skipped_by_the_ocr_classifier_is_reread_with_ocr_forced(
    monkeypatch, tmp_path
):
    from sova.extract import extract_pdf_document

    calls = []

    def to_markdown(_pdf, **kwargs):
        calls.append((kwargs.get("pages"), kwargs.get("force_ocr")))
        if kwargs.get("force_ocr"):
            return [_chunk(2, "Recovered scan")]
        return [_chunk(1, "# Digital\n\n"), _chunk(2, "")]

    _use_fake_extractor(monkeypatch, to_markdown)
    document = extract_pdf_document(_pdf(tmp_path / "mixed.pdf", ["text", "scan"]))
    assert calls == [(None, None), ([1], True)]
    assert document.markdown == "# Digital\n\nRecovered scan"
    assert document.structure["ocr"]["forced"] == [2]
    assert document.structure["ocr"]["textless"] == []
    assert document.structure["diagnostics"] == [{"page": 2, "kind": "ocr_forced"}]


@_PYMUPDF_SWIG_WARNING
def test_unreadable_scan_is_reported_and_blank_pages_are_left_alone(
    monkeypatch, tmp_path
):
    from sova.extract import extract_pdf_document

    calls = []

    def to_markdown(_pdf, **kwargs):
        calls.append((kwargs.get("pages"), kwargs.get("force_ocr")))
        if kwargs.get("force_ocr"):
            return [_chunk(2, "")]
        return [_chunk(1, "# Digital\n\n"), _chunk(2, ""), _chunk(3, "")]

    _use_fake_extractor(monkeypatch, to_markdown)
    document = extract_pdf_document(
        _pdf(tmp_path / "mixed.pdf", ["text", "scan", "blank"])
    )
    assert calls == [(None, None), ([1], True)]
    assert document.structure["ocr"]["textless"] == [2]
    assert document.structure["diagnostics"] == [
        {"page": 2, "kind": "ocr_forced"},
        {"page": 2, "kind": "page_without_text"},
    ]


@_PYMUPDF_SWIG_WARNING
def test_ocr_text_lost_by_layout_is_reported_without_a_second_read(
    monkeypatch, tmp_path
):
    import pymupdf4llm.ocr.exec_ocr_interface  # noqa: F401  (real OCR hook)

    from sova.extract import extract_pdf_document

    _use_fake_indra(monkeypatch, _FakeIndra())
    calls = []

    def to_markdown(pdf, **kwargs):
        calls.append(kwargs.get("force_ocr"))
        # OCR writes its lines into the page, but layout keeps none of them.
        kwargs["ocr_function"](pdf[0], dpi=72)
        assert "Scanned words" in pdf[0].get_text()
        return [_chunk(1, "")]

    _use_fake_extractor(monkeypatch, to_markdown)
    document = extract_pdf_document(_pdf(tmp_path / "scan.pdf", ["scan"]))
    assert calls == [None]
    assert document.structure["ocr"]["pages"] == [1]
    assert document.structure["diagnostics"] == [
        {"page": 1, "kind": "page_without_text"}
    ]
