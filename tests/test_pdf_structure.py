"""Structural extraction preserves cell relationships and source locations."""

from sova.extract import parse_sections
from sova.pdf_structure import adapt_pages, table_markdown


def test_html_spans_and_multiple_header_rows_preserve_meaning():
    html = '<table><tr><th rowspan="2">Register</th><th colspan="2">Width</th></tr><tr><th>Bits</th><th>Bytes</th></tr><tr><td rowspan="2">r0</td><td>32</td><td>4</td></tr><tr><td>64</td><td>8</td></tr><tr><td>r1</td><td></td><td>?</td></tr></table>'
    text, cells = table_markdown(html)
    assert text.splitlines() == [
        "|Register|Width / Bits|Width / Bytes|",
        "|---|---|---|",
        "|r0|32|4|",
        "|r0|64|8|",
        "|r1||?|",
        "",
    ]
    assert cells[0]["rowspan"] == 2
    assert cells[1]["colspan"] == 2
    assert next(c for c in cells if c["text"] == "r0")["rowspan"] == 2


def test_html_without_headers_does_not_promote_a_data_row():
    text, _ = table_markdown("<table><tr><td>A | B</td><td>1</td></tr></table>")
    assert text.startswith("|||\n|---|---|\n|A \\| B|1|")


def test_outline_is_used_and_formula_omission_is_reported_with_exact_offsets():
    class PDF:
        def get_toc(self, simple=True):
            return [[2, "Detailed topic", 1]]

    heading = "##### **Detailed topic**\n\n"
    table = "<table><tr><th>Name</th></tr><tr><td>x</td></tr></table>\n\n"
    text = heading + table + "\n\n"
    pages = [
        {
            "metadata": {"page_number": 1},
            "text": text,
            "page_boxes": [
                {
                    "class": "section-header",
                    "pos": [0, len(heading)],
                    "bbox": [0, 0, 50, 20],
                },
                {
                    "class": "table",
                    "pos": [len(heading), len(heading + table)],
                    "bbox": [0, 20, 50, 60],
                },
                {
                    "class": "formula",
                    "pos": [len(heading + table), len(text)],
                    "bbox": [0, 60, 50, 90],
                },
            ],
        }
    ]
    result = adapt_pages(pages, PDF())
    assert result.markdown.startswith("## **Detailed topic**")
    assert result.structure["diagnostics"][0]["kind"] == "formula_without_text"
    table_box = result.structure["blocks"][1]
    assert (
        "|Name|\n|---|\n|x|" in result.markdown[table_box["start"] : table_box["end"]]
    )
    assert result.structure["blocks"][0]["outline_level"] == 2


def test_outline_and_relative_font_levels_share_one_hierarchy():
    class PDF:
        def get_toc(self, simple=True):
            return [[3, "Compression", 1], [3, "Message schedule", 2]]

    pages: list[dict] = []
    for number, headings in enumerate(
        [
            ["# Compression", "# Synopsis"],
            ["# Included in", "# Message schedule", "# Synopsis"],
        ],
        1,
    ):
        text = ""
        boxes = []
        for heading in headings:
            fragment = heading + "\n\n"
            boxes.append(
                {
                    "class": "section-header",
                    "pos": [len(text), len(text + fragment)],
                    "bbox": [0, 0, 100, 20],
                }
            )
            text += fragment
        pages.append(
            {"metadata": {"page_number": number}, "text": text, "page_boxes": boxes}
        )
    result = adapt_pages(pages, PDF())
    sections = parse_sections(result.markdown.splitlines())
    assert [s["path"] for s in sections] == [
        "Compression",
        "Compression > Synopsis",
        "Compression > Included in",
        "Message schedule",
        "Message schedule > Synopsis",
    ]

    # A selection that skips pages cannot inherit the last selected page's owner.
    pages[1]["metadata"]["page_number"] = 4
    result = adapt_pages(pages, PDF())
    assert "\n# Included in\n" in result.markdown


def test_ocr_recoveries_and_textless_pages_are_reported_in_page_order():
    class PDF:
        def get_toc(self, simple=True):
            return []

    pages = [
        {"metadata": {"page_number": n}, "text": text, "page_boxes": []}
        for n, text in [(1, "a\n"), (2, ""), (3, "")]
    ]
    ocr = {
        "engine": "indra-ocr 1.1.2",
        "pages": [2, 3],
        "forced": [3],
        "textless": [2, 3],
    }
    result = adapt_pages(pages, PDF(), ocr=ocr)
    assert result.structure["ocr"] == ocr
    assert result.structure["diagnostics"] == [
        {"page": 2, "kind": "page_without_text"},
        {"page": 3, "kind": "ocr_forced"},
        {"page": 3, "kind": "page_without_text"},
    ]
