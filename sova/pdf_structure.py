"""Adapt PyMuPDF's layout output without discarding its source structure."""

import hashlib
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path


@dataclass
class ExtractedDocument:
    markdown: str
    structure: dict


def structure_path(markdown_path: Path) -> Path:
    return markdown_path.with_suffix(".structure.json")


def structure_json(document: ExtractedDocument) -> str:
    return json.dumps(document.structure, ensure_ascii=False, indent=2) + "\n"


def valid_structure(markdown_path: Path) -> bool:
    try:
        record = json.loads(structure_path(markdown_path).read_bytes())
        return (
            record["schema_version"] == 1
            and record["markdown_sha256"]
            == hashlib.sha256(markdown_path.read_bytes()).hexdigest()
        )
    except OSError, ValueError, KeyError, TypeError:
        return False


class _TableParser(HTMLParser):
    """Read the extractor's HTML cells, including their explicit spans."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[dict]] = []
        self.cell: dict | None = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            if not self.rows:
                self.rows.append([])
            self.cell = {
                "text": "",
                "header": tag == "th",
                "rowspan": max(1, int(attributes.get("rowspan") or 1)),
                "colspan": max(1, int(attributes.get("colspan") or 1)),
            }
            self.rows[-1].append(self.cell)
        elif tag in ("br", "p", "div", "li") and self.cell is not None:
            self.cell["text"] += "\n"

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self.cell = None
        elif tag in ("p", "div", "li") and self.cell is not None:
            self.cell["text"] += "\n"

    def handle_data(self, data):
        if self.cell is not None:
            self.cell["text"] += data


def table_markdown(html: str) -> tuple[str, list[dict]]:
    """Expand declared spans; never infer a merged cell from an empty value."""
    parser = _TableParser()
    parser.feed(html)
    grid: dict[tuple[int, int], dict] = {}
    cells = []
    for row, row_cells in enumerate(parser.rows):
        col = 0
        for cell in row_cells:
            while (row, col) in grid:
                col += 1
            cell = cell | {"row": row, "column": col}
            cell["text"] = " ".join(cell["text"].split())
            cells.append(cell)
            # Reject malformed markup instead of silently moving/overwriting cells.
            if cell["rowspan"] > 1000 or cell["colspan"] > 1000:
                raise ValueError("Unreasonable table cell span")
            for r in range(row, row + cell["rowspan"]):
                for c in range(col, col + cell["colspan"]):
                    if (r, c) in grid:
                        raise ValueError("Overlapping table cells")
                    grid[r, c] = cell
            col += cell["colspan"]
    if not grid:
        return html, cells
    height = max(r for r, _ in grid) + 1
    width = max(c for _, c in grid) + 1
    header_rows = 0
    for row in parser.rows:
        if row and all(cell["header"] for cell in row):
            header_rows += 1
        else:
            break

    # A table without explicit headers gets an empty header, not a promoted data row.
    def escaped(text):
        return (
            text.replace("\\", "\\\\")
            .replace("|", "\\|")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def value(row, col):
        return grid.get((row, col), {}).get("text", "")

    headers = []
    for col in range(width):
        labels = list(dict.fromkeys(value(row, col) for row in range(header_rows)))
        headers.append(" / ".join(label for label in labels if label))
    output = [
        "|" + "|".join(escaped(v) for v in headers) + "|",
        "|" + "|".join(["---"] * width) + "|",
    ]
    for row in range(header_rows, height):
        output.append(
            "|" + "|".join(escaped(value(row, col)) for col in range(width)) + "|"
        )
    return "\n".join(output) + "\n\n", cells


def _title_key(text: str) -> str:
    return re.sub(r"[\s*_`]+", "", text).casefold()


def adapt_pages(pages: list[dict], pdf) -> ExtractedDocument:
    """Keep page/block provenance and use PDF outline levels where unambiguous."""
    toc = pdf.get_toc(simple=True)
    output = []
    blocks = []
    offset = 0
    diagnostics = []
    outline_owner = None
    previous_page = None
    for page in pages:
        page_number = page["metadata"]["page_number"]
        if previous_page is not None and page_number != previous_page + 1:
            outline_owner = None
        previous_page = page_number
        text = page["text"]
        cursor = 0
        for box in page["page_boxes"]:
            start, end = box["pos"]
            if not 0 <= cursor <= start <= end <= len(text):
                raise ValueError("Invalid layout-to-Markdown coordinates")
            gap = text[cursor:start]
            output.append(gap)
            offset += len(gap)
            fragment = text[start:end]
            kind = box["class"]
            record = {
                "page": page_number,
                "kind": kind,
                "bbox": list(box["bbox"]),
            }
            if kind in ("title", "section-header"):
                match = re.match(r"^(#{1,6})\s+(.+)", fragment)
                if match:
                    raw_level = len(match[1])
                    record["original_heading_level"] = raw_level
                    matches = [
                        entry
                        for entry in toc
                        if entry[2] == record["page"]
                        and _title_key(entry[1]) == _title_key(match[2])
                    ]
                    levels = {entry[0] for entry in matches}
                    if len(levels) == 1:
                        level = levels.pop()
                        fragment = (
                            "#" * min(6, max(1, level)) + fragment[len(match[1]) :]
                        )
                        record["outline_level"] = level
                        outline_owner = (raw_level, level)
                    elif outline_owner is not None:
                        # Font ranks are relative to the extracted pages, whereas
                        # outline levels are document-wide. Do not mix those scales.
                        raw_owner, owner_level = outline_owner
                        if raw_level < raw_owner:
                            outline_owner = None
                        else:
                            level = min(6, owner_level + max(1, raw_level - raw_owner))
                            fragment = "#" * level + fragment[len(match[1]) :]
            if kind == "table":
                tables = list(
                    re.finditer(
                        r"<table\b[^>]*>.*?</table>",
                        fragment,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                )
                if tables:
                    parts = []
                    last = 0
                    table_cells = []
                    for table in tables:
                        rendered, cells = table_markdown(table[0])
                        parts.extend([fragment[last : table.start()], rendered])
                        table_cells.append(cells)
                        last = table.end()
                    parts.append(fragment[last:])
                    record["tables"] = table_cells
                    record["html"] = fragment
                    fragment = "".join(parts)
                else:
                    diagnostics.append(
                        {
                            "page": record["page"],
                            "kind": "table_markdown_fallback",
                            "bbox": record["bbox"],
                        }
                    )
            if kind == "formula" and not fragment.strip():
                # Keep the omission visible; recognition belongs to the extractor.
                diagnostics.append(
                    {
                        "page": record["page"],
                        "kind": "formula_without_text",
                        "bbox": record["bbox"],
                    }
                )
            record.update(start=offset, end=offset + len(fragment))
            blocks.append(record)
            output.append(fragment)
            offset += len(fragment)
            cursor = end
        output.append(text[cursor:])
        offset += len(text[cursor:])
    markdown = "".join(output)
    return ExtractedDocument(
        markdown,
        {
            "schema_version": 1,
            "markdown_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
            "coordinates": "0-based half-open Unicode characters in Markdown; PDF pages are 1-based",
            "toc": toc,
            "blocks": blocks,
            "diagnostics": diagnostics,
        },
    )
