"""PDF extraction and text processing."""

import re
import warnings
from pathlib import Path

from sova.config import CHUNK_SIZE, DATA_DIR, DOCS_DIR

# Suppress pymupdf layout warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pymupdf")


def find_docs() -> list[dict]:
    """Find all documents (PDFs and extracted markdown files)."""
    pdfs = list(DOCS_DIR.glob("*.pdf")) if DOCS_DIR.exists() else []
    mds = sorted(
        [m for m in DATA_DIR.glob("*.md")] if DATA_DIR.exists() else [],
        key=lambda p: p.name.lower(),
    )
    pdf_names = {p.stem for p in pdfs}

    docs = []
    for pdf in pdfs:
        md = DATA_DIR / f"{pdf.stem}.md"
        docs.append(
            {
                "name": pdf.stem,
                "pdf": pdf,
                "md": md if md.exists() else None,
                "size": pdf.stat().st_size,
            }
        )
    for md in mds:
        if md.stem not in pdf_names:
            docs.append(
                {"name": md.stem, "pdf": None, "md": md, "size": md.stat().st_size}
            )
    return sorted(docs, key=lambda d: (d["size"], d["name"].lower()))


def extract_pdf(pdf_path: Path) -> str:
    """Extract markdown from PDF using pymupdf4llm with layout analysis."""
    import pymupdf.layout  # noqa: F401 - Must import before pymupdf4llm to activate layout
    import pymupdf4llm

    return pymupdf4llm.to_markdown(str(pdf_path), header=False, footer=False)


def parse_sections(lines: list[str]) -> list[dict]:
    """Parse markdown headers into sections."""
    sections = []
    for i, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if match:
            sections.append(
                {
                    "title": match.group(2).strip()[:200],
                    "level": len(match.group(1)),
                    "start_line": i + 1,
                    "end_line": None,
                }
            )
    for i, s in enumerate(sections):
        s["end_line"] = (
            sections[i + 1]["start_line"] - 1 if i + 1 < len(sections) else len(lines)
        )
    return sections


def chunk_text(lines: list[str], target_words: int = CHUNK_SIZE) -> list[dict]:
    """Split text into chunks of approximately target_words."""
    chunks = []
    current_lines, current_words, chunk_start = [], 0, 1

    for i, line in enumerate(lines):
        line_words = len(line.split())
        current_lines.append(line)
        current_words += line_words

        is_break = line.strip() == "" or line.startswith("#")
        if (current_words >= target_words and is_break) or (
            line.startswith("#") and current_words > 50
        ):
            text = "\n".join(current_lines).strip()
            if text and current_words > 10:
                chunks.append(
                    {
                        "start_line": chunk_start,
                        "end_line": i + 1,
                        "word_count": current_words,
                        "text": text,
                    }
                )
            current_lines, current_words, chunk_start = [], 0, i + 2

    if current_lines:
        text = "\n".join(current_lines).strip()
        if text and current_words > 10:
            chunks.append(
                {
                    "start_line": chunk_start,
                    "end_line": len(lines),
                    "word_count": current_words,
                    "text": text,
                }
            )
    return chunks


def find_section(sections: list[dict], line: int) -> int | None:
    """Find which section a line belongs to."""
    for i, s in enumerate(sections):
        if s["start_line"] <= line <= (s["end_line"] or float("inf")):
            return i
    return None
