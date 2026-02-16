"""PDF extraction and text processing."""

import bisect
import re
import warnings
from pathlib import Path

from sova.config import CHUNK_SIZE, DATA_DIR, get_docs_dir

# pymupdf emits RuntimeWarnings about unsupported PDF features (fonts, etc.)
# that don't affect extraction quality. Safe to suppress.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pymupdf")


def find_docs() -> list[dict]:
    """Find all documents (PDFs and extracted markdown files)."""
    docs_dir = get_docs_dir()
    pdfs = list(docs_dir.glob("*.pdf")) if docs_dir and docs_dir.exists() else []
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
    # pymupdf4llm checks if pymupdf.layout was already imported to decide
    # whether to use layout analysis. Must be imported first or it silently
    # falls back to basic extraction with much worse quality.
    import pymupdf.layout  # noqa: F401
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

        # Two conditions to split: (1) reached target size AND at a natural
        # break (blank line or header), or (2) hit a header with enough content
        # (>50 words) to stand alone. This prevents both oversized chunks that
        # degrade embedding quality and tiny fragments under headers.
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
    """Find which section a line belongs to. O(log n) via bisect."""
    if not sections:
        return None
    # Sections are sorted by start_line. Find the rightmost section
    # whose start_line <= line, then check if line <= end_line.
    idx = bisect.bisect_right([s["start_line"] for s in sections], line) - 1
    if idx < 0:
        return None
    s = sections[idx]
    if line <= (s["end_line"] or float("inf")):
        return idx
    return None
