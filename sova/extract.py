"""PDF extraction and text processing."""

import bisect
import importlib
import importlib.metadata
import re
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from sova import config
from sova.external import call_external
from sova.pdf_structure import ExtractedDocument, adapt_pages


def get_docs_dir() -> Path | None:
    return config.get_docs_dir()


def get_data_dir() -> Path:
    return config.get_data_dir()


# pymupdf emits RuntimeWarnings about unsupported PDF features (fonts, etc.).
# that don't affect extraction quality. Safe to suppress.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pymupdf")


def find_docs() -> list[dict]:
    """Find source documents; generated Markdown is never treated as a source."""
    docs_dir = get_docs_dir()
    data_dir = get_data_dir()
    if not docs_dir or not docs_dir.exists():
        return []
    pdfs = list(docs_dir.glob("*.pdf"))
    source_mds = list(docs_dir.glob("*.md"))
    pdf_names = {p.stem for p in pdfs}

    docs = []
    for pdf in pdfs:
        md = data_dir / f"{pdf.stem}.md"
        docs.append(
            {
                "name": pdf.stem,
                "pdf": pdf,
                "md": md if md.exists() else None,
                "size": pdf.stat().st_size,
            }
        )
    for md in source_mds:
        if md.stem not in pdf_names:
            docs.append(
                {"name": md.stem, "pdf": None, "md": md, "size": md.stat().st_size}
            )
    return sorted(docs, key=lambda d: (d["size"], d["name"].lower()))


def extract_pdf(pdf_path: Path) -> str:
    """Return the searchable text of a structured PDF extraction."""
    return extract_pdf_document(pdf_path).markdown


class _IndraOCR:
    """Indra OCR behind pymupdf4llm's OCR hook.

    The page is rendered without its legible text and Indra reads the rest.
    Its lines become an invisible text layer before layout analysis, as with
    pymupdf4llm's Tesseract backend, so headings, tables and block offsets
    still come from pymupdf-layout. Visible glyphs made layout repeat nearby
    real text as picture text. The engine loads on the first page that needs it.
    """

    def __init__(self) -> None:
        self.engine = None
        self.pages: set[int] = set()

    def __call__(self, page, dpi=150, pixmap=None, language=None, keep_ocr_text=False):
        import numpy as np
        import pymupdf
        from pymupdf4llm.ocr.exec_ocr_interface import (
            FONT,
            FONTNAME,
            REPLACEMENT_UNICODE,
            adjust_width,
            ocr_text,
        )
        from pymupdf4llm.ocr.get_culled_pixmap import get_pixmap

        self.pages.add(page.number + 1)
        displaylist = page.get_displaylist()
        textpage = pymupdf.TextPage(
            displaylist.get_textpage(flags=pymupdf.TEXT_ACCURATE_BBOXES)
        )
        legible, stale = [], []
        for block in textpage.extractDICT()["blocks"]:
            for line in block["lines"]:
                for span in line["spans"]:
                    unreadable = ocr_text(span) or REPLACEMENT_UNICODE in span["text"]
                    (stale if unreadable else legible).append(span["bbox"])
        image, empty = get_pixmap(displaylist, dpi=dpi, rects=legible)
        if empty:
            return
        lines = self.lines(
            np.frombuffer(image.samples, np.uint8).reshape(image.height, image.width, 3)
        )
        if not lines:
            return
        # Earlier OCR text and unreadable glyphs give way to the new reading.
        for bbox in stale:
            page.add_redact_annot(bbox)
        if stale:
            page.apply_redactions(
                images=pymupdf.mupdf.PDF_REDACT_IMAGE_NONE,
                graphics=pymupdf.mupdf.PDF_REDACT_LINE_ART_NONE,
                text=pymupdf.mupdf.PDF_REDACT_TEXT_REMOVE,
            )
        page.insert_font(fontname=FONTNAME, fontbuffer=FONT.buffer)
        to_page = pymupdf.Rect(image.irect).torect(page.rect)
        for box, text in lines:
            rect = pymupdf.Rect(box) * to_page
            if not text.strip():
                continue
            size = rect.height
            page.insert_text(
                rect.bl + (0, -0.2 * size),
                text,
                fontsize=size,
                fontname=FONTNAME,
                morph=(rect.bl, adjust_width(text, size, rect)),
                render_mode=3,
            )

    def lines(self, image) -> list[tuple[list[float], str]]:
        import cv2
        import indra_ocr

        if self.engine is None:
            self.engine = indra_ocr.OrtOCR(indra_ocr.model_dir())
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        page = self.engine(gray)
        # A quarter-turned page comes back with boxes in the turned frame, but
        # pymupdf places text in image coordinates. Read such a page as it is.
        if page.rotation:
            page = self.engine(gray, turn=False)
        return [(line.box, line.text) for line in page.lines]


def extract_pdf_document(pdf_path: Path) -> ExtractedDocument:
    """Use layout blocks, outline and table spans from the installed extractor."""
    import pymupdf

    def capture_messages(stream: TextIO):
        previous = getattr(pymupdf, "_g_out_message", sys.stdout)
        pymupdf.set_messages(stream=stream)
        return lambda: pymupdf.set_messages(stream=previous)

    def convert() -> ExtractedDocument:
        # pymupdf4llm checks if pymupdf.layout was already imported to decide
        # whether to use layout analysis. Import it first or extraction silently
        # falls back to the lower-quality basic path.
        importlib.import_module("pymupdf.layout")
        import pymupdf4llm

        ocr = _IndraOCR()
        options = {
            "header": False,
            "footer": False,
            "page_chunks": True,
            "table_output": "html",
            "ocr_function": ocr,
        }
        with pymupdf.open(pdf_path) as pdf:
            # Images without a text layer are scans. Record them before OCR
            # writes its text into the pages.
            scans = {
                page.number + 1
                for page in pdf
                if page.get_images() and not page.get_text().strip()
            }

            def unread() -> list[int]:
                return [
                    chunk["metadata"]["page_number"]
                    for chunk in pages
                    if chunk["metadata"]["page_number"] in scans
                    and not chunk["text"].strip()
                ]

            pages = pymupdf4llm.to_markdown(pdf, **options)
            # pymupdf4llm's OCR classifier can pass over a scan, which then
            # yields no text at all. Read such pages again with OCR forced.
            forced = [number for number in unread() if number not in ocr.pages]
            if forced:
                retried = pymupdf4llm.to_markdown(
                    pdf, pages=[n - 1 for n in forced], force_ocr=True, **options
                )
                by_page = {c["metadata"]["page_number"]: c for c in retried}
                pages = [by_page.get(c["metadata"]["page_number"], c) for c in pages]
            textless = unread()
            return adapt_pages(
                pages,
                pdf,
                ocr={
                    "engine": f"indra-ocr {importlib.metadata.version('indra-ocr')}",
                    "pages": sorted(ocr.pages),
                    "forced": forced,
                    "textless": textless,
                },
            )

    return call_external(
        "PDF extraction",
        convert,
        messages=capture_messages,
    )


_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")
_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
_NUMBERED_RE = re.compile(
    r"^(?:(?:chapter|section|appendix)\s+)?"
    r"(?P<number>(?:\d+|[A-Z])(?:\.\d+)*)(?:\.\s*|\s+)(?=\S)",
    re.IGNORECASE,
)


def _section_number(title: str) -> tuple[str, ...] | None:
    match = _NUMBERED_RE.match(title)
    if match is None:
        return None
    number = match["number"]
    # A single letter without an explicit Appendix label is too ambiguous.
    if number.isalpha() and not title.lower().startswith("appendix "):
        return None
    return tuple(number.lower().split("."))


def _heading(line: str, *, in_fence: bool) -> tuple[int, str] | None:
    """Return a trustworthy Markdown heading, excluding code-like comments."""
    if in_fence:
        return None
    match = _HEADING_RE.match(line)
    if not match:
        return None

    raw_title = re.sub(r"\s+#+\s*$", "", match.group(2)).strip()
    # PDF conversion can leave assembler/shell comments at column zero. Such.
    # lines often contain another comment marker, inline markup, or dense code.
    # punctuation. Treating them as headings corrupts both boundaries and paths.
    if re.search(r"\s#\s", raw_title) or "<u>" in raw_title.lower():
        return None
    code_punctuation = sum(raw_title.count(char) for char in "%={}[]()")
    if code_punctuation >= 3 and len(raw_title) >= 80:
        return None

    title = raw_title
    for marker in ("**", "__", "*", "_", "`"):
        if title.startswith(marker) and title.endswith(marker):
            title = title[len(marker) : -len(marker)].strip()
            break
    title = " ".join(title.split())
    if not title or not any(char.isalnum() for char in title):
        return None
    return len(match.group(1)), title[:200]


def _headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """Return ``(line, level, title)`` headings outside fenced code."""
    headings: list[tuple[int, int, str]] = []
    fence: str | None = None
    for i, line in enumerate(lines, start=1):
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            marker_char = marker[0]
            if fence is None:
                fence = marker_char
            elif fence == marker_char:
                fence = None
            continue
        parsed = _heading(line, in_fence=fence is not None)
        if parsed is not None:
            level, title = parsed
            headings.append((i, level, title))
    # PDF font sizes frequently flatten numbered sections and their field labels.
    # Use explicit numbering to recover ancestry, without instruction-name lists.
    numbered: dict[tuple[str, ...], tuple[int, int]] = {}
    owner: tuple[tuple[str, ...], int, int] | None = None
    adjusted = []
    for line, raw_level, title in headings:
        level = raw_level
        number = _section_number(title)
        if number is not None:
            parents = [
                key
                for key in numbered
                if len(key) < len(number) and number[: len(key)] == key
            ]
            if parents:
                parent = max(parents, key=len)
                level = max(level, numbered[parent][1] + 1)
            peers = [
                value
                for key, value in numbered.items()
                if key[:-1] == number[:-1] and len(key) == len(number)
            ]
            if peers:
                level = peers[-1][1]
            numbered[number] = (raw_level, level)
            owner = (number, raw_level, level)
        elif owner is not None:
            if raw_level < owner[1]:
                owner = None
            elif len(owner[0]) > 1:
                level = max(level, owner[2] + 1)
        adjusted.append((line, level, title))
    return adjusted


def parse_sections(lines: list[str]) -> list[dict]:
    """Parse Markdown headings into sections with hierarchical breadcrumbs."""
    sections = []
    for line, level, title in _headings(lines):
        sections.append(
            {
                "title": title,
                "level": level,
                "start_line": line,
                "end_line": None,
            }
        )
    for i, s in enumerate(sections):
        next_start = sections[i + 1]["start_line"] if i + 1 < len(sections) else None
        s["end_line"] = next_start - 1 if isinstance(next_start, int) else len(lines)
    return add_section_paths(sections)


def add_section_paths(sections: list[dict]) -> list[dict]:
    """Add deterministic heading breadcrumbs to an ordered section list."""
    ancestors: list[tuple[int, str]] = []
    for section in sections:
        level = int(section["level"])
        title = str(section["title"])
        while ancestors and ancestors[-1][0] >= level:
            ancestors.pop()
        section["path"] = " > ".join(
            [ancestor_title for _, ancestor_title in ancestors] + [title]
        )
        ancestors.append((level, title))
    return sections


def _paragraph_blocks(lines: list[str], start_line: int, end_line: int) -> list[dict]:
    """Create line-preserving paragraph blocks for one structural section."""
    blocks: list[dict] = []
    block_lines: list[str] = []
    block_start = start_line
    for line_number in range(start_line, end_line + 1):
        line = lines[line_number - 1]
        if not block_lines:
            block_start = line_number
        block_lines.append(line)
        if line.strip() == "":
            text = "\n".join(block_lines).strip()
            if text:
                blocks.append(
                    {
                        "start_line": block_start,
                        "end_line": line_number,
                        "text": text,
                    }
                )
            block_lines = []
    if block_lines:
        text = "\n".join(block_lines).strip()
        if text:
            blocks.append(
                {
                    "start_line": block_start,
                    "end_line": end_line,
                    "text": text,
                }
            )
    return blocks


def _table_cells(line: str) -> list[str]:
    """Split Markdown cells without treating escaped or code-span pipes as borders."""
    cells = []
    pending = []
    code = False
    escaped = False
    text = line.strip().removeprefix("|")
    if text.endswith("|"):
        before = text[:-1]
        escapes = len(before) - len(before.rstrip("\\"))
        if escapes % 2 == 0:
            text = before
    for char in text:
        if char == "|" and not escaped and not code:
            cells.append("".join(pending).strip())
            pending = []
        else:
            pending.append(char)
            if char == "`" and not escaped:
                code = not code
        escaped = char == "\\" and not escaped
    cells.append("".join(pending).strip())
    return cells


def _table_blocks(lines: list[str], start: int, end: int) -> list[dict]:
    """Keep pipe tables distinct from adjacent prose, even without blank lines."""
    blocks = []
    cursor = start
    plain_start = start
    fence = None
    while cursor <= end:
        marker = _FENCE_RE.match(lines[cursor - 1])
        if marker:
            char = marker[1][0]
            fence = char if fence is None else (None if char == fence else fence)
        separator = _table_cells(lines[cursor]) if cursor < end else []
        if (
            fence is None
            and cursor < end
            and "|" in lines[cursor - 1]
            and "|" in lines[cursor]
            and separator
            and all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator)
            and len(_table_cells(lines[cursor - 1])) == len(separator)
        ):
            blocks.extend(_paragraph_blocks(lines, plain_start, cursor - 1))
            table_end = cursor + 1
            while (
                table_end < end
                and "|" in lines[table_end]
                and len(_table_cells(lines[table_end])) == len(separator)
            ):
                table_end += 1
            blocks.append(
                {
                    "start_line": cursor,
                    "end_line": table_end,
                    "text": "\n".join(lines[cursor - 1 : table_end]).strip(),
                    "table_header": [cursor, cursor + 1],
                }
            )
            cursor = table_end + 1
            plain_start = cursor
        else:
            cursor += 1
    blocks.extend(_paragraph_blocks(lines, plain_start, end))
    return blocks


def source_ranges(chunk: dict) -> list[list[int]]:
    return chunk.get("source_ranges", [[chunk["start_line"], chunk["end_line"]]])


def _with_ranges(chunk: dict, blocks: list[dict]) -> dict:
    if any("source_ranges" in block for block in blocks):
        chunk["source_ranges"] = [
            span for block in blocks for span in source_ranges(block)
        ]
    return chunk


def _validated_token_counts(
    counter: Callable[[list[str]], list[int]], texts: list[str]
) -> list[int]:
    counts = counter(texts)
    if len(counts) != len(texts):
        raise RuntimeError(
            f"tokenizer returned {len(counts)} counts for {len(texts)} texts"
        )
    if any(not isinstance(count, int) or count < 0 for count in counts):
        raise RuntimeError("tokenizer returned an invalid token count")
    return counts


def _chunk_from_blocks(blocks: list[dict]) -> dict:
    text = "\n\n".join(str(block["text"]) for block in blocks).strip()
    return _with_ranges(
        {
            "start_line": int(blocks[0]["start_line"]),
            "end_line": int(blocks[-1]["end_line"]),
            "word_count": len(text.split()),
            "text": text,
        },
        blocks,
    )


def _fit_block_group(
    blocks: list[dict],
    *,
    exact_count: int,
    target_tokens: int,
    minimum_fill: int,
    count_tokens_batch: Callable[[list[str]], list[int]],
) -> list[dict]:
    """Split a rare over-budget planned group at an exact paragraph boundary."""
    if exact_count <= target_tokens or len(blocks) == 1:
        return [_chunk_from_blocks(blocks)]

    prefix_texts = [
        str(_chunk_from_blocks(blocks[:end])["text"]) for end in range(1, len(blocks))
    ]
    prefix_counts = _validated_token_counts(count_tokens_batch, prefix_texts)
    valid_cuts = [
        end
        for end, count in enumerate(prefix_counts, start=1)
        if minimum_fill <= count <= target_tokens
    ]
    if not valid_cuts:
        # Keep a small heading attached to an indivisible oversized paragraph.
        return [_chunk_from_blocks(blocks)]

    cut = valid_cuts[-1]
    left = blocks[:cut]
    right = blocks[cut:]
    left_count = prefix_counts[cut - 1]
    right_text = str(_chunk_from_blocks(right)["text"])
    right_count = _validated_token_counts(count_tokens_batch, [right_text])[0]
    return _fit_block_group(
        left,
        exact_count=left_count,
        target_tokens=target_tokens,
        minimum_fill=minimum_fill,
        count_tokens_batch=count_tokens_batch,
    ) + _fit_block_group(
        right,
        exact_count=right_count,
        target_tokens=target_tokens,
        minimum_fill=minimum_fill,
        count_tokens_batch=count_tokens_batch,
    )


def _chunk_from_lines(line_blocks: list[dict]) -> dict:
    text = "\n".join(str(block["text"]) for block in line_blocks).strip()
    return {
        "start_line": int(line_blocks[0]["start_line"]),
        "end_line": int(line_blocks[-1]["end_line"]),
        "word_count": len(text.split()),
        "text": text,
    }


def _fit_line_group(
    line_blocks: list[dict],
    *,
    exact_count: int,
    target_tokens: int,
    count_tokens_batch: Callable[[list[str]], list[int]],
) -> list[dict]:
    if exact_count <= target_tokens or len(line_blocks) == 1:
        return [_chunk_from_lines(line_blocks)]
    prefix_texts = [
        str(_chunk_from_lines(line_blocks[:end])["text"])
        for end in range(1, len(line_blocks))
    ]
    prefix_counts = _validated_token_counts(count_tokens_batch, prefix_texts)
    valid_cuts = [
        end
        for end, count in enumerate(prefix_counts, start=1)
        if count <= target_tokens
    ]
    if not valid_cuts:
        # The first source line is itself over budget. Keep that indivisible
        # locator intact, then continue fitting the remaining lines normally.
        first = line_blocks[:1]
        rest = line_blocks[1:]
        rest_count = _validated_token_counts(
            count_tokens_batch, [str(_chunk_from_lines(rest)["text"])]
        )[0]
        return [_chunk_from_lines(first)] + _fit_line_group(
            rest,
            exact_count=rest_count,
            target_tokens=target_tokens,
            count_tokens_batch=count_tokens_batch,
        )
    cut = valid_cuts[-1]
    left = line_blocks[:cut]
    right = line_blocks[cut:]
    right_count = _validated_token_counts(
        count_tokens_batch, [str(_chunk_from_lines(right)["text"])]
    )[0]
    return [_chunk_from_lines(left)] + _fit_line_group(
        right,
        exact_count=right_count,
        target_tokens=target_tokens,
        count_tokens_batch=count_tokens_batch,
    )


def _split_oversized_block(
    block: dict,
    lines: list[str],
    *,
    target_tokens: int,
    count_tokens_batch: Callable[[list[str]], list[int]],
) -> list[dict]:
    """Split a large paragraph or table at source line boundaries."""
    if "table_header" in block:
        return _split_table(block, lines, target_tokens, count_tokens_batch)
    line_blocks = [
        {"start_line": line_number, "end_line": line_number, "text": line}
        for line_number in range(int(block["start_line"]), int(block["end_line"]) + 1)
        if (line := lines[line_number - 1]).strip()
    ]
    if len(line_blocks) <= 1:
        return [block]

    line_counts = _validated_token_counts(
        count_tokens_batch, [str(line_block["text"]) for line_block in line_blocks]
    )
    newline_tokens = _validated_token_counts(count_tokens_batch, ["\n"])[0]
    groups: list[list[dict]] = []
    pending: list[dict] = []
    pending_tokens = 0
    for line_block, line_count in zip(line_blocks, line_counts, strict=True):
        candidate_tokens = pending_tokens + line_count
        if pending:
            candidate_tokens += newline_tokens
        if pending and candidate_tokens > target_tokens:
            groups.append(pending)
            pending = []
            candidate_tokens = line_count
        pending.append(line_block)
        pending_tokens = candidate_tokens
    if pending:
        groups.append(pending)

    exact_counts = _validated_token_counts(
        count_tokens_batch,
        [str(_chunk_from_lines(group)["text"]) for group in groups],
    )
    chunks: list[dict] = []
    for group, exact_count in zip(groups, exact_counts, strict=True):
        chunks.extend(
            _fit_line_group(
                group,
                exact_count=exact_count,
                target_tokens=target_tokens,
                count_tokens_batch=count_tokens_batch,
            )
        )
    return chunks


def _split_table(block, lines, target, counter) -> list[dict]:
    header_start, header_end = block["table_header"]
    header = "\n".join(lines[header_start - 1 : header_end])
    rows = list(range(header_end + 1, block["end_line"] + 1))
    if not rows:
        return [block]
    counts = _validated_token_counts(
        counter, [header, "\n", *[lines[row - 1] for row in rows]]
    )
    header_count, newline_count, *row_counts = counts
    groups = []
    group = []
    size = header_count
    for row, count in zip(rows, row_counts, strict=True):
        if group and size + newline_count + count > target:
            groups.append(group)
            group, size = [], header_count
        group.append(row)
        size += newline_count + count
    groups.append(group)

    def text_for(group):
        return header + "\n" + "\n".join(lines[row - 1] for row in group)

    def fit(group, count):
        if count > target and len(group) > 1:
            mid = len(group) // 2
            halves = [group[:mid], group[mid:]]
            sizes = _validated_token_counts(
                counter, [text_for(part) for part in halves]
            )
            return [
                chunk
                for part, size in zip(halves, sizes, strict=True)
                for chunk in fit(part, size)
            ]
        text = text_for(group)
        return [
            {
                "start_line": header_start if group[0] == rows[0] else group[0],
                "end_line": group[-1],
                "text": text,
                "word_count": len(text.split()),
                "source_ranges": [[header_start, header_end], [group[0], group[-1]]],
            }
        ]

    exact = _validated_token_counts(counter, [text_for(group) for group in groups])
    return [
        chunk
        for group, count in zip(groups, exact, strict=True)
        for chunk in fit(group, count)
    ]


def chunk_text(
    lines: list[str],
    target_tokens: int = config.CHUNK_TARGET_TOKENS,
    *,
    count_tokens_batch: Callable[[list[str]], list[int]],
) -> list[dict]:
    """Split text on structural boundaries using exact model token counts."""
    if not lines or not any(line.strip() for line in lines):
        return []
    target_tokens = max(1, target_tokens)
    minimum_fill = min(96, max(16, target_tokens // 8))
    separator_tokens = _validated_token_counts(count_tokens_batch, ["\n\n"])[0]

    # H1-H5 are hard semantic boundaries. H6 is intentionally soft: PDF
    # converters commonly use it for every field label, table caption, and
    # instruction attribute. Keeping the heading text while allowing adjacent
    # H6 blocks to coalesce avoids tens of thousands of tiny model calls.
    headings = _headings(lines)
    section_starts = [
        line for line, level, title in headings if level < 6 or _section_number(title)
    ]
    segment_starts = sorted({1, *section_starts})
    segments: list[list[dict]] = []

    for segment_index, segment_start in enumerate(segment_starts):
        segment_end = (
            segment_starts[segment_index + 1] - 1
            if segment_index + 1 < len(segment_starts)
            else len(lines)
        )
        nonblank = [
            lines[line_number - 1]
            for line_number in range(segment_start, segment_end + 1)
            if lines[line_number - 1].strip()
        ]
        own_heading = next((h for h in headings if h[0] == segment_start), None)
        next_heading = next((h for h in headings if h[0] > segment_start), None)
        if (
            len(nonblank) == 1
            and own_heading is not None
            and next_heading is not None
            and next_heading[1] > own_heading[1]
        ):
            # An outline-only parent is already represented in every descendant's
            # breadcrumb and adds no searchable source content of its own.
            continue
        blocks = _table_blocks(lines, segment_start, segment_end)
        if blocks:
            segments.append(blocks)

    all_blocks = [block for blocks in segments for block in blocks]
    all_counts = _validated_token_counts(
        count_tokens_batch, [str(block["text"]) for block in all_blocks]
    )
    for block, token_count in zip(all_blocks, all_counts, strict=True):
        block["tokens"] = token_count

    planned_groups: list[list[dict]] = []
    for blocks in segments:
        pending: list[dict] = []
        pending_tokens = 0

        def flush() -> None:
            nonlocal pending, pending_tokens
            if not pending:
                return
            planned_groups.append(pending)
            pending = []
            pending_tokens = 0

        for block in blocks:
            block_tokens = int(block["tokens"])
            if block_tokens > target_tokens:
                attach_heading = bool(pending and pending_tokens < minimum_fill)
                split_budget = (
                    max(1, target_tokens - pending_tokens - separator_tokens)
                    if attach_heading
                    else target_tokens
                )
                split_chunks = _split_oversized_block(
                    block,
                    lines,
                    target_tokens=split_budget,
                    count_tokens_batch=count_tokens_batch,
                )
                split_blocks = [dict(chunk) for chunk in split_chunks]
                if attach_heading:
                    planned_groups.append(pending + [split_blocks[0]])
                    pending = []
                    pending_tokens = 0
                    split_blocks = split_blocks[1:]
                else:
                    flush()
                planned_groups.extend([[split_block] for split_block in split_blocks])
                continue
            candidate_tokens = pending_tokens + block_tokens
            if pending:
                candidate_tokens += separator_tokens
            if (
                pending
                and pending_tokens >= minimum_fill
                and candidate_tokens > target_tokens
            ):
                flush()
                candidate_tokens = block_tokens
            pending.append(block)
            pending_tokens = candidate_tokens
            if pending_tokens >= target_tokens:
                flush()
        flush()

    # The additive planning above minimizes server calls. Verify the actual
    # assembled strings in one batch: tokenization across paragraph boundaries
    # can only be trusted after the final text has been formed.
    exact_counts = _validated_token_counts(
        count_tokens_batch,
        [str(_chunk_from_blocks(group)["text"]) for group in planned_groups],
    )
    chunks: list[dict] = []
    for group, exact_count in zip(planned_groups, exact_counts, strict=True):
        chunks.extend(
            _fit_block_group(
                group,
                exact_count=exact_count,
                target_tokens=target_tokens,
                minimum_fill=minimum_fill,
                count_tokens_batch=count_tokens_batch,
            )
        )
    return chunks


def find_section(
    sections: list[dict], line: int, end_line: int | None = None
) -> int | None:
    """Find a line's section, or the common ancestor covering an inclusive span.

    Section end lines describe their own content, not their whole subtree.
    A chunk spanning soft H6 boundaries therefore belongs to their common
    parent even when the parent's own content ends before the chunk starts.
    """
    if not sections:
        return None
    if end_line is None:
        end_line = line
    if end_line < line:
        raise ValueError("section span ends before it starts")
    # Locate the first and last content sections before comparing ancestry.
    starts = [s["start_line"] for s in sections]
    idx = bisect.bisect_right(starts, line) - 1
    if idx < 0:
        return None
    last_idx = bisect.bisect_right(starts, end_line) - 1
    if line > (sections[idx]["end_line"] or float("inf")) or end_line > (
        sections[last_idx]["end_line"] or float("inf")
    ):
        return None
    if idx == last_idx:
        return idx

    # Crossing a heading closes ancestors of the same or deeper level. The
    # closest preceding shallower heading is the common ancestor. Compare
    # structural positions and levels, not titles: siblings may share a name.
    crossed_level = min(int(s["level"]) for s in sections[idx + 1 : last_idx + 1])
    for ancestor in range(idx, -1, -1):
        if int(sections[ancestor]["level"]) < crossed_level:
            return ancestor
    return None


def context_scope(sections: list[dict], chunk: dict) -> int | None:
    """Use a structural owner, not a neighboring instruction, for disambiguation."""
    index = find_section(sections, chunk["start_line"], chunk["end_line"])
    if index is None:
        return None
    level = float("inf")
    fallback = sections[index]["start_line"]
    found_fallback = False
    for section in reversed(sections[: index + 1]):
        if section["level"] < level:
            level = section["level"]
            number = _section_number(section["title"])
            if number is not None and len(number) > 1:
                return section["start_line"]
            if level < 6 and not found_fallback:
                fallback = section["start_line"]
                found_fallback = True
    return fallback


def context_neighbors(chunks: list[dict], index: int) -> tuple[str, str]:
    """Frozen legacy chunks retain their old inputs; new chunks carry an owner."""
    chunk = chunks[index]
    neighbors = []
    for other in (index - 1, index + 1):
        if (
            not 0 <= other < len(chunks)
            or "context_scope" in chunk
            and (
                chunk["context_scope"] is None
                or chunks[other].get("context_scope") != chunk["context_scope"]
            )
        ):
            neighbors.append("")
        else:
            neighbors.append(chunks[other]["text"])
    return neighbors[0][-500:], neighbors[1][:500]
