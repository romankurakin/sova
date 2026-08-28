"""PDF extraction and text processing."""

import bisect
import importlib
import re
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from sova import config
from sova.external import call_external


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
    """Extract markdown from PDF using pymupdf4llm with layout analysis."""
    import pymupdf

    def capture_messages(stream: TextIO):
        previous = getattr(pymupdf, "_g_out_message", sys.stdout)
        pymupdf.set_messages(stream=stream)
        return lambda: pymupdf.set_messages(stream=previous)

    def convert() -> str:
        # pymupdf4llm checks if pymupdf.layout was already imported to decide
        # whether to use layout analysis. Import it first or extraction silently
        # falls back to the lower-quality basic path.
        importlib.import_module("pymupdf.layout")
        import pymupdf4llm

        return pymupdf4llm.to_markdown(str(pdf_path), header=False, footer=False)

    return call_external(
        "PDF extraction",
        convert,
        messages=capture_messages,
    )


_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")
_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")


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
    return headings


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
    return {
        "start_line": int(blocks[0]["start_line"]),
        "end_line": int(blocks[-1]["end_line"]),
        "word_count": len(text.split()),
        "text": text,
    }


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
    section_starts = [line for line, level, _title in _headings(lines) if level < 6]
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
        if len(nonblank) == 1 and _heading(nonblank[0], in_fence=False) is not None:
            # An outline-only parent is already represented in every descendant's.
            # breadcrumb and adds no searchable source content of its own.
            continue
        blocks = _paragraph_blocks(lines, segment_start, segment_end)
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
                split_blocks = [
                    {
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                        "text": chunk["text"],
                    }
                    for chunk in split_chunks
                ]
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


def find_section(sections: list[dict], line: int) -> int | None:
    """Find which section a line belongs to. O(log n) via bisect."""
    if not sections:
        return None
    # Sections are sorted by start_line. Find the rightmost section.
    # whose start_line <= line, then check if line <= end_line.
    idx = bisect.bisect_right([s["start_line"] for s in sections], line) - 1
    if idx < 0:
        return None
    s = sections[idx]
    if line <= (s["end_line"] or float("inf")):
        return idx
    return None
