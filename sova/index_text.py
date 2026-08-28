"""Canonical text representation shared by dense and lexical indexes."""


def contextualized_text(
    document: str,
    section_path: str | None,
    chunk: str,
    context: str | None = None,
) -> str:
    """Build retrieval text while keeping the stored source chunk untouched."""
    header = f"[{document}"
    if section_path and section_path.strip():
        header += f" | {section_path.strip()}"
    header += "]"
    parts = [header]
    if context and context.strip():
        parts.append(context.strip())
    parts.append(chunk.strip())
    return "\n\n".join(parts)
