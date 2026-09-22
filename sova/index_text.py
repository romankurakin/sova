"""Canonical text representation shared by dense and lexical indexes."""


def contextualized_prefix(
    document: str,
    section_path: str | None,
    context: str | None = None,
) -> str:
    """Build the small prefix stored alongside the source chunk."""
    header = f"[{document}"
    if section_path and section_path.strip():
        header += f" | {section_path.strip()}"
    header += "]"
    parts = [header]
    if context and context.strip():
        parts.append(context.strip())
    return "\n\n".join(parts) + "\n\n"


def contextualized_text(
    document: str,
    section_path: str | None,
    chunk: str,
    context: str | None = None,
) -> str:
    """Build retrieval text while keeping the stored source chunk untouched."""
    return contextualized_prefix(document, section_path, context) + chunk.strip()
