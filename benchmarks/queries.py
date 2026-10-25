"""Load and check the search questions in the repository."""

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.corpus import DEFAULT_LOCK, CorpusError, SourceLock

QUERY_DIR = Path(__file__).with_name("queries")
DEFAULT_QUERIES = QUERY_DIR / "queries.json"
DEFAULT_PROBES = QUERY_DIR / "language-probes.json"


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source_id: str
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    pdf_page: int = Field(gt=0)
    section: str = Field(min_length=1)


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(pattern=r"^(q[0-9]+|probe-(ru|zh)-q[0-9]+)$")
    text: str = Field(min_length=1)
    language: Literal["en", "ru", "zh"]
    intent: str = Field(min_length=1)
    category: Literal[
        "conceptual",
        "exact_lookup",
        "comparison",
        "problem_description",
        "procedure",
    ]
    topic: Literal["instructions", "privileged", "os"]
    comparison_group: str = Field(min_length=1)
    architecture_scope: Literal["both", "riscv", "loongarch", "x86", "unspecified"]
    query_style: Literal["question", "short_search_request"]
    source_refs: list[SourceReference] = Field(min_length=1)
    anchor_id: str | None = None
    anchor_text_sha256: str | None = None

    def judge_input(self) -> dict[str, str]:
        """Return only the fields that the judge sees: no references or hypotheses."""
        return {"id": self.id, "text": self.text, "intent": self.intent}


class QueryBank(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[7]
    corpus_id: str
    source_lock_sha256: str
    editor_revision: int = Field(gt=0)
    updated: str
    notes: list[str]
    queries: list[Query] = Field(min_length=1)


def load_query_banks(
    *,
    core_path: Path = DEFAULT_QUERIES,
    probes_path: Path = DEFAULT_PROBES,
    lock_path: Path = DEFAULT_LOCK,
) -> tuple[QueryBank, QueryBank]:
    """Check both question banks, their source references and translation anchors."""
    try:
        lock_bytes = lock_path.read_bytes()
        lock = SourceLock.model_validate_json(lock_bytes)
        banks = tuple(
            QueryBank.model_validate_json(path.read_bytes())
            for path in (core_path, probes_path)
        )
        sources = {file.source_id: file for file in lock.files}
        all_ids: set[str] = set()
        for bank in banks:
            if (
                bank.corpus_id != lock.corpus_id
                or bank.source_lock_sha256 != hashlib.sha256(lock_bytes).hexdigest()
            ):
                raise ValueError("query bank does not match the pinned source lock")
            texts: set[str] = set()
            for query in bank.queries:
                text = " ".join(query.text.casefold().split())
                if query.id in all_ids or not text or text in texts:
                    raise ValueError(f"duplicate/empty query: {query.id}")
                all_ids.add(query.id)
                texts.add(text)
                for ref in query.source_refs:
                    source = sources.get(ref.source_id)
                    if (
                        source is None
                        or ref.source_sha256 != source.sha256
                        or ref.pdf_page > source.pages
                    ):
                        raise ValueError(f"invalid source reference: {query.id}")
                # A query about both ISAs needs a source for each of them.
                if query.architecture_scope == "both":
                    refs = {ref.source_id for ref in query.source_refs}
                    if "loongarch-vol1" not in refs or not refs.intersection(
                        {"riscv-privileged", "riscv-unprivileged"}
                    ):
                        raise ValueError(
                            f"both-architecture support missing: {query.id}"
                        )
        core, probes = banks
        anchors = {q.id: q for q in core.queries}
        if any(q.language != "en" or q.anchor_id is not None for q in core.queries):
            raise ValueError("core bank must contain English queries")
        translated: dict[str, set[str]] = {"ru": set(), "zh": set()}
        for query in probes.queries:
            anchor = anchors.get(query.anchor_id or "")
            if query.language not in translated or anchor is None:
                raise ValueError(f"invalid translation anchor: {query.id}")
            if anchor.id in translated[query.language]:
                raise ValueError(f"duplicate translation anchor: {query.id}")
            translated[query.language].add(anchor.id)
            if (
                query.anchor_text_sha256
                != hashlib.sha256(anchor.text.encode()).hexdigest()
                or query.comparison_group != anchor.comparison_group
                or query.intent != anchor.intent
                or query.category != anchor.category
                or query.architecture_scope != anchor.architecture_scope
            ):
                raise ValueError(f"stale translation anchor: {query.id}")
        if translated["ru"] != translated["zh"]:
            raise ValueError("Russian and Chinese probes must share English anchors")
        return core, probes
    except (OSError, ValueError) as exc:
        raise CorpusError(f"Invalid query bank: {exc}") from exc


def load_evaluation_queries() -> list[Query]:
    """Return the questions in all languages."""
    core, probes = load_query_banks()
    return core.queries + probes.queries


def questions_markdown(
    core: QueryBank,
    lock_path: Path = DEFAULT_LOCK,
    *,
    probes: QueryBank | None = None,
) -> str:
    """Make a reading list with links to the original PDFs, not to local paths."""
    lock = SourceLock.model_validate_json(lock_path.read_bytes())
    sources = {source.source_id: source for source in lock.files}
    titles = {
        "instructions": "Instructions, atomics and memory ordering",
        "privileged": "Privileges, exceptions and address translation",
        "os": "Operating systems, xv6, files, networking and security",
    }
    lines = [
        "# Search questions",
        "",
        (
            f"{len(core.queries)} English queries; editorial revision {core.editor_revision}. "
            "The editor checked every source reference; independent review is pending."
        ),
        "",
        (
            "Generated from [queries.json](queries.json). References use 1-based PDF pages; "
            "they are authoring aids, not relevance labels or exclusive answer locations."
        ),
        "",
    ]
    translations: dict[str, list[Query]] = {}
    if probes is not None:
        for query in probes.queries:
            translations.setdefault(query.anchor_id or "", []).append(query)
        lines.extend(
            [
                (
                    f"{len(probes.queries)} translations included by default; editorial revision "
                    f"{probes.editor_revision}. Generated from "
                    "[language-probes.json](language-probes.json) and shown below their "
                    "English anchors. They share the anchor's intent and are scored "
                    "separately by language. Independent translation review is pending; "
                    "this does not disable any queries."
                ),
                "",
            ]
        )
    for topic, title in titles.items():
        lines.extend([f"## {title}", ""])
        for query in core.queries:
            if query.topic != topic:
                continue
            lines.extend(
                [
                    f"### {query.id} — {query.text}",
                    "",
                    f"Intent: {query.intent}",
                    "",
                    (
                        f"Category: `{query.category}` · ISA: `{query.architecture_scope}` · "
                        f"Comparison group: `{query.comparison_group}`"
                    ),
                    "",
                ]
            )
            for ref in query.source_refs:
                lines.append(
                    f"- [{ref.source_id}, PDF p. {ref.pdf_page}]"
                    f"({sources[ref.source_id].url}#page={ref.pdf_page}) — {ref.section}"
                )
            lines.append("")
            for translation in sorted(
                translations.get(query.id, []), key=lambda item: item.language
            ):
                lines.extend(
                    [
                        f"#### {translation.id} ({translation.language.upper()})",
                        "",
                        translation.text,
                        "",
                    ]
                )
    return "\n".join(lines)
