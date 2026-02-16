"""LLM-as-Judge for creating ground truth relevance judgments."""

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel

from .search_interface import get_backend


class JudgeError(Exception):
    """Raised when the judge model fails permanently (e.g. model not found)."""


JUDGE_MODEL = "ministral-3-14b-instruct-2512"

# Retry settings
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0


JUDGE_PROMPT = """You are an information retrieval judge. Rate how well this document chunk satisfies the search query. Imagine a developer searched for this query while building an OS kernel — would this chunk help them?

Query: {query}

Document chunk:
---
{chunk}
---

Scoring rules:
- Judge ONLY what the chunk explicitly states. Never infer or assume content beyond the text.
- A chunk can be relevant through exact terminology OR through describing the same concept, mechanism, or procedure the query asks about.
- However, a similar concept from an unrelated domain is NOT relevant (e.g., "Python exception handling" is not relevant to "ARM exception handling").
- A chunk that provides context needed to understand the query topic (e.g., prerequisite concepts, surrounding architecture) is at least score 1.

Score definitions:
3 = Directly and thoroughly addresses the query. A developer would bookmark this.
2 = Provides useful detail — definitions, mechanisms, formats, or procedures related to the query topic.
1 = Partially relevant — mentions the topic briefly, covers a prerequisite, or addresses a related aspect without detail.
0 = Not relevant — no meaningful connection to what the developer is looking for.

Return JSON with:
- score: integer 0-3
- confidence: float 0.0-1.0 (lower when the chunk is borderline between two scores)
- subtopics: list of 1-3 specific technical concepts this chunk covers (e.g., "page table walk", "ELR_EL1 fields", "trap vector setup")
- reason: one sentence explaining your score; mention which query concepts are or are not present in the chunk"""


class JudgmentResponse(BaseModel):
    score: int
    confidence: float = 1.0
    subtopics: list[str] = []
    reason: str


@dataclass
class QuerySpec:
    id: str
    query: str
    category: str
    subtopics: list[str] = field(default_factory=list)


@dataclass
class Judgment:
    chunk_id: int
    doc: str
    score: int
    reason: str
    confidence: float = 1.0
    subtopics: list[str] = field(default_factory=list)
    text_preview: str = ""


@dataclass
class QueryJudgments:
    query: QuerySpec
    judgments: list[Judgment]
    timestamp: str = ""


# 20 queries in 5 categories (4 each)
QUERY_SET: list[QuerySpec] = [
    # Exact lookup (4) — specific terms, BM25 should handle well
    QuerySpec(
        "e01", "ELR_EL1 register", "exact_lookup", ["exception_link", "return_address"]
    ),
    QuerySpec("e02", "mcause CSR", "exact_lookup", ["trap_cause", "exception_code"]),
    QuerySpec(
        "e03", "GICD_ISENABLER", "exact_lookup", ["interrupt_enable", "distributor"]
    ),
    QuerySpec(
        "e04", "sv39 page table", "exact_lookup", ["page_table_entry", "translation"]
    ),
    # Conceptual (4) — needs semantic understanding across sections
    QuerySpec(
        "c01",
        "how the OS reclaims memory from a terminated process",
        "conceptual",
        ["memory_free", "process_exit"],
    ),
    QuerySpec(
        "c02",
        "why kernel code runs in privileged mode",
        "conceptual",
        ["protection", "privilege"],
    ),
    QuerySpec(
        "c03",
        "interrupt handling from hardware signal to handler return",
        "conceptual",
        ["irq_flow", "context_save"],
    ),
    QuerySpec(
        "c04",
        "how virtual addresses get translated to physical",
        "conceptual",
        ["mmu", "page_walk"],
    ),
    # Cross-doc (4) — queries answerable by chunks from multiple documents
    QuerySpec(
        "d01",
        "exception vector table base address register",
        "cross_doc",
        ["vbar_el1", "mtvec"],
    ),
    QuerySpec(
        "d02",
        "function argument passing in registers",
        "cross_doc",
        ["arm_abi", "riscv_abi"],
    ),
    QuerySpec(
        "d03",
        "interrupt priority registers",
        "cross_doc",
        ["gic_priority", "plic_priority"],
    ),
    QuerySpec(
        "d04",
        "page table entry permission bits",
        "cross_doc",
        ["arm_pte", "riscv_pte"],
    ),
    # Natural/vague (4) — real user phrasing, no exact keyword match
    QuerySpec(
        "n01",
        "my kernel crashes right after enabling the MMU",
        "natural",
        ["mmu_enable", "fault"],
    ),
    QuerySpec(
        "n02",
        "context switch is losing register values",
        "natural",
        ["save_restore", "callee_saved"],
    ),
    QuerySpec(
        "n03",
        "how do I set up interrupts from scratch",
        "natural",
        ["irq_setup", "vector_table"],
    ),
    QuerySpec(
        "n04",
        "what order should I initialize hardware on boot",
        "natural",
        ["boot_sequence", "init"],
    ),
    # Negative (4) — should return nothing relevant, tests false positive rate
    QuerySpec("x01", "Python asyncio event loop", "negative", []),
    QuerySpec("x02", "React component lifecycle hooks", "negative", []),
    QuerySpec("x03", "SQL JOIN optimization strategies", "negative", []),
    QuerySpec("x04", "Kubernetes pod scheduling", "negative", []),
]


def _is_permanent_error(exc: Exception) -> bool:
    """Check if an error is permanent (no point retrying)."""
    msg = str(exc).lower()
    return "not found" in msg or "status code: 404" in msg


def _post_json(url: str, payload: dict, timeout: float = 60.0) -> dict:
    """POST JSON to llama-server endpoint."""
    import urllib.request
    import urllib.error

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


_JUDGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "judgment",
        "strict": True,
        "schema": JudgmentResponse.model_json_schema(),
    },
}


def _call_judge(prompt: str) -> JudgmentResponse:
    """Call the judge model via llama-server chat completions API."""
    from sova.config import CONTEXT_SERVER_URL

    resp = _post_json(
        f"{CONTEXT_SERVER_URL}/v1/chat/completions",
        {
            "model": JUDGE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": _JUDGE_RESPONSE_FORMAT,
        },
    )
    content = (resp["choices"][0]["message"]["content"] or "").strip()
    if not content:
        raise ValueError("empty response from model")
    return JudgmentResponse.model_validate_json(content)


def judge_chunk(
    query: str, chunk_text: str, max_retries: int = 2
) -> tuple[int, str, float, list[str]]:
    """Judge relevance of a single chunk. Returns (score, reason, confidence, subtopics).

    Raises JudgeError on permanent failures or after exhausting retries.
    """
    prompt = JUDGE_PROMPT.format(query=query, chunk=chunk_text[:1500])
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            result = _call_judge(prompt)
            return (
                max(0, min(3, result.score)),
                result.reason.strip(),
                max(0.0, min(1.0, result.confidence)),
                result.subtopics[:5] if result.subtopics else [],
            )
        except Exception as e:
            last_error = e
            if _is_permanent_error(e):
                raise JudgeError(str(e)) from e
            if attempt < max_retries:
                time.sleep(_RETRY_BASE_DELAY * (2**attempt))

    raise JudgeError(f"failed after {max_retries + 1} attempts: {last_error}")


def judge_chunk_with_debiasing(
    query: str, chunk_text: str
) -> tuple[int, str, float, list[str], float]:
    """Judge with position bias mitigation. Returns (score, reason, confidence, subtopics, variance)."""
    score1, reason1, conf1, subs1 = judge_chunk(query, chunk_text)

    # Re-judge borderline cases
    if conf1 >= 0.7 and score1 not in {1, 2}:
        return score1, reason1, conf1, subs1, 0.0

    # Re-judge with attention shift
    padding = f"Reference: {uuid.uuid4()}\n\n"
    padded_prompt = padding + JUDGE_PROMPT.format(query=query, chunk=chunk_text[:1400])

    try:
        result = _call_judge(padded_prompt)
        score2 = max(0, min(3, result.score))
        subs2 = result.subtopics[:5] if result.subtopics else []
    except Exception:
        return score1, reason1, conf1, subs1, 0.0

    avg_score = round((score1 + score2) / 2)
    variance = abs(score1 - score2)
    all_subs = list(set(subs1 + subs2))[:5]

    return (
        avg_score,
        reason1,
        (conf1 + max(0.0, min(1.0, result.confidence))) / 2,
        all_subs,
        variance,
    )


def collect_candidates(query: str, k: int = 50) -> list[dict]:
    """Run search to collect candidate chunks for judgment."""
    backend = get_backend()
    results = backend.search(query, limit=k)
    return [
        {
            "chunk_id": r.chunk_id,
            "doc": r.doc,
            "text": r.text,
            "section_id": r.section_id,
        }
        for r in results
    ]


def collect_pool(query: str, k_per_strategy: int = 20) -> list[dict]:
    """Multi-source pooling: union of hybrid, FTS-only, and vector-only results.

    Returns deduplicated candidates (~30-50 per query). This is the TREC
    pooling approach — ground truth is independent of any single retrieval strategy.
    """
    backend = get_backend()

    seen: dict[int, dict] = {}

    def _add_results(results):
        for r in results:
            if r.chunk_id not in seen:
                seen[r.chunk_id] = {
                    "chunk_id": r.chunk_id,
                    "doc": r.doc,
                    "text": r.text,
                    "section_id": r.section_id,
                }

    # Strategy 1: hybrid (existing search)
    _add_results(backend.search(query, limit=k_per_strategy))

    # Strategy 2: BM25-only
    _add_results(backend.search_fts_only(query, limit=k_per_strategy))

    # Strategy 3: vector-only
    _add_results(backend.search_vector_only(query, limit=k_per_strategy))

    return list(seen.values())


def judge_query(
    spec: QuerySpec,
    k: int = 50,
    verbose: bool = True,
    use_debiasing: bool = False,
    on_chunk_done: Callable[[], None] | None = None,
    on_chunk_judged: Callable[["Judgment"], None] | None = None,
    existing_judgments: dict[int, int] | None = None,
    k_per_strategy: int | None = None,
) -> QueryJudgments:
    """Collect and judge all candidates for a query.

    If k_per_strategy is set, uses multi-source pooling (3 strategies).
    Otherwise falls back to single-source collection with limit=k.

    existing_judgments maps chunk_id -> score for chunks already judged.
    These are preserved and not re-judged.

    on_chunk_judged is called after each successful judgment, enabling
    per-chunk checkpointing. Exceptions propagate to the caller.
    """
    if k_per_strategy is not None:
        candidates = collect_pool(spec.query, k_per_strategy=k_per_strategy)
    else:
        candidates = collect_candidates(spec.query, k=k)

    judgments = []

    for hit in candidates:
        chunk_id = hit["chunk_id"]

        # Skip already-judged chunks (incremental)
        if existing_judgments and chunk_id in existing_judgments:
            continue

        if use_debiasing:
            score, reason, confidence, subtopics, _ = judge_chunk_with_debiasing(
                spec.query, hit["text"]
            )
        else:
            score, reason, confidence, subtopics = judge_chunk(spec.query, hit["text"])

        j = Judgment(
            chunk_id=chunk_id,
            doc=hit["doc"],
            score=score,
            reason=reason,
            confidence=confidence,
            subtopics=subtopics,
            text_preview=hit["text"][:100].replace("\n", " "),
        )
        judgments.append(j)
        if on_chunk_judged:
            on_chunk_judged(j)
        if on_chunk_done:
            on_chunk_done()

    return QueryJudgments(
        query=spec, judgments=judgments, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
    )


def judge_single_chunk(
    query: str,
    chunk_id: int,
    text: str,
    doc: str,
    use_debiasing: bool = True,
) -> Judgment:
    """Judge a single chunk. Used by auto-fill during cmd_run."""
    if use_debiasing:
        score, reason, confidence, subtopics, _ = judge_chunk_with_debiasing(
            query, text
        )
    else:
        score, reason, confidence, subtopics = judge_chunk(query, text)

    return Judgment(
        chunk_id=chunk_id,
        doc=doc,
        score=score,
        reason=reason,
        confidence=confidence,
        subtopics=subtopics,
        text_preview=text[:100].replace("\n", " "),
    )


def collect_query_subtopics(judgments: list[Judgment]) -> list[str]:
    """Aggregate subtopics from relevant chunks."""
    subs = set()
    for j in judgments:
        if j.score >= 2 and j.subtopics:
            subs.update(j.subtopics)
    return sorted(subs)
