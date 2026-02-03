"""LLM-as-Judge for creating ground truth relevance judgments."""

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import ollama
from pydantic import BaseModel

from .search_interface import get_backend

JUDGE_MODEL = "gemma3:27b"

JUDGE_PROMPT = """Rate how well this document chunk answers the search query.

Query: {query}

Document chunk:
---
{chunk}
---

Rules:
- Score ONLY what the chunk actually says. Do not assume or infer content.
- If the query names a specific term, that term must appear in the chunk to score
  above 0. A similar concept from a different domain does not count.
- Definitions, mechanisms, and format descriptions about the query topic are score 2.
  Score 1 is only for passing mentions without explanation.

Scores:
3 = Directly and thoroughly answers the query
2 = Explains, defines, or gives useful detail about the query topic
1 = Mentions a query term in passing without explaining it
0 = No meaningful connection to the query

Return JSON with:
- score: integer 0-3
- confidence: float 0.0-1.0 (lower when uncertain)
- subtopics: list of 1-3 technical concepts this chunk covers
- reason: one sentence; note whether key query terms appear in the chunk"""


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
    # Cross-doc (4) — should return results from multiple documents
    QuerySpec(
        "d01",
        "how ARM and RISC-V differ in exception handling",
        "cross_doc",
        ["arm_exception", "riscv_trap"],
    ),
    QuerySpec(
        "d02",
        "calling convention register usage",
        "cross_doc",
        ["arm_abi", "riscv_abi"],
    ),
    QuerySpec(
        "d03", "interrupt controller setup and priority", "cross_doc", ["gic", "plic"]
    ),
    QuerySpec(
        "d04",
        "page table entry format and permission bits",
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


def judge_chunk(
    query: str, chunk_text: str, max_retries: int = 2
) -> tuple[int, str, float, list[str]]:
    """Judge relevance of a single chunk. Returns (score, reason, confidence, subtopics)."""
    prompt = JUDGE_PROMPT.format(query=query, chunk=chunk_text[:1500])

    for attempt in range(max_retries + 1):
        try:
            response = ollama.chat(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                format=JudgmentResponse.model_json_schema(),
                options={"num_predict": 200, "temperature": 0},
            )
            content = response.message.content or ""
            result = JudgmentResponse.model_validate_json(content)
            return (
                max(0, min(3, result.score)),
                result.reason.strip(),
                max(0.0, min(1.0, result.confidence)),
                result.subtopics[:5] if result.subtopics else [],
            )
        except Exception as e:
            if attempt == max_retries:
                return 0, f"error: {e}", 0.0, []
            time.sleep(0.5)

    return 0, "max_retries", 0.0, []


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
        response = ollama.chat(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": padded_prompt}],
            format=JudgmentResponse.model_json_schema(),
            options={"num_predict": 200, "temperature": 0},
        )
        content = response.message.content or ""
        result = JudgmentResponse.model_validate_json(content)
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


def judge_query(
    spec: QuerySpec,
    k: int = 50,
    verbose: bool = True,
    use_debiasing: bool = False,
    on_chunk_done: Callable[[], None] | None = None,
) -> QueryJudgments:
    """Collect and judge all candidates for a query."""
    candidates = collect_candidates(spec.query, k=k)
    judgments = []

    for hit in candidates:
        if use_debiasing:
            score, reason, confidence, subtopics, _ = judge_chunk_with_debiasing(
                spec.query, hit["text"]
            )
        else:
            score, reason, confidence, subtopics = judge_chunk(spec.query, hit["text"])

        judgments.append(
            Judgment(
                chunk_id=hit["chunk_id"],
                doc=hit["doc"],
                score=score,
                reason=reason,
                confidence=confidence,
                subtopics=subtopics,
                text_preview=hit["text"][:100].replace("\n", " "),
            )
        )
        if on_chunk_done:
            on_chunk_done()

    return QueryJudgments(
        query=spec, judgments=judgments, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
    )


def collect_query_subtopics(judgments: list[Judgment]) -> list[str]:
    """Aggregate subtopics from relevant chunks."""
    subs = set()
    for j in judgments:
        if j.score >= 2 and j.subtopics:
            subs.update(j.subtopics)
    return sorted(subs)
