"""LLM-as-Judge for creating ground truth relevance judgments."""

import time
import uuid
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

Score definitions with examples:

3 = HIGHLY RELEVANT - Directly answers or explains the query topic
    Example: Query "ARM exception handling" → chunk explains exception vectors, fault types, handler setup

2 = RELEVANT - Contains useful information about the query topic
    Example: Query "ARM exception handling" → chunk about interrupt handling that mentions exceptions

1 = MARGINALLY RELEVANT - Mentions query terms but doesn't explain them
    Example: Query "ARM exception handling" → chunk lists "exceptions" in a feature list without details

0 = NOT RELEVANT - No meaningful connection to query
    Example: Query "ARM exception handling" → chunk about memory allocation or unrelated topic

Return JSON with:
- score: 0-3 (use the full range, most chunks should be 0-2)
- confidence: 0.0-1.0 (lower if uncertain)
- subtopics: 1-3 technical concepts this chunk covers
- reason: brief explanation"""


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


# 45 queries in 4 categories
QUERY_SET: list[QuerySpec] = [
    # Factoid (15)
    QuerySpec(
        "f01", "ARM exception handling", "factoid", ["exception_vectors", "fault_types"]
    ),
    QuerySpec(
        "f02",
        "AArch64 register conventions",
        "factoid",
        ["caller_saved", "callee_saved"],
    ),
    QuerySpec(
        "f03", "GIC interrupt priority", "factoid", ["priority_mask", "preemption"]
    ),
    QuerySpec(
        "f04", "ARM memory barrier instructions", "factoid", ["dmb", "dsb", "isb"]
    ),
    QuerySpec(
        "f05",
        "procedure call standard stack frame",
        "factoid",
        ["frame_pointer", "alignment"],
    ),
    QuerySpec("f06", "RISC-V trap handling", "factoid", ["trap_vector", "mcause"]),
    QuerySpec(
        "f07", "RISC-V privilege levels", "factoid", ["machine_mode", "supervisor_mode"]
    ),
    QuerySpec("f08", "SBI console functions", "factoid", ["putchar", "getchar"]),
    QuerySpec(
        "f09", "PLIC interrupt pending", "factoid", ["pending_register", "claim"]
    ),
    QuerySpec("f10", "RISC-V CSR registers", "factoid", ["mstatus", "mepc"]),
    QuerySpec(
        "f11", "process scheduling algorithm", "factoid", ["round_robin", "priority"]
    ),
    QuerySpec(
        "f12",
        "virtual memory page table",
        "factoid",
        ["page_table_entry", "translation"],
    ),
    QuerySpec(
        "f13",
        "file system inode structure",
        "factoid",
        ["inode_fields", "direct_blocks"],
    ),
    QuerySpec(
        "f14",
        "context switch implementation",
        "factoid",
        ["save_state", "restore_state"],
    ),
    QuerySpec("f15", "deadlock prevention", "factoid", ["conditions", "avoidance"]),
    # Conceptual (15)
    QuerySpec(
        "c01",
        "interrupt controller configuration",
        "conceptual",
        ["gic", "plic", "setup"],
    ),
    QuerySpec("c02", "memory management unit", "conceptual", ["arm_mmu", "riscv_mmu"]),
    QuerySpec("c03", "system call interface", "conceptual", ["syscall_entry", "trap"]),
    QuerySpec("c04", "boot sequence initialization", "conceptual", ["startup", "init"]),
    QuerySpec(
        "c05", "timer interrupt handling", "conceptual", ["timer_irq", "scheduling"]
    ),
    QuerySpec(
        "c06",
        "how to save registers on function call",
        "conceptual",
        ["callee_saved", "prologue"],
    ),
    QuerySpec(
        "c07", "page fault handler", "conceptual", ["fault_handling", "demand_paging"]
    ),
    QuerySpec("c08", "spinlock implementation", "conceptual", ["atomic_ops", "lock"]),
    QuerySpec("c09", "cache coherency protocol", "conceptual", ["mesi", "snoop"]),
    QuerySpec(
        "c10",
        "privilege escalation mechanism",
        "conceptual",
        ["mode_switch", "syscall"],
    ),
    QuerySpec("c11", "mutex vs semaphore", "conceptual", ["mutex", "semaphore"]),
    QuerySpec(
        "c12", "kernel stack layout", "conceptual", ["stack_frame", "trap_frame"]
    ),
    QuerySpec("c13", "TLB flush mechanism", "conceptual", ["tlb_invalidate", "asid"]),
    QuerySpec("c14", "DMA configuration", "conceptual", ["dma_setup", "transfer"]),
    QuerySpec("c15", "power management states", "conceptual", ["sleep", "idle"]),
    # Cross-domain (10)
    QuerySpec(
        "x01", "atomic operations", "cross_domain", ["arm_ldxr_stxr", "riscv_amo"]
    ),
    QuerySpec(
        "x02", "floating point registers", "cross_domain", ["arm_simd", "riscv_f"]
    ),
    QuerySpec(
        "x03",
        "instruction encoding format",
        "cross_domain",
        ["arm_encoding", "riscv_encoding"],
    ),
    QuerySpec(
        "x04", "privilege mode switching", "cross_domain", ["arm_el", "riscv_modes"]
    ),
    QuerySpec(
        "x05", "interrupt vector table", "cross_domain", ["gic_vectors", "plic_vectors"]
    ),
    QuerySpec(
        "x06", "callee saved registers", "cross_domain", ["arm_callee", "riscv_callee"]
    ),
    QuerySpec("x07", "exception return address", "cross_domain", ["elr", "mepc"]),
    QuerySpec(
        "x08", "nested interrupt handling", "cross_domain", ["nested_irq", "preemption"]
    ),
    QuerySpec(
        "x09", "page table entry format", "cross_domain", ["arm_pte", "riscv_pte"]
    ),
    QuerySpec(
        "x10",
        "function prologue epilogue",
        "cross_domain",
        ["arm_prologue", "riscv_prologue"],
    ),
    # Natural language (5)
    QuerySpec(
        "n01",
        "how does the CPU save state during interrupts",
        "natural_language",
        ["context_save", "registers"],
    ),
    QuerySpec(
        "n02",
        "what happens when a page is not in memory",
        "natural_language",
        ["page_fault", "demand_paging"],
    ),
    QuerySpec(
        "n03",
        "difference between supervisor and user mode",
        "natural_language",
        ["privilege", "protection"],
    ),
    QuerySpec(
        "n04",
        "steps to handle a system call",
        "natural_language",
        ["syscall_flow", "trap"],
    ),
    QuerySpec(
        "n05",
        "why use memory barriers",
        "natural_language",
        ["ordering", "consistency"],
    ),
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
                result.reason.strip()[:200],
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
