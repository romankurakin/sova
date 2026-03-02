"""LLM-as-Judge for creating ground truth relevance judgments."""

import ast
import json
import os
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel
from sova.config import CONTEXT_MODEL

from .search_interface import get_backend


class JudgeError(Exception):
    """Raised when the judge model fails permanently (e.g. model not found)."""


JUDGE_MODEL = os.environ.get("SOVA_BENCH_JUDGE_MODEL", CONTEXT_MODEL)

# Retry settings.
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0


JUDGE_PROMPT = """You are an information retrieval judge. Rate how well this document chunk satisfies the search query. Imagine a developer searched this mixed technical corpus — would this chunk help them complete the task in the query?

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


QUERY_SET: list[QuerySpec] = [
    # Exact lookup (6) — identifier-heavy retrieval that should reward BM25.
    QuerySpec("q01", "mcause CSR", "exact_lookup", ["mcause", "trap_cause"]),
    QuerySpec("q02", "mtvec CSR", "exact_lookup", ["mtvec", "trap_vector"]),
    QuerySpec("q03", "satp CSR", "exact_lookup", ["satp", "address_translation"]),
    QuerySpec(
        "q04",
        "PLIC claim complete register",
        "exact_lookup",
        ["plic_claim", "interrupt_complete"],
    ),
    QuerySpec(
        "q05",
        "SBI system reset extension",
        "exact_lookup",
        ["sbi_system_reset", "runtime_services"],
    ),
    QuerySpec(
        "q06",
        "AAPCS64 callee-saved registers",
        "exact_lookup",
        ["aapcs64", "callee_saved", "calling_convention"],
    ),
    # Conceptual (6) — semantic retrieval over mechanisms and procedures.
    QuerySpec(
        "q07",
        "how trap delegation moves exceptions from machine mode to supervisor mode",
        "conceptual",
        ["trap_delegation", "machine_mode", "supervisor_mode"],
    ),
    QuerySpec(
        "q08",
        "how sv39 virtual address translation works",
        "conceptual",
        ["sv39", "page_walk", "address_translation"],
    ),
    QuerySpec(
        "q09",
        "how plic prioritization and thresholding decide delivered interrupts",
        "conceptual",
        ["plic_priority", "threshold", "interrupt_delivery"],
    ),
    QuerySpec(
        "q10",
        "how sbi mediates supervisor requests to machine-level firmware",
        "conceptual",
        ["sbi_calls", "firmware_interface", "privilege_boundary"],
    ),
    QuerySpec(
        "q11",
        "how calling conventions preserve registers across function calls",
        "conceptual",
        ["register_preservation", "prologue_epilogue", "abi_rules"],
    ),
    QuerySpec(
        "q12",
        "how kernel and user privilege separation is enforced",
        "conceptual",
        ["privilege_isolation", "user_kernel_boundary", "protection"],
    ),
    # Cross-doc (8) — broad tasks expected to draw from multiple documents.
    QuerySpec(
        "q13",
        "function argument passing and return value conventions in low-level systems code",
        "cross_doc",
        ["arg_passing", "return_values", "abi_conventions"],
    ),
    QuerySpec(
        "q14",
        "interrupt handling path from external signal to handler return",
        "cross_doc",
        ["interrupt_flow", "handler_entry", "handler_return"],
    ),
    QuerySpec(
        "q15",
        "syscall path from user code through trap handling to kernel service",
        "cross_doc",
        ["syscall_entry", "trap_handling", "kernel_service"],
    ),
    QuerySpec(
        "q16",
        "register context that must be saved during context switch and exception handling",
        "cross_doc",
        ["context_save_restore", "exception_context", "register_state"],
    ),
    QuerySpec(
        "q17",
        "boot flow from firmware initialization to first user process",
        "cross_doc",
        ["boot_flow", "kernel_init", "first_user_process"],
    ),
    QuerySpec(
        "q18",
        "memory protection bits and page table permissions for user and kernel pages",
        "cross_doc",
        ["memory_protection", "page_permissions", "user_kernel_access"],
    ),
    QuerySpec(
        "q19",
        "trap vector setup and control transfer on exception entry",
        "cross_doc",
        ["trap_vector", "exception_entry", "control_transfer"],
    ),
    QuerySpec(
        "q20",
        "timer and software interrupts used by kernels and supervisors",
        "cross_doc",
        ["timer_interrupts", "software_interrupts", "scheduler_tick"],
    ),
    # Natural/vague (6) — realistic troubleshooting phrasing.
    QuerySpec(
        "q21",
        "my trap handler returns to the wrong instruction address",
        "natural",
        ["trap_return", "saved_pc", "control_flow_bug"],
    ),
    QuerySpec(
        "q22",
        "external interrupts are pending but never reach my supervisor handler",
        "natural",
        ["interrupt_routing", "pending_interrupts", "supervisor_handler"],
    ),
    QuerySpec(
        "q23",
        "after enabling virtual memory my kernel immediately page-faults",
        "natural",
        ["vm_enable", "page_fault", "translation_setup"],
    ),
    QuerySpec(
        "q24",
        "porting from arm64 to riscv broke function call register usage",
        "natural",
        ["porting", "calling_convention", "register_mismatch"],
    ),
    QuerySpec(
        "q25",
        "context switch loses register values between processes",
        "natural",
        ["context_switch_bug", "register_corruption", "save_restore"],
    ),
    QuerySpec(
        "q26",
        "firmware call returns not supported for an sbi feature",
        "natural",
        ["sbi_error_codes", "firmware_call", "feature_support"],
    ),
    # Negative (4) — hard negatives (near-domain + far-domain).
    QuerySpec("q27", "x86 APIC ICR delivery mode", "negative", []),
    QuerySpec("q28", "Linux cgroup v2 cpu.max throttling", "negative", []),
    QuerySpec("q29", "PostgreSQL query planner cost model tuning", "negative", []),
    QuerySpec(
        "q30", "TensorFlow distributed training checkpoint sharding", "negative", []
    ),
]


def _is_permanent_error(exc: Exception) -> bool:
    """Check if an error is permanent (no point retrying)."""
    msg = str(exc).lower()
    permanent_markers = (
        "not found",
        "status code: 404",
        "status code: 401",
        "status code: 403",
        "invalid api key",
        "api key not valid",
        "permission denied",
        "quota exceeded",
        "insufficient_quota",
    )
    return any(marker in msg for marker in permanent_markers)


def _post_json(url: str, payload: dict, timeout: float = 60.0) -> dict:
    """POST JSON and parse JSON response."""
    import urllib.request
    import urllib.error

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace").strip()
        except Exception:
            body = ""

        detail = body
        if body:
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict):
                    detail = str(parsed.get("error") or parsed)
            except Exception:
                pass

        message = f"HTTP Error {e.code}: {e.reason}"
        if detail:
            message = f"{message} - {detail}"
        raise RuntimeError(message) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"request failed: {e.reason}") from e


_JUDGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "judgment",
        "strict": True,
        "schema": JudgmentResponse.model_json_schema(),
    },
}


def _iter_payload_candidates(text: str) -> list[str]:
    """Collect likely JSON payload candidates from model output."""
    stripped = text.strip()
    if not stripped:
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def _add(candidate: str) -> None:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            return
        seen.add(candidate)
        candidates.append(candidate)

    # Fenced candidates first (models often wrap JSON in markdown).
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL):
        _add(m.group(1))

    # Any decodable JSON object/array in free text.
    decoder = json.JSONDecoder()
    for i, ch in enumerate(stripped):
        if ch not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(stripped[i:])
            _add(stripped[i : i + end])
        except json.JSONDecodeError:
            continue

    # Whole response as a last resort.
    _add(stripped)
    return candidates


def _cleanup_json_like_text(payload: str) -> str:
    """Normalize common model output issues before parsing."""
    cleaned = payload.strip()
    if cleaned.lower().startswith("json"):
        parts = cleaned.splitlines()
        if len(parts) > 1:
            cleaned = "\n".join(parts[1:])
        else:
            cleaned = cleaned[4:]
    cleaned = cleaned.strip().strip("`")
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
    cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
    cleaned = re.sub(r"(?m)^\s*//.*$", "", cleaned)
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    return cleaned.strip()


def _python_literal_variants(payload: str) -> list[str]:
    variants = [payload]
    # Allow json-like literals with true/false/null for ast literal_eval fallback.
    normalized = re.sub(r"\btrue\b", "True", payload, flags=re.IGNORECASE)
    normalized = re.sub(r"\bfalse\b", "False", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bnull\b", "None", normalized, flags=re.IGNORECASE)
    if normalized != payload:
        variants.append(normalized)
    return variants


def _recover_with_regex(text: str) -> JudgmentResponse | None:
    """Best-effort recovery for non-JSON outputs that still contain fields."""
    score_match = re.search(r"(?i)[\"']?score[\"']?\s*[:=]\s*(-?\d+)", text)
    if not score_match:
        return None

    conf_match = re.search(
        r"(?i)[\"']?confidence[\"']?\s*[:=]\s*(-?\d+(?:\.\d+)?)", text
    )
    confidence = float(conf_match.group(1)) if conf_match else 0.5

    reason_match = re.search(r"(?is)[\"']?reason[\"']?\s*[:=]\s*([\"'])(.*?)\1", text)
    if reason_match:
        reason = reason_match.group(2).strip()
    else:
        reason_line = re.search(r"(?im)^\s*[\"']?reason[\"']?\s*[:=]\s*(.+)$", text)
        reason = (
            reason_line.group(1).strip() if reason_line else "parsed from model output"
        )

    subs: list[str] = []
    subs_match = re.search(r"(?is)[\"']?subtopics?[\"']?\s*[:=]\s*\[(.*?)\]", text)
    if subs_match:
        inner = subs_match.group(1)
        quoted = re.findall(r'(?s)"([^"]+)"|\'([^\']+)\'', inner)
        for a, b in quoted:
            token = (a or b).strip()
            if token:
                subs.append(token)
        if not subs:
            for token in inner.split(","):
                value = token.strip().strip("'\"")
                if value:
                    subs.append(value)

    return JudgmentResponse(
        score=int(score_match.group(1)),
        confidence=confidence,
        subtopics=subs[:5],
        reason=reason,
    )


def _parse_judgment_response(text: str) -> JudgmentResponse:
    """Parse model output into JudgmentResponse with tolerant fallback formats."""
    candidates = _iter_payload_candidates(text)
    last_error: Exception | None = None

    for payload in candidates:
        cleaned = _cleanup_json_like_text(payload)
        for candidate in (payload, cleaned):
            if not candidate:
                continue
            try:
                return JudgmentResponse.model_validate_json(candidate)
            except Exception as json_error:
                last_error = json_error

            # Some models emit Python dict literals instead of strict JSON.
            for literal_candidate in _python_literal_variants(candidate):
                try:
                    parsed = ast.literal_eval(literal_candidate)
                except (SyntaxError, ValueError):
                    continue
                if isinstance(parsed, dict):
                    try:
                        return JudgmentResponse.model_validate(parsed)
                    except Exception as validate_error:
                        last_error = validate_error

    recovered = _recover_with_regex(text)
    if recovered is not None:
        return recovered

    if last_error is not None:
        raise last_error
    raise ValueError("empty or unparseable judge response")


def _parse_bool(raw: str | None) -> bool | None:
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def should_use_debiasing() -> bool:
    """Return whether debiasing should be enabled for judge calls."""
    forced = _parse_bool(os.environ.get("SOVA_BENCH_USE_DEBIASING"))
    return True if forced is None else forced


def _call_judge_llama(prompt: str) -> JudgmentResponse:
    """Call a local llama-server chat model and parse JSON response."""
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
    return _parse_judgment_response(content)


def _call_judge(prompt: str) -> JudgmentResponse:
    """Call the local llama-server judge model."""
    return _call_judge_llama(prompt)


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

    # Re-judge borderline cases.
    if conf1 >= 0.7 and score1 not in {1, 2}:
        return score1, reason1, conf1, subs1, 0.0

    # Re-judge with attention shift.
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

    # Strategy 1: hybrid (existing search).
    _add_results(backend.search(query, limit=k_per_strategy))

    # Strategy 2: BM25-only.
    _add_results(backend.search_fts_only(query, limit=k_per_strategy))

    # Strategy 3: vector-only.
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

        # Skip already-judged chunks (incremental).
        if existing_judgments and chunk_id in existing_judgments:
            continue

        try:
            if use_debiasing:
                score, reason, confidence, subtopics, _ = judge_chunk_with_debiasing(
                    spec.query, hit["text"]
                )
            else:
                score, reason, confidence, subtopics = judge_chunk(
                    spec.query, hit["text"]
                )
        except JudgeError as e:
            raise JudgeError(
                f"query {spec.id} chunk {chunk_id} ({hit['doc']}): {e}"
            ) from e

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
