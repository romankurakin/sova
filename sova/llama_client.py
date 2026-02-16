"""llama-server API client for embeddings, reranking, and context generation."""

import json
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path

from sova.config import (
    CONTEXT_MODEL,
    CONTEXT_SERVER_URL,
    EMBEDDING_MODEL,
    EMBEDDING_SERVER_URL,
    RERANK_TIMEOUT,
    RERANKER_SERVER_URL,
    SOVA_HOME,
)

# Map server URLs to launchd service labels for on-demand startup
_SERVICE_LABELS = {
    EMBEDDING_SERVER_URL: "com.sova.embedding",
    RERANKER_SERVER_URL: "com.sova.reranker",
    CONTEXT_SERVER_URL: "com.sova.chat",
}

# Idle timeout in seconds; services stop after this much inactivity
_IDLE_TIMEOUT = 3600  # 1 hour

_ACTIVITY_DIR = SOVA_HOME / "activity"

# Asymmetric retrieval: queries get this instruction prefix, chunks don't
# This tells the model to optimize for query-passage matching rather than
# generic similarity, which measurably improves recall on retrieval tasks
QUERY_TASK = "Given a search query, retrieve relevant passages that answer the query"

CONTEXT_PROMPT = """\
<document title="{doc_name}" section="{section_title}">
{prev_chunk}
[CHUNK START]
{chunk_text}
[CHUNK END]
{next_chunk}
</document>

Succinctly describe what this chunk covers to improve search retrieval. Strictly one sentence, no preamble."""


class ServerError(RuntimeError):
    """Raised when a llama-server instance is unreachable or returns an error."""


def _post_json(url: str, payload: dict, timeout: float = 30.0) -> dict:
    """POST JSON to a URL and return parsed response."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise ServerError(f"server error {e.code} from {url}") from e
    except urllib.error.URLError as e:
        raise ServerError(f"server not reachable at {url} ({e.reason})") from e


def _touch_activity(label: str) -> None:
    """Record that a service was just used."""
    _ACTIVITY_DIR.mkdir(parents=True, exist_ok=True)
    (_ACTIVITY_DIR / label).write_bytes(b"")


def _plist_exists(label: str) -> bool:
    """Check if a launchd plist is installed for the given label."""
    plist = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    return plist.exists()


def _ensure_server(url: str, timeout: float = 120.0) -> bool:
    """Start a launchd service if its server is not responding, then wait for health."""
    label = _SERVICE_LABELS.get(url)

    try:
        req = urllib.request.Request(f"{url}/health")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if json.loads(resp.read()).get("status") == "ok":
                if label:
                    _touch_activity(label)
                return True
    except Exception:
        pass

    if not label or not _plist_exists(label):
        return False

    subprocess.run(["launchctl", "start", label], capture_output=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(2)
        try:
            req = urllib.request.Request(f"{url}/health")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if json.loads(resp.read()).get("status") == "ok":
                    _touch_activity(label)
                    return True
        except Exception:
            pass
    return False


def cleanup_idle_services() -> None:
    """Stop services that have been idle longer than _IDLE_TIMEOUT."""
    if not _ACTIVITY_DIR.exists():
        return
    now = time.time()
    for label_file in _ACTIVITY_DIR.iterdir():
        label = label_file.name
        idle = now - label_file.stat().st_mtime
        if idle > _IDLE_TIMEOUT:
            subprocess.run(["launchctl", "stop", label], capture_output=True)
            label_file.unlink(missing_ok=True)


def check_servers() -> tuple[bool, str]:
    """Probe /health on all three llama-server instances, starting them if needed.

    Reranker failure is a warning, not fatal.
    """
    servers = [
        ("embedding", EMBEDDING_SERVER_URL),
        ("chat", CONTEXT_SERVER_URL),
    ]
    for name, url in servers:
        if not _ensure_server(url):
            return False, f"{name} server not reachable at {url}"

    # Reranker is optional; warn but don't fail.
    if not _ensure_server(RERANKER_SERVER_URL, timeout=30.0):
        return True, "ready (reranker unavailable)"

    return True, "ready"


def get_query_embedding(query: str) -> list[float]:
    """Embed query with instruction prefix for asymmetric retrieval."""
    _ensure_server(EMBEDDING_SERVER_URL)
    prompt = f"Instruct: {QUERY_TASK}\nQuery: {query}"
    resp = _post_json(
        f"{EMBEDDING_SERVER_URL}/v1/embeddings",
        {"model": EMBEDDING_MODEL, "input": prompt},
    )
    return resp["data"][0]["embedding"]


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Embed document chunks - NO instruction prefix."""
    _ensure_server(EMBEDDING_SERVER_URL)
    resp = _post_json(
        f"{EMBEDDING_SERVER_URL}/v1/embeddings",
        {"model": EMBEDDING_MODEL, "input": texts},
    )
    # Sort by index to guarantee order matches input.
    items = sorted(resp["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in items]


def generate_context(
    doc_name: str,
    section_title: str | None,
    chunk_text: str,
    prev_chunk: str = "",
    next_chunk: str = "",
) -> str:
    """Generate a contextual summary for a chunk using a local LLM."""
    prompt = CONTEXT_PROMPT.format(
        doc_name=doc_name,
        section_title=section_title or "(no section)",
        prev_chunk=prev_chunk[-500:] if prev_chunk else "(start of document)",
        chunk_text=chunk_text[:1000],
        next_chunk=next_chunk[:500] if next_chunk else "(end of document)",
    )
    resp = _post_json(
        f"{CONTEXT_SERVER_URL}/v1/chat/completions",
        {
            "model": CONTEXT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
        },
    )
    return (resp["choices"][0]["message"]["content"] or "").strip()


def rerank(query: str, documents: list[str], top_n: int = 10) -> list[dict] | None:
    """Rerank documents via /v1/rerank. Returns None on failure (graceful)."""
    if not _ensure_server(RERANKER_SERVER_URL, timeout=30.0):
        return None
    try:
        resp = _post_json(
            f"{RERANKER_SERVER_URL}/v1/rerank",
            {"query": query, "documents": documents, "top_n": top_n},
            timeout=RERANK_TIMEOUT,
        )
        return resp.get("results", None)
    except Exception:
        return None
