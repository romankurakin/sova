"""llama-server API client for embeddings, reranking, and context generation."""

import json
import math
import plistlib
import subprocess
import time
import urllib.parse
import urllib.request
import urllib.error
from collections import deque
from collections.abc import Callable
from pathlib import Path

from sova.config import (
    CONTEXT_MODEL,
    CONTEXT_SERVER_URL,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_SERVER_URL,
    RERANK_TIMEOUT,
    RERANKER_SERVER_URL,
    SOVA_HOME,
    get_memory_hard_cap_gib,
    get_model_memory_estimate_gib,
)

# Map server URLs to launchd service labels for on-demand startup.
_SERVICE_LABELS = {
    EMBEDDING_SERVER_URL: "com.sova.embedding",
    RERANKER_SERVER_URL: "com.sova.reranker",
    CONTEXT_SERVER_URL: "com.sova.chat",
}
_SERVICE_PORTS = {
    EMBEDDING_SERVER_URL: 8081,
    RERANKER_SERVER_URL: 8082,
    CONTEXT_SERVER_URL: 8083,
}

# llama.cpp model cache directory.
_LLAMA_CACHE = Path.home() / "Library" / "Caches" / "llama.cpp"

# Map service labels to expected cache file basenames.
_CACHE_FILES = {
    "com.sova.embedding": "Qwen_Qwen3-Embedding-4B-GGUF_Qwen3-Embedding-4B-Q8_0.gguf",
    "com.sova.chat": "mistralai_Ministral-3-14B-Instruct-2512-GGUF_Ministral-3-14B-Instruct-2512-Q8_0.gguf",
    "com.sova.reranker": "ggml-org_Qwen3-Reranker-0.6B-Q8_0-GGUF_qwen3-reranker-0.6b-q8_0.gguf",
}

# Idle timeout in seconds; services stop after this much inactivity.
_IDLE_TIMEOUT = 900  # 15 minutes.
_RERANK_MIN_UBATCH = 4096

_ACTIVITY_DIR = SOVA_HOME / "activity"
_SEARCH_ACTIVITY_LABEL = "com.sova.search"
_SEARCH_PAIR_LABELS = frozenset({"com.sova.embedding", "com.sova.reranker"})

# Fixed embedding batch size. Concurrency is controlled by llama-server slots.
_EMBED_BATCH_SIZE = 12
_EMBED_TIMEOUT_FLOOR = 75.0
_EMBED_TIMEOUT_CEIL = 420.0
_EMBED_TOKEN_SAFETY_MARGIN_RATIO = 0.02
_EMBED_TOKEN_SAFETY_MARGIN_MIN = 128
_EMBED_TOKEN_SAFETY_MARGIN_MAX = 256
_EMBED_STABLE_TOKEN_BUDGET = 4096
_EMBED_TOKENIZE_TIMEOUT_S = 20.0
_EMBED_HEADER_KEEP_TAIL_PARTS = 2
_EMBED_RECOVERY_TOKEN_BUDGET_STEPS = (512, 256, 192)
_EMBED_RECOVERY_MAX_ATTEMPTS = 6
_EMBED_RECOVERY_RESTART_PAUSE_S = 1.0
_EMBED_RECOVERY_CANARY_REQUESTS = 2

_RERANK_COMPACT_CHAR_BUDGETS = (3000, 2000, 1400, 1000, 800, 600, 450)
_RERANK_KEEP_RATIO_FALLBACKS = (1.0, 0.75, 0.5, 0.35, 0.25)

# Startup probe: send sequential requests to warm up the Metal autorelease.
# pool and fail early if the embedding server is unstable (llama.cpp #18568).
_EMBED_CANARY_REQUESTS = 24
_EMBED_CANARY_TEXT = "sova embedding canary"
_DOWNLOAD_PROGRESS_STEP_GIB = 0.5

# Required-model admission is estimate-based; allow a small mode-specific.
# slack to avoid false negatives on machines where real RSS is lower than.
# static estimates.
_REQUIRED_SOFT_MARGIN_GIB = {
    "index": 2.0,
    "search": 0.5,
}

_MODE_SERVERS = {
    "index_context": [
        ("chat", CONTEXT_SERVER_URL, True),
    ],
    "index_embed": [
        ("embedding", EMBEDDING_SERVER_URL, True),
    ],
    "search": [
        ("embedding", EMBEDDING_SERVER_URL, True),
        ("reranker", RERANKER_SERVER_URL, True),
    ],
}

# Asymmetric retrieval: queries get this instruction prefix, chunks don't.
# This tells the model to optimize for query-passage matching rather than.
# generic similarity, which measurably improves recall on retrieval tasks.
QUERY_TASK = "Given a search query, retrieve relevant passages that answer the query"

CONTEXT_SYSTEM_PROMPT = (
    "Return exactly one plain-text sentence for technical retrieval. "
    "No markdown formatting: do not use backticks, emphasis markers, headings, "
    "table pipes, or list markers. "
    "Do not start with demonstratives or meta intros: never begin with "
    "'This', 'This chunk', 'This section', 'This document', "
    "'These', 'In this chunk', 'In this section', 'The chunk', "
    "'The section', 'The document', 'Here', or similar lead-ins. "
    "Begin directly with concrete technical terms copied from <chunk>. "
    "Keep technical tokens raw and never wrap them. "
    "Bad: 'This chunk describes PLIC priorities.' "
    "Good: 'PLIC priorities are encoded in ...'."
)

CONTEXT_USER_PROMPT = """\
<document title="{doc_name}" section="{section_title}">
<previous>{prev_chunk}</previous>
<chunk>{chunk_text}</chunk>
<next>{next_chunk}</next>
</document>

One sentence in plain text."""


class ServerError(RuntimeError):
    """Raised when a llama-server instance is unreachable or returns an error."""


def _http_error_body(exc: urllib.error.HTTPError) -> str:
    """Best-effort extract of HTTP error body for diagnostics."""
    try:
        data = exc.read()
    except Exception:
        return ""
    if not data:
        return ""
    return data.decode("utf-8", errors="replace").strip()


def _compact_http_error_body(raw: str) -> str:
    """Normalize server JSON/text errors into a compact one-line message."""
    text = " ".join(raw.split())
    if not text:
        return ""
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                msg = err.get("message")
                if isinstance(msg, str) and msg.strip():
                    return " ".join(msg.split())[:220]
    except Exception:
        pass
    return text[:220]


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
            data = json.loads(resp.read())
            split = urllib.parse.urlsplit(url)
            base_url = f"{split.scheme}://{split.netloc}"
            label = _SERVICE_LABELS.get(base_url)
            if label:
                _touch_activity(label)
            return data
    except urllib.error.HTTPError as e:
        body = _http_error_body(e)
        if body:
            body = _compact_http_error_body(body)
            raise ServerError(f"server error {e.code} from {url}: {body}") from e
        raise ServerError(f"server error {e.code} from {url}") from e
    except urllib.error.URLError as e:
        raise ServerError(f"server not reachable at {url} ({e.reason})") from e
    except TimeoutError as e:
        raise ServerError(f"server timeout from {url}") from e
    except json.JSONDecodeError as e:
        raise ServerError(f"invalid JSON from {url}") from e


def _touch_activity(label: str) -> None:
    """Record that a service was just used."""
    try:
        _ACTIVITY_DIR.mkdir(parents=True, exist_ok=True)
        (_ACTIVITY_DIR / label).write_bytes(b"")
        if label in _SEARCH_PAIR_LABELS:
            (_ACTIVITY_DIR / _SEARCH_ACTIVITY_LABEL).write_bytes(b"")
    except OSError:
        # Activity tracking is best-effort; health checks should not fail on FS errors.
        pass


def _plist_exists(label: str) -> bool:
    """Check if a launchd plist is installed for the given label."""
    plist = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    return plist.exists()


def _configured_ubatch_size(label: str) -> int | None:
    """Read configured --ubatch-size from launchd plist ProgramArguments."""
    plist = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    try:
        data = plistlib.loads(plist.read_bytes())
    except Exception:
        return None
    args = data.get("ProgramArguments")
    if not isinstance(args, list):
        return None
    for i, arg in enumerate(args):
        if arg in {"--ubatch-size", "-ub"} and (i + 1) < len(args):
            try:
                return int(str(args[i + 1]))
            except ValueError:
                return None
    return None


def _configured_ctx_size(label: str) -> int | None:
    """Read configured --ctx-size from launchd plist ProgramArguments."""
    plist = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    try:
        data = plistlib.loads(plist.read_bytes())
    except Exception:
        return None
    args = data.get("ProgramArguments")
    if not isinstance(args, list):
        return None
    for i, arg in enumerate(args):
        if arg in {"--ctx-size", "-c"} and (i + 1) < len(args):
            try:
                return int(str(args[i + 1]))
            except ValueError:
                return None
    return None


def _server_status(label: str) -> str:
    """Return human-readable status for a service: downloading, loading, or waiting."""
    cache_file = _CACHE_FILES.get(label)
    if not cache_file:
        return "waiting"
    dl_path = _LLAMA_CACHE / f"{cache_file}.downloadInProgress"
    cached_path = _LLAMA_CACHE / cache_file
    if dl_path.exists():
        try:
            size_gb = dl_path.stat().st_size / (1024**3)
            # Coarsen progress to avoid noisy status spam during long downloads.
            bucketed = (
                math.floor(size_gb / _DOWNLOAD_PROGRESS_STEP_GIB)
                * _DOWNLOAD_PROGRESS_STEP_GIB
            )
            return f"downloading ({bucketed:.1f} GB)"
        except OSError:
            return "downloading"
    if not cached_path.exists():
        return "starting"
    return "loading"


def _ensure_server(url: str, timeout: float = 300.0) -> bool:
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


def stop_server(url: str, *, suppress_interrupt: bool = False) -> None:
    """Stop a launchd-managed server and wait for it to fully exit.

    If ``suppress_interrupt`` is True, Ctrl+C during shutdown does not bubble
    up (best-effort cleanup path).
    """
    label = _SERVICE_LABELS.get(url)
    if not label:
        return
    activity_labels = [label]
    if label in _SEARCH_PAIR_LABELS:
        activity_labels.append(_SEARCH_ACTIVITY_LABEL)
    try:
        subprocess.run(["launchctl", "stop", label], capture_output=True)
        # Wait for the process to actually exit and free memory.
        wait_s = 10 if suppress_interrupt else 30
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            result = subprocess.run(
                ["launchctl", "list", label], capture_output=True, text=True
            )
            # PID column is "-" when the service is not running.
            if result.returncode != 0 or result.stdout.startswith("-"):
                break
            time.sleep(1)
    except KeyboardInterrupt:
        if not suppress_interrupt:
            raise
    finally:
        for activity_label in activity_labels:
            try:
                (_ACTIVITY_DIR / activity_label).unlink(missing_ok=True)
            except KeyboardInterrupt:
                if not suppress_interrupt:
                    raise


def _is_file_idle(path: Path, now: float) -> bool:
    """Return whether an activity file exceeded idle timeout."""
    try:
        return (now - path.stat().st_mtime) > _IDLE_TIMEOUT
    except FileNotFoundError:
        return False


def _stop_and_unlink_if_still_idle(label: str, path: Path, now: float) -> None:
    """Stop a service and unlink its activity marker if it remained idle."""
    if not _is_file_idle(path, now):
        return
    subprocess.run(["launchctl", "stop", label], capture_output=True)
    try:
        if _is_file_idle(path, now):
            path.unlink(missing_ok=True)
    except FileNotFoundError:
        pass


def cleanup_idle_services() -> None:
    """Stop services that have been idle longer than _IDLE_TIMEOUT."""
    if not _ACTIVITY_DIR.exists():
        return
    now = time.time()
    label_files = {path.name: path for path in _ACTIVITY_DIR.iterdir()}

    # Keep embedding+rereanker lifecycle coupled to avoid "orphan" leftovers.
    search_file = label_files.get(_SEARCH_ACTIVITY_LABEL)
    if search_file and _is_file_idle(search_file, now):
        for label in _SEARCH_PAIR_LABELS:
            subprocess.run(["launchctl", "stop", label], capture_output=True)
        if _is_file_idle(search_file, now):
            try:
                search_file.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
            for label in _SEARCH_PAIR_LABELS:
                label_path = label_files.get(label)
                if label_path is not None:
                    try:
                        label_path.unlink(missing_ok=True)
                    except FileNotFoundError:
                        pass

    for label, label_file in label_files.items():
        if label == _SEARCH_ACTIVITY_LABEL or label in _SEARCH_PAIR_LABELS:
            continue
        _stop_and_unlink_if_still_idle(label, label_file, now)


def _service_memory_estimate_gib(service_name: str) -> float:
    """Map service names to fixed model memory estimates."""
    if service_name == "embedding":
        return get_model_memory_estimate_gib("embed")
    return get_model_memory_estimate_gib(service_name)


def _admit_services_for_mode(
    mode: str, services: list[tuple[str, str, bool]]
) -> tuple[list[tuple[str, str, bool]], str | None]:
    """Apply a simple memory hard-cap admission check.

    Required services must fit the cap; optional services are dropped first.
    """
    profile_mode = "search" if mode == "search" else "index"
    hard_cap_gib = get_memory_hard_cap_gib(profile_mode)
    soft_margin_gib = _REQUIRED_SOFT_MARGIN_GIB.get(profile_mode, 0.0)
    admitted: list[tuple[str, str, bool]] = []
    used_gib = 0.0
    optional_reason: str | None = None

    for name, url, required in services:
        estimate = _service_memory_estimate_gib(name)
        projected = used_gib + estimate
        if required and (mode in {"index_context", "index_embed"} or mode == "search"):
            # Index phases run exactly one required model at a time.
            # Search required services should also skip estimate-only preflight.
            # Rely on real server startup/health for.
            # the final capacity decision.
            admitted.append((name, url, required))
            used_gib += estimate
            continue
        if projected <= hard_cap_gib:
            admitted.append((name, url, required))
            used_gib += estimate
            continue
        if required and projected <= (hard_cap_gib + soft_margin_gib):
            admitted.append((name, url, required))
            used_gib += estimate
            continue
        if required:
            needed = projected
            return [], (
                "memory hard-cap exceeded: "
                f"need ~{needed:.1f} GiB for {name}, cap={hard_cap_gib:.1f} GiB"
            )
        optional_reason = "disabled by memory hard-cap"

    return admitted, optional_reason


def _health_ok(url: str, timeout: float = 1.0) -> bool:
    """Lightweight /health probe without autostart side-effects."""
    try:
        req = urllib.request.Request(f"{url}/health")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read()).get("status") == "ok"
    except Exception:
        return False


def _pid_for_port(port: int) -> int | None:
    """Return listener PID for a TCP port, if any."""
    try:
        result = subprocess.run(
            ["lsof", "-n", "-P", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        raw = line.strip()
        if raw.isdigit():
            return int(raw)
    return None


def _rss_mib_for_pid(pid: int) -> float | None:
    """Return RSS memory for PID in MiB."""
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    line = result.stdout.strip().splitlines()
    if not line:
        return None
    try:
        rss_kib = int(line[0].strip())
    except ValueError:
        return None
    return round(rss_kib / 1024.0, 1)


def get_services_runtime_status() -> list[dict[str, str | int | float | bool | None]]:
    """Return runtime status for known llama services (no autostart)."""
    services = [
        ("embedding", EMBEDDING_SERVER_URL),
        ("reranker", RERANKER_SERVER_URL),
        ("chat", CONTEXT_SERVER_URL),
    ]
    rows: list[dict[str, str | int | float | bool | None]] = []
    for name, url in services:
        label = _SERVICE_LABELS[url]
        port = _SERVICE_PORTS[url]
        installed = _plist_exists(label)
        pid = _pid_for_port(port)
        healthy = _health_ok(url)
        rss_mib = _rss_mib_for_pid(pid) if pid is not None else None

        if not installed:
            state = "not installed"
        elif healthy:
            state = "running"
        elif pid is not None:
            state = "starting"
        else:
            state = "stopped"

        rows.append(
            {
                "name": name,
                "label": label,
                "port": port,
                "state": state,
                "pid": pid,
                "rss_mib": rss_mib,
                "healthy": healthy,
                "installed": installed,
            }
        )
    return rows


def check_servers(
    on_status: Callable[[str], None] | None = None,
    mode: str = "search",
    fast_only: bool = False,
    use_reranker: bool = True,
) -> tuple[bool, str]:
    """Probe /health on needed llama-server instances, starting them if needed.

    Kicks off all services in parallel, then waits for each to become healthy.
    ``mode`` selects which servers are needed:
      - "index_context": chat only
      - "index_embed": embedding only
      - "search": embedding (+ reranker when enabled; chat skipped)
    ``on_status`` is called with a human-readable status string on each poll.
    """
    timeout = 300.0
    all_servers = _MODE_SERVERS.get(mode)
    if all_servers is None:
        raise ValueError(f"unknown server mode: {mode}")
    if mode == "search" and not use_reranker:
        all_servers = [srv for srv in all_servers if srv[0] != "reranker"]

    if mode == "search":
        for name, url, required in all_servers:
            label = _SERVICE_LABELS.get(url, "")
            if name == "reranker" and required and label and _plist_exists(label):
                ubatch = _configured_ubatch_size(label)
                effective_ubatch = 512 if ubatch is None else ubatch
                if effective_ubatch < _RERANK_MIN_UBATCH:
                    return (
                        False,
                        "reranker service configured with too small --ubatch-size "
                        f"({effective_ubatch}); run sova-install to update services",
                    )

    # Warm path: when required services are already healthy, skip launchctl and.
    # polling loops to keep repeated searches snappy.
    required_servers = [(name, url) for name, url, required in all_servers if required]
    if required_servers and all(
        _health_ok(url, timeout=0.2) for _name, url in required_servers
    ):
        for name, url in required_servers:
            label = _SERVICE_LABELS.get(url)
            if label:
                _touch_activity(label)
        return True, "ready"

    if fast_only:
        return False, "warm check failed"

    all_servers, admission_note = _admit_services_for_mode(mode, all_servers)
    if not all_servers and admission_note:
        return False, admission_note

    # Kick off launchctl start for all services that aren't already healthy.
    to_wait: list[tuple[str, str, str, bool]] = []  # (name, url, label, required).
    for name, url, required in all_servers:
        label = _SERVICE_LABELS.get(url, "")
        try:
            req = urllib.request.Request(f"{url}/health")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if json.loads(resp.read()).get("status") == "ok":
                    if label:
                        _touch_activity(label)
                    continue
        except Exception:
            pass
        if not label or not _plist_exists(label):
            if required:
                return (
                    False,
                    f"{name} server not reachable at {url} (service not installed)",
                )
            continue
        subprocess.run(["launchctl", "start", label], capture_output=True)
        to_wait.append((name, url, label, required))

    # Poll all started services concurrently in a single loop.
    healthy: set[str] = set()
    last_status: str | None = None
    if to_wait:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and len(healthy) < len(to_wait):
            time.sleep(2)
            # Build status string for pending services.
            if on_status:
                parts = []
                for name, _url, label, _req in to_wait:
                    if name in healthy:
                        continue
                    parts.append(f"{name}: {_server_status(label)}")
                status_text = ", ".join(parts)
                if status_text and status_text != last_status:
                    on_status(status_text)
                    last_status = status_text

            for name, url, label, _req in to_wait:
                if name in healthy:
                    continue
                try:
                    req = urllib.request.Request(f"{url}/health")
                    with urllib.request.urlopen(req, timeout=3) as resp:
                        if json.loads(resp.read()).get("status") == "ok":
                            _touch_activity(label)
                            healthy.add(name)
                except Exception:
                    pass

    for name, url, _label, required in to_wait:
        if required and name not in healthy:
            return False, f"{name} server not reachable at {url}"

    for name, _url, _label, required in to_wait:
        if not required and name not in healthy:
            admission_note = "unavailable"

    if admission_note:
        return True, f"ready (reranker {admission_note})"

    return True, "ready"


def get_query_embedding(query: str) -> list[float]:
    """Embed query with instruction prefix for asymmetric retrieval."""
    if not _ensure_server(EMBEDDING_SERVER_URL):
        raise ServerError(f"embedding server not reachable at {EMBEDDING_SERVER_URL}")
    prompt = f"Instruct: {QUERY_TASK}\nQuery: {query}"
    return _embed_inputs_via_server([prompt], timeout=45.0)[0]


def _embedding_timeout_for_batch(texts: list[str]) -> float:
    """Timeout scales with total text size in a batch."""
    total_chars = sum(len(t) for t in texts)
    estimate = 60.0 + (total_chars / 1200.0)
    return max(_EMBED_TIMEOUT_FLOOR, min(_EMBED_TIMEOUT_CEIL, estimate))


def _is_recoverable_embedding_error(reason: str) -> bool:
    markers = (
        "remote end closed connection",
        "server not reachable",
        "connection reset",
        "broken pipe",
        "server timeout",
        "timed out",
        "timeout",
        "exceeds the available context size",
    )
    low = reason.lower()
    return any(marker in low for marker in markers)


def _embedding_token_budget() -> int:
    """Token budget per embedding input with context and stability margins."""
    ctx_size = _configured_ctx_size("com.sova.embedding") or 4096
    margin = int(math.ceil(ctx_size * _EMBED_TOKEN_SAFETY_MARGIN_RATIO))
    margin = max(_EMBED_TOKEN_SAFETY_MARGIN_MIN, margin)
    margin = min(_EMBED_TOKEN_SAFETY_MARGIN_MAX, margin)
    return min(_EMBED_STABLE_TOKEN_BUDGET, max(512, ctx_size - margin))


def _token_count_via_server(text: str) -> int:
    """Count tokens for a text using llama-server /tokenize."""
    resp = _post_json(
        f"{EMBEDDING_SERVER_URL}/tokenize",
        {"content": text},
        timeout=_EMBED_TOKENIZE_TIMEOUT_S,
    )
    tokens = resp.get("tokens")
    if not isinstance(tokens, list):
        raise ServerError("invalid tokenize response: missing tokens")
    return len(tokens)


def _fit_text_to_token_budget(text: str, token_budget: int) -> str:
    """Shrink text by tail-trimming until tokenized length fits token_budget.

    Tail-trimming preserves the beginning where document/section semantics live.
    """
    if token_budget <= 0:
        return ""

    cache: dict[int, int] = {}

    def token_len(char_count: int) -> int:
        key = max(0, min(len(text), char_count))
        if key not in cache:
            cache[key] = _token_count_via_server(text[:key])
        return cache[key]

    total_chars = len(text)
    if token_len(total_chars) <= token_budget:
        return text

    lo = 1
    hi = total_chars
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if token_len(mid) <= token_budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    if best <= 0:
        return text[:1]
    return text[:best].rstrip()


def _split_embedding_head_body(text: str) -> tuple[str, str]:
    """Split embedding text into a semantic head and body at first blank line."""
    marker = text.find("\n\n")
    if marker < 0:
        return text.strip(), ""
    head = text[:marker].strip()
    body = text[marker + 2 :].lstrip("\n")
    return head, body


def _compact_header_line(header: str) -> str:
    """Compact a bracketed header path while keeping doc and nearest sections."""
    text = header.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return text
    parts = [part.strip() for part in text[1:-1].split("|") if part.strip()]
    if len(parts) <= 1:
        return text
    doc = parts[0]
    tail = parts[-_EMBED_HEADER_KEEP_TAIL_PARTS:]
    compact: list[str] = [doc]
    for part in tail:
        if part != doc:
            compact.append(part)
    return "[" + " | ".join(compact) + "]"


def _fit_body_with_prefix_to_budget(prefix: str, body: str, token_budget: int) -> str:
    """Tail-trim body text while keeping prefix fully intact."""
    cache: dict[int, int] = {}

    def token_len(char_count: int) -> int:
        key = max(0, min(len(body), char_count))
        if key not in cache:
            cache[key] = _token_count_via_server(prefix + body[:key])
        return cache[key]

    if token_len(len(body)) <= token_budget:
        return body

    lo = 0
    hi = len(body)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if token_len(mid) <= token_budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return body[:best].rstrip()


def _prepare_embedding_text(
    text: str,
    *,
    token_budget: int | None = None,
) -> str:
    """Token-first normalization and budget enforcement for embeddings."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    budget = _embedding_token_budget() if token_budget is None else token_budget
    if _token_count_via_server(normalized) <= budget:
        return normalized

    head, body = _split_embedding_head_body(normalized)
    if not body:
        return _fit_text_to_token_budget(normalized, budget)

    compacted_head = _compact_header_line(head)
    if (
        _token_count_via_server(head) > budget
        or _token_count_via_server(f"{head}\n\n") > budget
    ):
        head = compacted_head
        if _token_count_via_server(head) > budget:
            return _fit_text_to_token_budget(head, budget)

    prefix = f"{head}\n\n"
    if _token_count_via_server(prefix) > budget:
        return _fit_text_to_token_budget(prefix, budget)

    body = _fit_body_with_prefix_to_budget(prefix, body, budget)
    return (prefix + body).rstrip()


def _parse_embeddings_response(resp: dict, expected: int) -> list[list[float]]:
    """Validate /v1/embeddings response and return embeddings in input order."""
    data = resp.get("data")
    if not isinstance(data, list):
        raise ServerError("invalid embedding response: missing data")

    ordered: list[list[float] | None] = [None] * expected
    for item in data:
        if not isinstance(item, dict):
            raise ServerError("invalid embedding response: non-object item")
        idx = item.get("index")
        emb = item.get("embedding")
        if not isinstance(idx, int) or idx < 0 or idx >= expected:
            raise ServerError("invalid embedding response: bad index")
        if not isinstance(emb, list):
            raise ServerError("invalid embedding response: bad embedding payload")
        try:
            ordered[idx] = [float(v) for v in emb]
        except (TypeError, ValueError) as e:
            raise ServerError("invalid embedding response: non-numeric value") from e
        if len(ordered[idx]) != EMBEDDING_DIM:
            raise ServerError(
                "invalid embedding response: dimension mismatch "
                f"(expected={EMBEDDING_DIM}, got={len(ordered[idx])})"
            )

    if any(v is None for v in ordered):
        raise ServerError("invalid embedding response: incomplete data")

    return ordered  # type: ignore[return-value]


def _embed_inputs_via_server(
    texts: list[str], timeout: float | None = None
) -> list[list[float]]:
    """Embed a list of texts via llama-server /v1/embeddings."""
    if not texts:
        return []
    resp = _post_json(
        f"{EMBEDDING_SERVER_URL}/v1/embeddings",
        {"model": EMBEDDING_MODEL, "input": texts},
        timeout=timeout or _embedding_timeout_for_batch(texts),
    )
    return _parse_embeddings_response(resp, expected=len(texts))


def run_embedding_canary(requests: int | None = None) -> None:
    """Probe embedding server with sequential small requests.

    ``requests`` defaults to ``_EMBED_CANARY_REQUESTS`` and can be lowered for
    lightweight mid-run warmups after a controlled server recycle.
    """
    if not _ensure_server(EMBEDDING_SERVER_URL):
        raise ServerError(f"embedding server not reachable at {EMBEDDING_SERVER_URL}")
    total = _EMBED_CANARY_REQUESTS if requests is None else max(1, int(requests))
    for i in range(total):
        _embed_inputs_via_server([f"{_EMBED_CANARY_TEXT} {i}"], timeout=45.0)


def _tail_lines(path: Path, max_lines: int = 12) -> list[str]:
    """Read last N lines from a text file."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return [line.rstrip("\n") for line in deque(f, maxlen=max_lines)]
    except OSError:
        return []


def _pick_log_hint(lines: list[str]) -> str:
    """Pick one useful line from a service log tail."""

    def _compact(line: str) -> str:
        text = " ".join(line.split())
        if "|" in text:
            parts = [part.strip() for part in text.split("|") if part.strip()]
            for part in reversed(parts):
                if part and part.lower() != "srv":
                    text = part
                    break
        return text[:180]

    keywords = (
        "error",
        "failed",
        "exception",
        "oom",
        "out of memory",
        "timeout",
        "too large",
        "not reachable",
        "500",
        "invalid",
    )
    noise = (
        "done request:",
        "update_slots:",
        "all slots are idle",
        "slot release:",
    )
    for line in reversed(lines):
        text = _compact(line)
        if not text:
            continue
        low = text.lower()
        if any(n in low for n in noise):
            continue
        if any(k in low for k in keywords):
            return text
    return ""


def get_service_diagnostics(url: str) -> str:
    """Return compact diagnostics for a launchd-managed llama service."""
    label = _SERVICE_LABELS.get(url)
    if not label:
        return ""
    state = "stopped"
    status = subprocess.run(
        ["launchctl", "list", label], capture_output=True, text=True
    )
    if status.returncode == 0 and status.stdout.strip():
        first = status.stdout.splitlines()[0].split()
        if first and first[0].isdigit() and int(first[0]) > 0:
            state = "running"
        else:
            state = "loaded"

    service_name = label.rsplit(".", 1)[-1]
    err_log = SOVA_HOME / "logs" / f"{service_name}.err.log"
    if not err_log.exists():
        return state

    tail = [line for line in _tail_lines(err_log, max_lines=20) if line.strip()]
    if not tail:
        return state

    hint = _pick_log_hint(tail)
    if not hint:
        return state
    return f"{state} | last-error: {hint}"


def get_embeddings_batch(
    texts: list[str],
    on_batch: Callable[[list[int], list[list[float]], dict[str, float | int]], None]
    | None = None,
) -> list[list[float]]:
    """Embed texts via llama-server with a fixed single-worker request plan."""
    if not texts:
        return []
    if not _ensure_server(EMBEDDING_SERVER_URL):
        raise ServerError(f"embedding server not reachable at {EMBEDDING_SERVER_URL}")

    token_budget = _embedding_token_budget()
    total = len(texts)
    results: list[list[float]] = []
    for start in range(0, total, _EMBED_BATCH_SIZE):
        end = min(total, start + _EMBED_BATCH_SIZE)
        raw_batch = texts[start:end]
        try:
            batch_texts = [
                _prepare_embedding_text(text, token_budget=token_budget)
                for text in raw_batch
            ]
        except Exception as e:
            reason = str(e).strip() or e.__class__.__name__
            raise ServerError(
                "embedding preflight failed "
                f"(completed={start}/{total}, batch_size={len(raw_batch)}, workers=1): "
                f"{reason}"
            ) from e
        started = time.monotonic()
        try:
            batch_embeddings = _embed_inputs_via_server(
                batch_texts,
                _embedding_timeout_for_batch(batch_texts),
            )
        except Exception as e:
            reason = str(e).strip() or e.__class__.__name__
            can_recover = _is_recoverable_embedding_error(reason)
            if not can_recover:
                raise ServerError(
                    "embedding server failed "
                    f"(completed={start}/{total}, batch_size={len(batch_texts)}, workers=1): "
                    f"{reason}"
                ) from e

            try:
                batch_embeddings = []
                for idx, text in enumerate(batch_texts):
                    raw_text = raw_batch[idx]
                    if not _ensure_server(EMBEDDING_SERVER_URL):
                        raise ServerError(
                            f"embedding server not reachable at {EMBEDDING_SERVER_URL}"
                        )
                    candidates: list[str] = [text]
                    for budget_step in _EMBED_RECOVERY_TOKEN_BUDGET_STEPS:
                        if budget_step >= token_budget:
                            continue
                        try:
                            reduced = _prepare_embedding_text(
                                raw_text,
                                token_budget=budget_step,
                            )
                        except Exception:
                            continue
                        if reduced != candidates[-1]:
                            candidates.append(reduced)

                    last_reason = ""
                    embedded_vector: list[float] | None = None
                    for attempt in range(_EMBED_RECOVERY_MAX_ATTEMPTS):
                        candidate = candidates[min(attempt, len(candidates) - 1)]
                        if not _ensure_server(EMBEDDING_SERVER_URL):
                            raise ServerError(
                                f"embedding server not reachable at {EMBEDDING_SERVER_URL}"
                            )
                        try:
                            embedded = _embed_inputs_via_server(
                                [candidate],
                                _embedding_timeout_for_batch([candidate]),
                            )
                            if len(embedded) != 1:
                                raise ServerError(
                                    "embedding server returned incomplete recovery batch "
                                    f"(expected=1, got={len(embedded)})"
                                )
                            embedded_vector = embedded[0]
                            break
                        except Exception as single_exc:
                            single_reason = (
                                str(single_exc).strip() or single_exc.__class__.__name__
                            )
                            last_reason = single_reason
                            if attempt >= (
                                _EMBED_RECOVERY_MAX_ATTEMPTS - 1
                            ) or not _is_recoverable_embedding_error(single_reason):
                                raise
                            stop_server(
                                EMBEDDING_SERVER_URL,
                                suppress_interrupt=True,
                            )
                            time.sleep(_EMBED_RECOVERY_RESTART_PAUSE_S)
                            try:
                                run_embedding_canary(
                                    requests=_EMBED_RECOVERY_CANARY_REQUESTS
                                )
                            except Exception:
                                pass

                    if embedded_vector is None:
                        raise ServerError(
                            "embedding recovery exhausted attempts "
                            f"(attempts={_EMBED_RECOVERY_MAX_ATTEMPTS}): {last_reason}"
                        )
                    batch_embeddings.append(embedded_vector)
            except Exception as recover_exc:
                recover_reason = (
                    str(recover_exc).strip() or recover_exc.__class__.__name__
                )
                raise ServerError(
                    "embedding server failed "
                    f"(completed={start}/{total}, batch_size={len(batch_texts)}, workers=1): "
                    f"{reason} | recovery failed: {recover_reason}"
                ) from recover_exc

        if len(batch_embeddings) != len(batch_texts):
            raise ServerError(
                "embedding server returned incomplete batch "
                f"(expected={len(batch_texts)}, got={len(batch_embeddings)})"
            )

        results.extend(batch_embeddings)
        if on_batch:
            on_batch(
                list(range(start, end)),
                batch_embeddings,
                {
                    "batch_size": len(batch_texts),
                    "workers": 1,
                    "duration_s": round(time.monotonic() - started, 3),
                },
            )

    return results


def generate_context(
    doc_name: str,
    section_title: str | None,
    chunk_text: str,
    prev_chunk: str = "",
    next_chunk: str = "",
) -> str:
    """Generate a contextual summary for a chunk using a local LLM."""
    _ensure_server(CONTEXT_SERVER_URL)
    prompt = CONTEXT_USER_PROMPT.format(
        doc_name=doc_name,
        section_title=section_title or "(no section)",
        prev_chunk=prev_chunk[-500:] if prev_chunk else "(start of document)",
        chunk_text=chunk_text[:1000],
        next_chunk=next_chunk[:500] if next_chunk else "(end of document)",
    )
    try:
        resp = _post_json(
            f"{CONTEXT_SERVER_URL}/v1/chat/completions",
            {
                "model": CONTEXT_MODEL,
                "messages": [
                    {"role": "system", "content": CONTEXT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 96,
                "temperature": 0.0,
            },
        )
        return (resp["choices"][0]["message"]["content"] or "").strip()
    except ServerError as e:
        msg = str(e).lower()
        if "failed to parse input at pos 0" not in msg:
            raise
        # llama.cpp occasionally fails parsing chat output for specific prompts.
        # Retry via completion endpoint with equivalent instruction text.
        fallback = _post_json(
            f"{CONTEXT_SERVER_URL}/completion",
            {
                "prompt": f"{CONTEXT_SYSTEM_PROMPT}\n\n{prompt}",
                "n_predict": 96,
                "temperature": 0.0,
            },
        )
        text = (fallback.get("content") or "").strip()
        if text:
            return text
        raise ServerError(
            "context fallback via /completion returned empty content"
        ) from e


def rerank(query: str, documents: list[str], top_n: int = 10) -> list[dict]:
    """Rerank documents via /v1/rerank."""

    def _is_batch_overflow(msg: str) -> bool:
        low = msg.lower()
        return "too large to process" in low and "physical batch size" in low

    def _is_timeout(msg: str) -> bool:
        low = msg.lower()
        return "server timeout" in low or "timed out" in low

    def _compact_doc(text: str, char_budget: int) -> str:
        if char_budget <= 0:
            return ""
        if len(text) <= char_budget:
            return text
        # Preserve both lead and tail signals when truncating for reranking.
        head = max(200, int(char_budget * 0.72))
        tail = max(0, char_budget - head - 5)
        if tail <= 0:
            return text[:char_budget]
        return f"{text[:head]}\n...\n{text[-tail:]}"

    if not _ensure_server(RERANKER_SERVER_URL, timeout=30.0):
        raise ServerError(f"reranker server not reachable at {RERANKER_SERVER_URL}")

    doc_count = len(documents)
    if doc_count == 0:
        return []

    last_overflow: ServerError | None = None
    last_timeout: ServerError | None = None
    smallest_budget = _RERANK_COMPACT_CHAR_BUDGETS[-1]
    attempts: list[tuple[int, int | None]] = [(doc_count, None)]

    for keep_ratio in _RERANK_KEEP_RATIO_FALLBACKS:
        keep_count = max(1, int(round(doc_count * keep_ratio)))
        keep_count = min(keep_count, doc_count)
        for budget in _RERANK_COMPACT_CHAR_BUDGETS:
            attempts.append((keep_count, budget))

    seen_attempts: set[tuple[int, int | None]] = set()
    for keep_count, budget in attempts:
        attempt_key = (keep_count, budget)
        if attempt_key in seen_attempts:
            continue
        seen_attempts.add(attempt_key)

        docs_slice = documents[:keep_count]
        if budget is None:
            docs_payload = docs_slice
        else:
            docs_payload = [_compact_doc(doc, budget) for doc in docs_slice]

        try:
            resp = _post_json(
                f"{RERANKER_SERVER_URL}/v1/rerank",
                {
                    "query": query,
                    "documents": docs_payload,
                    "top_n": min(top_n, len(docs_payload)),
                },
                timeout=RERANK_TIMEOUT,
            )
            results = resp.get("results")
            if not isinstance(results, list):
                raise ServerError("invalid rerank response: missing results")
            return results
        except ServerError as e:
            if _is_batch_overflow(str(e)):
                last_overflow = e
                continue
            if _is_timeout(str(e)):
                # Slow reranker calls are retried with tighter payloads.
                last_timeout = e
                continue
            raise

    if last_overflow is not None:
        raise ServerError(
            "reranker input exceeds llama-server physical batch size even after "
            f"adaptive compaction (min doc budget={smallest_budget} chars); "
            "increase com.sova.reranker --ubatch-size and restart services"
        ) from last_overflow
    if last_timeout is not None:
        raise ServerError(
            "reranker timed out even after adaptive compaction; "
            "increase reranker timeout or reduce system load"
        ) from last_timeout

    raise ServerError("reranker failed unexpectedly")
