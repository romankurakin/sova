"""Ollama API client for embeddings and query expansion."""

import re

import ollama
from pydantic import BaseModel

from sova.config import DOMAIN_MODEL, EMBEDDING_MODEL, LLM_MODEL


class DomainResponse(BaseModel):
    """Structured response for domain detection."""

    domain: str


def detect_domain(doc_name: str, section_titles: list[str]) -> str:
    """Detect document domain using LLM from section titles."""
    # Limit sections to avoid overwhelming the model
    titles_text = "\n".join(section_titles[:50])

    prompt = f"""Classify document field. Keep specific names (ARM, RISC-V, React, PyTorch, etc).

Examples:
Document: arm-cortex-m-guide
Field: ARM Cortex-M Programming

Document: riscv-vector-extension
Field: RISC-V Vector Extension

Document: react-hooks-reference
Field: React Hooks

Document: pytorch-neural-networks
Field: PyTorch Deep Learning

Document: kubernetes-networking
Field: Kubernetes Networking

Document: rust-ownership-guide
Field: Rust Memory Management

Document: {doc_name}
Headers: {titles_text}
Field:"""

    try:
        response = ollama.chat(
            model=DOMAIN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format=DomainResponse.model_json_schema(),
            options={"num_predict": 30, "temperature": 0},
        )
        result = DomainResponse.model_validate_json(response.message.content)
        domain = result.domain.strip()[:60]
        if domain:
            return domain
    except Exception:
        pass
    return doc_name  # Fallback to document name


def check_ollama() -> tuple[bool, str]:
    """Check if Ollama is running and required models are available."""
    try:
        models = [m.model for m in ollama.list().models if m.model]
        if not any(EMBEDDING_MODEL in m for m in models):
            ollama.pull(EMBEDDING_MODEL)
        if not any(LLM_MODEL in m for m in models):
            ollama.pull(LLM_MODEL)
        if not any(DOMAIN_MODEL in m for m in models):
            ollama.pull(DOMAIN_MODEL)
        return True, "ready"
    except ollama.ResponseError as e:
        return False, e.error
    except Exception:
        return False, "not running (ollama serve)"


def get_embeddings_batch(texts: list[str]) -> list[list[float | int]]:
    """Get embeddings for a batch of texts."""
    response = ollama.embed(model=EMBEDDING_MODEL, input=texts)
    return [list(emb) for emb in response.embeddings]


def expand_query(query: str) -> list[str]:
    """Use LLM to expand query with related technical terms."""
    prompt = f"List 3-5 alternative search terms for: {query}. Include synonyms used in different contexts. One term per line, no explanations."

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 80},
        )
        content = response.message.content or ""
        terms = []
        for line in content.split("\n"):
            term = re.sub(r"^[\d\.\-\*•\s]+", "", line).strip()
            if term and 2 <= len(term) <= 60:
                terms.append(term)
        return terms[:5]
    except ollama.ResponseError:
        return []
