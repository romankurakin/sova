"""Ollama API client for embeddings and query expansion."""

import re

import ollama

from sova.config import EMBEDDING_MODEL, LLM_MODEL


def detect_domain(doc_name: str, section_titles: list[str]) -> str:
    """Detect document domain using LLM from section titles."""
    titles_text = "\n".join(section_titles)  # Use all sections

    prompt = f"""What domain/field is this document about? Reply with 2-5 words only.

Document: {doc_name}
Sections:
{titles_text}

Domain:"""

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 20},
        )
        domain = (response.message.content or "").strip()
        # Clean up: first line only, limit length
        domain = domain.split("\n")[0][:60]
        if domain:
            return domain
    except ollama.ResponseError:
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
