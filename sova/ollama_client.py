"""Ollama API client for embeddings."""

import ollama

from sova.config import EMBEDDING_MODEL, QUERY_TASK


def check_ollama() -> tuple[bool, str]:
    """Check if Ollama is running and required models are available."""
    try:
        models = [m.model for m in ollama.list().models if m.model]
        if not any(EMBEDDING_MODEL in m for m in models):
            ollama.pull(EMBEDDING_MODEL)
        return True, "ready"
    except ollama.ResponseError as e:
        return False, e.error
    except Exception:
        return False, "not running (ollama serve)"


def get_query_embedding(query: str) -> list[float]:
    """Embed query with instruction prefix for asymmetric retrieval."""
    # Queries and chunks are embedded differently on purpose. The instruction
    # prefix tells the model this is a search query, which shifts the embedding
    # toward retrieval-relevant dimensions. Chunks are embedded raw so they
    # represent their content faithfully. See config.QUERY_TASK.
    prompt = f"Instruct: {QUERY_TASK}\nQuery: {query}"
    response = ollama.embed(model=EMBEDDING_MODEL, input=prompt)
    return list(response.embeddings[0])


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Embed document chunks - NO instruction prefix."""
    response = ollama.embed(model=EMBEDDING_MODEL, input=texts)
    return [list(emb) for emb in response.embeddings]
