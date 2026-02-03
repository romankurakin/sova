"""Ollama API client for embeddings and context generation."""

import ollama

from sova.config import CONTEXT_MODEL, EMBEDDING_MODEL

# Asymmetric retrieval: queries get this instruction prefix, chunks don't.
# This tells the model to optimize for query-passage matching rather than
# generic similarity, which measurably improves recall on retrieval tasks.
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


def check_ollama() -> tuple[bool, str]:
    """Check if Ollama is running and required models are available."""
    try:
        models = [m.model for m in ollama.list().models if m.model]
        for model in [EMBEDDING_MODEL, CONTEXT_MODEL]:
            if not any(model in m for m in models):
                ollama.pull(model)
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
    response = ollama.chat(
        model=CONTEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 64},
    )
    return response.message.content.strip()
