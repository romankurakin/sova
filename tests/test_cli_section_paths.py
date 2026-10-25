"""Chunk ancestry stays consistent across durable and model inputs."""

import re

import pytest

from sova import cli, config
from sova.db import init_db
from sova.extract import chunk_text
from sova.index_text import contextualized_prefix


@pytest.fixture
def prepared_sections(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "get_data_dir", lambda: tmp_path / "data")
    monkeypatch.setattr(config, "get_db_path", lambda: tmp_path / "index.db")

    def count_tokens(texts):
        return [len(re.findall(r"\w+|[^\w\s]", text)) for text in texts]

    monkeypatch.setattr(cli, "get_token_counts_batch", count_tokens)
    monkeypatch.setattr(
        cli,
        "chunk_text",
        lambda lines, **kwargs: chunk_text(lines, target_tokens=48, **kwargs),
    )
    md = tmp_path / "doc.md"
    md.write_text(
        "\n".join(
            [
                "# Parent > literal separator",
                "",
                " ".join(["intro"] * 60),
                "",
                "###### First field",
                "",
                "First field details.",
                "",
                "###### Second field",
                "",
                "Second field details.",
            ]
        )
    )
    source = cli._PreparedSource("doc", md, "source-signature")
    conn = init_db()
    cli._tokenize_doc(source, conn)
    try:
        yield conn, source
    finally:
        conn.close()


def test_section_span_agrees_in_storage_context_and_embedding(
    prepared_sections, monkeypatch
):
    conn, _source = prepared_sections
    chunks, sections = cli._load_prepared_doc(conn, 1)
    path = "Parent > literal separator"
    assert conn.execute(
        """
        SELECT c.start_line, s.start_line, c.section_path, c.search_prefix
        FROM chunks c JOIN sections s ON c.section_id = s.id
        ORDER BY c.start_line
        """
    ).fetchall() == [
        (chunk["start_line"], 1, path, contextualized_prefix("doc", path))
        for chunk in chunks
    ]
    contexts = []

    def generate(_name, section, text, _prev, _next):
        contexts.append((section, text))
        return "Generated context."

    embeddings = []

    def embed(texts, *, on_batch):
        embeddings.extend(texts)
        on_batch(
            list(range(len(texts))), [[1.0] * config.EMBEDDING_DIM] * len(texts), {}
        )

    monkeypatch.setattr(cli, "generate_context", generate)
    monkeypatch.setattr(cli, "get_embeddings_batch", embed)
    cli._generate_contexts("doc", 1, chunks, sections, conn)
    cli._embed_doc("doc", 1, chunks, sections, conn)

    assert contexts == [(path, chunk["text"]) for chunk in chunks]
    assert embeddings == [
        row[0]
        for row in conn.execute("SELECT search_text FROM chunks ORDER BY start_line")
    ]
    assert all(text.startswith(f"[doc | {path}]\n\n") for text in embeddings)


def test_old_checkpoint_recomputes_paths_and_invalidates_only_changed_inputs(
    prepared_sections,
):
    conn, source = prepared_sections
    old_signature = cli._signature(
        [
            str(config.CHUNK_TARGET_TOKENS),
            config.EMBEDDING_MODEL,
            "structure-token-chunks.v4",
        ]
    )
    conn.execute("UPDATE documents SET chunk_signature = ?", (old_signature,))
    conn.execute(
        "UPDATE chunks SET embedding = ?, embedding_signature = ?",
        (b"retained-vector", cli._embedding_pipeline_signature()),
    )
    conn.execute(
        """
        INSERT INTO chunk_contexts (chunk_id, context, model, pipeline_signature)
        SELECT id, 'retained context', 'model', ? FROM chunks
        """,
        (cli._context_pipeline_signature(),),
    )
    conn.execute(
        """
        UPDATE chunks SET
            section_id = (SELECT id FROM sections WHERE start_line = 5),
            section_path = 'Parent > literal separator > First field',
            search_prefix = 'old incorrect prefix'
        WHERE start_line = 5
        """
    )
    conn.commit()

    assert cli._current_tokenized_doc_id(conn, source, old_signature) == 1
    assert (
        cli._current_tokenized_doc_id(conn, source, cli._chunk_pipeline_signature())
        is None
    )
    cli._tokenize_doc(source, conn)

    rows = conn.execute(
        """
        SELECT c.start_line, c.section_path, c.embedding, cc.context
        FROM chunks c LEFT JOIN chunk_contexts cc ON c.id = cc.chunk_id
        ORDER BY c.start_line
        """
    ).fetchall()
    assert rows == [
        (1, "Parent > literal separator", b"retained-vector", "retained context"),
        (5, "Parent > literal separator", None, None),
    ]
    assert (
        cli._current_tokenized_doc_id(conn, source, cli._chunk_pipeline_signature())
        == 1
    )
