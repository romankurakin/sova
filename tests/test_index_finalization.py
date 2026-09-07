"""Index finalization against real SQLite vectors, without model services."""

import pytest

from sova import cli, config
from sova.cache import SemanticCache
from sova.db import (
    VECTOR_INDEX_STATE_KEY,
    connect_readonly,
    embedding_to_blob,
    get_meta,
    init_db,
    quantize_vectors,
)
from sova.search import get_vector_candidates, search_vector


@pytest.fixture
def indexed_project(monkeypatch, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    monkeypatch.setattr(config, "get_docs_dir", lambda: docs)
    monkeypatch.setattr(config, "get_data_dir", lambda: tmp_path / "data")
    monkeypatch.setattr(config, "get_db_path", lambda: tmp_path / "indexed.db")
    monkeypatch.setattr(config, "get_active_project_id", lambda: "test")
    monkeypatch.setattr(cli, "stop_server", lambda *_args, **_kwargs: None)

    def unexpected_model_load(**_kwargs):
        pytest.fail("Completed checkpoints should not load a model")

    monkeypatch.setattr(cli, "check_servers", unexpected_model_load)
    conn = init_db()
    state = cli._sync_index_signatures(conn)
    for doc_id, name, count in [(1, "removed", 49), (2, "kept", 51)]:
        source = docs / f"{name}.md"
        source.write_text(f"Source for {name}\n", encoding="utf-8")
        conn.execute(
            """
            INSERT INTO documents
                (id, name, path, expected_chunks, source_signature, chunk_signature)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                name,
                str(source),
                count,
                cli._file_signature(source),
                state.chunk_sig,
            ),
        )
    for chunk_id in range(1, 101):
        vector = [(101 - chunk_id) / 100, chunk_id / 100] + [0.0] * (
            config.EMBEDDING_DIM - 2
        )
        conn.execute(
            """
            INSERT INTO chunks
                (id, doc_id, start_line, end_line, word_count, text, search_text,
                 embedding, embedding_signature)
            VALUES (?, ?, ?, ?, 2, 'source passage', 'source passage', ?, ?)
            """,
            (
                chunk_id,
                1 if chunk_id <= 49 else 2,
                chunk_id,
                chunk_id,
                embedding_to_blob(vector),
                state.embed_sig,
            ),
        )
        conn.execute(
            """
            INSERT INTO chunk_contexts (chunk_id, context, model, pipeline_signature)
            VALUES (?, 'Source context.', ?, ?)
            """,
            (chunk_id, config.CONTEXT_MODEL, state.context_sig),
        )
    cli._commit_index_signatures(conn, state)
    quantize_vectors(conn)
    conn.close()
    return docs


def test_deleted_documents_do_not_crowd_out_vector_candidates(indexed_project):
    (indexed_project / "removed.md").unlink()
    cli._run_index_mode()

    conn = connect_readonly()
    query = [1.0] + [0.0] * (config.EMBEDDING_DIM - 1)
    results = search_vector(conn, embedding_to_blob(query), 50)
    assert len(results) == 50
    assert all(chunk_id >= 50 for chunk_id, _score in results)
    assert get_meta(conn, VECTOR_INDEX_STATE_KEY) == "ready"
    conn.close()


def test_unchanged_index_reuses_quantization_and_cache(indexed_project, monkeypatch):
    def unexpected_quantization(_conn):
        pytest.fail("An unchanged index should reuse its vector checkpoint")

    monkeypatch.setattr(cli, "quantize_vectors", unexpected_quantization)
    cache = SemanticCache()
    cache.put([1.0, 0.0], [(50, 0.7)])
    conn = connect_readonly()
    cached = conn.execute("SELECT * FROM query_cache").fetchall()
    conn.close()

    cli._run_index_mode()

    conn = connect_readonly()
    assert conn.execute("SELECT * FROM query_cache").fetchall() == cached
    conn.close()


def test_finalization_failure_preserves_pending_work_for_retry(
    indexed_project, monkeypatch
):
    (indexed_project / "removed.md").unlink()
    cache = SemanticCache()
    cache.put([1.0, 0.0], [(1, 0.99)])

    def fail_quantization(_conn):
        raise RuntimeError("simulated finalization failure")

    with monkeypatch.context() as failure:
        failure.setattr(cli, "quantize_vectors", fail_quantization)
        with pytest.raises(SystemExit) as exc:
            cli._run_index_mode()
        assert exc.value.code == 1

    conn = connect_readonly()
    assert get_meta(conn, VECTOR_INDEX_STATE_KEY) == "pending"
    assert conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0] == 0
    query = [1.0] + [0.0] * (config.EMBEDDING_DIM - 1)
    # Search must use current embeddings while the quantized snapshot is stale.
    assert len(get_vector_candidates(conn, query, 10, candidates=50)) == 50
    conn.close()

    cli._run_index_mode()

    conn = connect_readonly()
    assert get_meta(conn, VECTOR_INDEX_STATE_KEY) == "ready"
    assert len(search_vector(conn, embedding_to_blob(query), 50)) == 50
    conn.close()


def test_embedding_changes_invalidate_cache_and_survive_reopening(indexed_project):
    cache = SemanticCache()
    cache.put([1.0, 0.0], [(1, 0.99)])
    conn = init_db()
    query = [1.0] + [0.0] * (config.EMBEDDING_DIM - 1)
    conn.execute(
        "UPDATE chunks SET embedding = ? WHERE id = 100", (embedding_to_blob(query),)
    )
    assert get_meta(conn, VECTOR_INDEX_STATE_KEY) == "pending"
    assert conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0] == 0
    conn.rollback()
    assert get_meta(conn, VECTOR_INDEX_STATE_KEY) == "ready"
    assert conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0] == 1

    conn.execute(
        "UPDATE chunks SET embedding = ? WHERE id = 100", (embedding_to_blob(query),)
    )
    conn.commit()
    conn.close()

    conn = init_db()
    assert get_meta(conn, VECTOR_INDEX_STATE_KEY) == "pending"
    assert get_vector_candidates(conn, query, 10)[0][0] == 100
    conn.close()
    cli._run_index_mode()
    conn = connect_readonly()
    assert get_meta(conn, VECTOR_INDEX_STATE_KEY) == "ready"
    assert search_vector(conn, embedding_to_blob(query), 50)[0][0] == 100
    conn.close()


def test_legacy_index_rebuilds_once_to_adopt_vector_checkpoint(indexed_project):
    conn = init_db()
    conn.execute("DELETE FROM index_meta WHERE key = ?", (VECTOR_INDEX_STATE_KEY,))
    conn.commit()
    conn.close()
    cli._run_index_mode()
    conn = connect_readonly()
    assert get_meta(conn, VECTOR_INDEX_STATE_KEY) == "ready"
    conn.close()
