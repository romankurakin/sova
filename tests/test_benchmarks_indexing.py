"""Full isolated indexing with synthetic model outputs and real SQLite search."""

import json
import shutil
import sqlite3

import pytest

from benchmarks import indexing
from benchmarks.corpus import CorpusError
from sova import config, db, search
from tests.test_benchmarks_passages import make_reading_inputs, prepare


class FakeModels:
    def __init__(self, context_version="one", embedding_version="one", build="b1"):
        self.versions = {"context": context_version, "embedding": embedding_version}
        self.engine = {"build": build}
        self.context_calls = 0
        self.embedding_calls = 0
        self.closed = False
        self.started = []
        self.fail_context = False
        self.fail_embedding_after = None

    def start(self, phase):
        self.started.append(phase)
        return {"request_model": f"test-{phase}", "revision": self.versions[phase]}

    def verify(self):
        pass

    def context(self, request):
        self.context_calls += 1
        if self.fail_context:
            raise RuntimeError("context unavailable")
        return f"Synthetic context for {request['doc_name']}."

    def embed(self, texts, on_batch, on_inputs):
        for i, text in enumerate(texts):
            if self.fail_embedding_after == self.embedding_calls:
                raise KeyboardInterrupt
            self.embedding_calls += 1
            vector = [1.0, i / 10] + [0.0] * (config.EMBEDDING_DIM - 2)
            on_inputs([i], [text])
            on_batch([i], [vector], {"duration_s": 0.1})

    def close(self):
        self.closed = True


@pytest.fixture
def snapshot(tmp_path):
    return prepare(make_reading_inputs(tmp_path))


def build(snapshot, models=None):
    return indexing.prepare_index(
        snapshot, models=models or FakeModels(), notify=lambda _: None
    )


def test_full_index_uses_production_search_without_personal_paths(
    snapshot, monkeypatch
):
    monkeypatch.setattr(
        config, "get_db_path", lambda: pytest.fail("Personal DB must not be used")
    )
    monkeypatch.setattr(
        config, "get_data_dir", lambda: pytest.fail("Personal data must not be used")
    )
    monkeypatch.setattr(
        config, "activate_project", lambda **_: pytest.fail("No project switch")
    )
    models = FakeModels()
    result = build(snapshot, models)
    assert models.closed
    assert models.started == ["context", "embedding"]
    assert models.context_calls == models.embedding_calls == 2
    assert indexing.load_index(result)["passages"] == 2
    conn = db.connect_readonly(result.parent / "index.sqlite3")
    try:
        assert len(search.search_fts(conn, "useful", 10)) == 2
        assert (
            len(
                search.get_vector_candidates(
                    conn, [1.0] + [0.0] * (config.EMBEDDING_DIM - 1), 10
                )
            )
            == 2
        )
        assert (
            len(conn.execute("SELECT passage_id FROM benchmark_passages").fetchall())
            == 2
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE text LIKE '%Synthetic context%'"
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE search_text LIKE '%Synthetic context%'"
            ).fetchone()[0]
            == 2
        )
    finally:
        conn.close()


def test_reuse_and_variants_preserve_completed_a(snapshot):
    a = build(snapshot)
    original = a.read_bytes()
    second = FakeModels(embedding_version="two")
    b = build(snapshot, second)
    assert b != a
    assert second.context_calls == 0
    assert second.embedding_calls == 2
    again = FakeModels()
    assert build(snapshot, again) == a
    assert again.context_calls == again.embedding_calls == 0
    assert again.started == ["context", "embedding"]
    assert a.read_bytes() == original
    failed = FakeModels(context_version="failed")
    failed.fail_context = True
    with pytest.raises(CorpusError, match="context unavailable"):
        build(snapshot, failed)
    assert failed.closed
    assert a.read_bytes() == original
    indexing.load_index(a)


def test_llama_cpp_update_alone_reuses_every_artifact(snapshot):
    first = build(snapshot)
    updated = FakeModels(build="b2")
    assert build(snapshot, updated) == first
    assert updated.context_calls == updated.embedding_calls == 0


def test_contexts_follow_the_request_the_model_sees(snapshot, monkeypatch):
    build(snapshot)
    monkeypatch.setattr(
        indexing.llama_client, "CONTEXT_SYSTEM_PROMPT", "A changed instruction."
    )
    changed = FakeModels()
    build(snapshot, changed)
    assert changed.context_calls == 2


def test_interruption_keeps_completed_embedding_batch(snapshot):
    interrupted = FakeModels()
    interrupted.fail_embedding_after = 1
    with pytest.raises(KeyboardInterrupt):
        build(snapshot, interrupted)
    assert interrupted.closed
    assert len(list(snapshot.parents[2].glob("embeddings/*/*.json"))) == 1
    assert not list(snapshot.parents[2].glob("indexes/*/manifest.json"))
    resume = FakeModels()
    result = build(snapshot, resume)
    assert resume.context_calls == 0
    assert resume.embedding_calls == 1
    assert indexing.load_index(result)["passages"] == 2


def test_quantization_failure_retries_without_model_calls(snapshot, monkeypatch):
    real_quantize = db.quantize_vectors
    monkeypatch.setattr(
        db,
        "quantize_vectors",
        lambda _: (_ for _ in ()).throw(RuntimeError("quantize failed")),
    )
    with pytest.raises(CorpusError, match="quantize failed"):
        build(snapshot)
    assert not list(snapshot.parents[2].glob("indexes/*/manifest.json"))
    monkeypatch.setattr(db, "quantize_vectors", real_quantize)
    resume = FakeModels()
    result = build(snapshot, resume)
    assert resume.context_calls == resume.embedding_calls == 0
    indexing.load_index(result)


def test_vacuum_and_cache_do_not_change_logical_identity(snapshot):
    result = build(snapshot)
    before = result.read_bytes()
    conn = sqlite3.connect(result.parent / "index.sqlite3")
    conn.execute("VACUUM")
    conn.close()
    indexing.load_index(result)
    assert build(snapshot).read_bytes() == before


def test_engine_provenance_does_not_change_index_identity(snapshot):
    result = build(snapshot)
    before = result.read_bytes()
    for folder in ("contexts", "embeddings"):
        for path in snapshot.parents[2].glob(f"{folder}/*/*.json"):
            artifact = json.loads(path.read_bytes())
            artifact["engine"] = {"build": "another-llama-cpp"}
            artifact["sha256"] = indexing._hash(
                {key: value for key, value in artifact.items() if key != "sha256"}
            )
            path.write_bytes(indexing._json(artifact))
    assert build(snapshot) == result
    assert result.read_bytes() == before


def test_index_source_paths_survive_data_directory_move(snapshot, tmp_path):
    result = build(snapshot)
    original = snapshot.parents[2]
    relocated = tmp_path / "moved"
    shutil.copytree(original, relocated)
    moved = relocated / result.relative_to(original)
    conn = db.connect_readonly(moved.parent / "index.sqlite3")
    try:
        for (path,) in conn.execute("SELECT path FROM documents"):
            assert (moved.parent / path).resolve().is_relative_to(relocated)
            assert (moved.parent / path).is_file()
    finally:
        conn.close()
    assert (
        indexing.load_index(moved)["logical_sha256"]
        == indexing.load_index(result)["logical_sha256"]
    )


def test_corrupt_finished_index_and_context_are_reported(snapshot):
    result = build(snapshot)
    conn = db.init_db(result.parent / "index.sqlite3")
    conn.execute("UPDATE chunks SET text='damaged' WHERE id=1")
    conn.commit()
    conn.close()
    with pytest.raises(CorpusError, match="database content differs"):
        build(snapshot)
    context = next(snapshot.parents[2].glob("contexts/*/*.json"))
    context.write_text("{}")
    with pytest.raises(CorpusError, match="Damaged immutable artifact"):
        build(snapshot)


def test_actual_embedding_input_is_saved_and_raw_passages_preserved(snapshot):
    class Shortening(FakeModels):
        def embed(self, texts, on_batch, on_inputs):
            for i in range(len(texts)):
                on_inputs([i], ["effective shortened input"])
                on_batch(
                    [i],
                    [[1.0] + [0.0] * (config.EMBEDDING_DIM - 1)],
                    {"duration_s": 0.1},
                )

    result = build(snapshot, Shortening())
    assert indexing.load_index(result)["changed_embedding_inputs"] == 2
    saved = [
        json.loads(path.read_bytes())
        for path in snapshot.parents[2].glob("embeddings/*/*.json")
    ]
    assert all(row["effective_text"] == "effective shortened input" for row in saved)
    assert all("useful text" in row["request"]["text"] for row in saved)


def test_bad_vector_never_publishes_ready_index(snapshot):
    class BadVector(FakeModels):
        def embed(self, texts, on_batch, on_inputs):
            on_inputs([0], [texts[0]])
            on_batch([0], [[float("nan")] * config.EMBEDDING_DIM], {"duration_s": 0.1})

    with pytest.raises(CorpusError, match="non-finite"):
        build(snapshot, BadVector())
    assert not list(snapshot.parents[2].glob("indexes/*/manifest.json"))


def test_equal_embedding_inputs_share_one_immutable_artifact(snapshot):
    models = FakeModels()
    preparation = indexing.IndexPreparation(
        snapshot, models=models, notify=lambda _: None
    )
    first = preparation.items[0]
    second = first | {"passage_id": "another-provenance"}
    for item in (first, second):
        preparation.contexts[item["passage_id"]] = {"context": "Shared context."}
    preparation.prepare_vectors([first, second])
    assert models.embedding_calls == 1
    assert (
        preparation.vectors[first["passage_id"]]
        == preparation.vectors[second["passage_id"]]
    )
    assert len(list(snapshot.parents[2].glob("embeddings/*/*.json"))) == 1


def test_context_neighbors_do_not_cross_document_boundaries(snapshot):
    preparation = indexing.IndexPreparation(
        snapshot, models=FakeModels(), notify=lambda _: None
    )
    for item in preparation.items:
        assert item["context_request"]["prev_chunk"] == ""
        assert item["context_request"]["next_chunk"] == ""
