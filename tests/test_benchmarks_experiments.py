"""Experiment files, the private embedder and the experiments table."""

import json

import pytest
from pydantic import ValidationError

from benchmarks import indexing
from benchmarks.corpus import CorpusError
from benchmarks.embedders import PrivateEmbedder, embedding_scope
from benchmarks.experiments import Embedder, Experiment, load_experiment
from benchmarks.report import experiments_markdown
from sova import config, db, llama_client, search
from tests.test_benchmarks_indexing import FakeModels, build
from tests.test_benchmarks_passages import make_reading_inputs, prepare

GEMMA = {
    "repository": "org/model-GGUF",
    "file": "model-Q8_0.gguf",
    "dimensions": 3,
    "pooling": "mean",
    "context_size": 2048,
    "query_template": "task: search result | query: {query}",
    "document_template": "title: none | text: {text}",
}


def test_description_is_not_a_setting():
    baseline = load_experiment("baseline")
    edited = baseline.model_copy(update={"description": "Another text."})
    assert edited.settings() == baseline.settings()
    assert "description" not in baseline.settings()
    assert baseline.native
    assert not Experiment(
        description="x", embedder=Embedder.model_validate(GEMMA)
    ).native
    with pytest.raises(ValidationError, match="one line"):
        Experiment(description="first\nsecond")
    with pytest.raises(ValidationError, match="bm25"):
        Experiment(
            description="x", method="bm25", embedder=Embedder.model_validate(GEMMA)
        )
    with pytest.raises(ValidationError, match="query"):
        Embedder.model_validate(GEMMA | {"query_template": "no placeholder"})
    with pytest.raises(CorpusError, match="lowercase"):
        load_experiment("../baseline")


def test_private_embedder_uses_recorded_contexts_and_its_templates(
    tmp_path, monkeypatch
):
    models = PrivateEmbedder(
        Embedder.model_validate(GEMMA),
        tmp_path,
        context={"request_model": "ctx"},
        notify=print,
    )
    # Contexts come from bench prepare, so the context model never starts.
    assert models.start("context") == {"request_model": "ctx"}
    assert models.process is None
    with pytest.raises(CorpusError, match="bench prepare"):
        models.context({})
    command = models.command("/bin/llama-server")
    assert command[:5] == [
        "/bin/llama-server",
        "-hf",
        "org/model-GGUF",
        "-hff",
        "model-Q8_0.gguf",
    ]
    assert "--embedding" in command
    assert "--no-mmproj" in command
    assert command[command.index("--pooling") + 1] == "mean"
    assert command[command.index("--ctx-size") + 1] == "2048"
    assert command[command.index("--ubatch-size") + 1] == "2048"
    sent = []

    def embed(texts, timeout):
        sent.extend(texts)
        return [[1.0, 0.0, 0.0]]

    monkeypatch.setattr(llama_client, "_embed_inputs_via_server", embed)
    models.query_embedding("What is a TLB?")
    assert sent == ["task: search result | query: What is a TLB?"]


def test_scope_points_sova_at_the_private_server_and_restores_it():
    production = (config.EMBEDDING_DIM, llama_client.EMBEDDING_SERVER_URL)
    llama_client._TOKENIZE_SEPARATOR_IDS["<|endoftext|>"] = 1
    with embedding_scope("http://127.0.0.1:9", "other", 3, 2048):
        assert config.EMBEDDING_DIM == db.EMBEDDING_DIM == search.EMBEDDING_DIM == 3
        assert llama_client.EMBEDDING_SERVER_URL == "http://127.0.0.1:9"
        assert llama_client._embedding_token_budget() < 2048
        # Token IDs of the production tokenizer do not apply here.
        assert not llama_client._TOKENIZE_SEPARATOR_IDS
    assert (config.EMBEDDING_DIM, llama_client.EMBEDDING_SERVER_URL) == production
    assert db.EMBEDDING_DIM == search.EMBEDDING_DIM == production[0]


class OtherEmbedder(FakeModels):
    def start(self, phase):
        signature = super().start(phase)
        if phase == "embedding":
            signature["document_template"] = "passage: {text}"
        return signature


def test_other_embedder_builds_its_own_index(tmp_path):
    snapshot = prepare(make_reading_inputs(tmp_path))
    production = build(snapshot)
    with embedding_scope("http://127.0.0.1:9", "other", 3, 2048):
        other = build(snapshot, OtherEmbedder(embedding_version="two"))
        assert other != production
        assert indexing.load_index(other)["inputs"]["embedding_dim"] == 3
        conn = db.connect_readonly(other.parent / "index.sqlite3")
        try:
            assert len(search.get_vector_candidates(conn, [1.0, 0.0, 0.0], 10)) == 2
        finally:
            conn.close()
    texts = [
        json.loads(path.read_bytes())["request"]["text"]
        for path in (snapshot.parents[2] / "embeddings").rglob("*.json")
    ]
    assert sum(text.startswith("passage: ") for text in texts) == 2
    assert len(texts) == 4


def test_table_marks_direct_adapter_times():
    row = {
        "name": "gemma",
        "date": "2026-10-24",
        "description": "Use a | smaller embedder.",
        "ndcg@10": {
            "en": {"mean": 0.5, "low": 0.4, "high": 0.6},
            "ru": None,
            "zh": 0.25,
        },
        "unjudged_queries": 3,
        "wrong_isa_results": 0.1,
        "warm_p50_ms": 120.0,
        "measurement": "direct_adapter",
        "server_rss_bytes": 2**31,
        "result": "results/gemma/2026-10-24-abc.json",
    }
    table = experiments_markdown(
        {"status": "ready", "suite_revision": "f" * 64, "experiments": [row]}
    )
    assert (
        "| 2026-10-24 | [gemma](results/gemma/2026-10-24-abc.json) "
        "| 0.500 (0.400–0.600) | – | 0.250 | 10.0% | 120 ms* | 2.0 GiB | 3 "
        "| Use a \\| smaller embedder. |" in table
    )
    assert "Direct adapter" in table
