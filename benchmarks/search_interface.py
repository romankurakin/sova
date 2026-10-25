"""Adapters use the Sova retrieval code and return passage IDs in rank order."""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from benchmarks.corpus import CorpusError
from benchmarks.embedders import PrivateEmbedder
from benchmarks.experiments import Experiment
from benchmarks.indexing import _load_passages, load_index, prepare_index
from benchmarks.models import ProductionModels
from benchmarks.passages import _hash
from benchmarks.records import SearchRun
from benchmarks.storage import read
from sova import config, db, llama_client, search


class SovaBackend:
    def __init__(self, index: Path, snapshot: Path, *, models=None):
        self.index_path = index
        self.manifest = load_index(index)
        self.snapshot, items = _load_passages(snapshot)
        if self.snapshot["snapshot_sha256"] != self.manifest["inputs"]["snapshot"]:
            raise CorpusError("Index and passage snapshot differ")
        self.items = {item["passage_id"]: item for item in items}
        self.conn = db.connect_readonly(index.parent / self.manifest["database"])
        self.rows = dict(
            self.conn.execute("SELECT chunk_id, passage_id FROM benchmark_passages")
        )
        self.models = models or ProductionModels()
        self.started = False

    def start(self):
        if not self.started:
            actual = self.models.start("embedding")
            if actual != self.manifest["inputs"]["models"]["embedding"]:
                self.models.close()
                raise CorpusError(
                    "Query embedding model differs from the indexed vectors; prepare a matching variant"
                )
            self.started = True

    def close(self):
        self.conn.close()
        if self.started:
            self.models.close()
            self.started = False

    def query_embedding(self, query):
        self.start()
        self.models.verify()
        return self.models.query_embedding(query)

    def retrieve(self, query, *, method="hybrid", limit=10, depth=None, embedding=None):
        if method == "bm25":
            ids = [row[0] for row in search.search_fts(self.conn, query, limit)]
        else:
            embedding = (
                embedding if embedding is not None else self.query_embedding(query)
            )
            vectors = search.get_vector_candidates(
                self.conn, embedding, limit, candidates=depth
            )
            if method == "dense":
                ids = [row[0] for row in vectors[:limit]]
            elif method == "hybrid":
                original_fts = search.search_fts
                if depth is None:
                    hits, _, _ = search.fuse_and_rank(self.conn, vectors, query, limit)
                else:
                    # Only the FTS depth changes here. The native CLI does not use
                    # this path.
                    with patch.object(
                        search,
                        "search_fts",
                        lambda conn, query, limit: original_fts(conn, query, depth),
                    ):
                        hits, _, _ = search.fuse_and_rank(
                            self.conn, vectors, query, limit
                        )
                ids = [hit["chunk_id"] for hit in hits]
            else:
                raise CorpusError(f"Unsupported search method: {method}")
        return [self.rows[cid] for cid in ids]

    def run(self, query, ids):
        return SearchRun(
            query_id=query.id,
            query_sha256=_hash(query.judge_input()),
            hits=tuple(ids),
        )

    def from_cli(self, query, results):
        by_location = {
            (item["document"], item["start_line"], item["end_line"]): pid
            for pid, item in self.items.items()
        }
        ids = []
        for rank, result in enumerate(results, 1):
            key = (result["document"], result["start_line"], result["end_line"])
            if result["rank"] != rank or key not in by_location:
                raise CorpusError("Invalid CLI result rank or source location")
            if result["text"] != self.items[by_location[key]]["text"]:
                raise CorpusError("CLI changed the displayed passage text")
            ids.append(by_location[key])
        return self.run(query, ids)


@contextmanager
def open_backend(root: Path, experiment: Experiment, *, notify=print):
    """Open the prepared index, or an index for the experiment embedder."""
    state = read(root / "preparation.json")
    index, snapshot = root / state["index"], root / state["passages"]
    models = None
    try:
        if experiment.embedder is not None:
            models = PrivateEmbedder(
                experiment.embedder,
                root,
                context=load_index(index)["inputs"]["models"]["context"],
                notify=notify,
            )
            index = prepare_index(snapshot, models=models, notify=notify)
        backend = SovaBackend(index, snapshot, models=models)
        try:
            yield backend
        finally:
            backend.close()
    finally:
        if models is not None:
            models.close()


def code_identity() -> dict:
    from benchmarks import worker
    from benchmarks.reading import _digest
    from sova import cli, ui

    return {
        module.__name__: _digest(Path(module.__file__).read_bytes())
        for module in (search, llama_client, db, config, cli, ui, worker)
    }
