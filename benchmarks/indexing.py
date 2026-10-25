"""Make contexts, vectors and the search database with the production models.

Each step continues after an interruption.
"""

import base64
import fcntl
import importlib.metadata
import json
import math
import sqlite3
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from benchmarks.corpus import CorpusError, SourceLock, _write_atomic
from benchmarks.models import ProductionModels
from benchmarks.passages import _hash, _json, load_snapshot
from benchmarks.reading import _digest
from sova import config, db, index_text, llama_client, search
from sova.extract import context_neighbors


def _read_object(path: Path, request: dict) -> dict | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_bytes())
        if value["request"] != request or value["sha256"] != _hash(
            {key: item for key, item in value.items() if key != "sha256"}
        ):
            raise ValueError("artifact hash or inputs differ")
        return value
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CorpusError(f"Damaged immutable artifact {path}: {exc}") from exc


def _save_object(path: Path, request: dict, **output) -> dict:
    value = {"request": request} | output
    value["sha256"] = _hash(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, _json(value))
    return value


def _load_passages(snapshot_path: Path) -> tuple[dict, list[dict]]:
    snapshot = load_snapshot(snapshot_path)
    lock = SourceLock.model_validate_json(
        (snapshot_path.parents[2] / "sources.lock.json").read_bytes()
    )
    sources = {source.source_id: source for source in lock.files}
    items = []
    for doc in snapshot["documents"]:
        record = json.loads((snapshot_path.parent / doc["file"]).read_bytes())
        source = sources.get(record["input"]["source_id"])
        if source is None or source.sha256 != record["input"]["source_sha256"]:
            raise CorpusError("Passage snapshot does not match the current source lock")
        passages = record["passages"]
        for i, passage in enumerate(passages):
            previous, following = context_neighbors(passages, i)
            items.append(
                passage
                | {
                    "document": Path(source.filename).stem,
                    "markdown_file": f"{source.source_id}.md",
                    "context_request": {
                        "doc_name": Path(source.filename).stem,
                        "section_title": passage["section_path"] or None,
                        "chunk_text": passage["text"],
                        "prev_chunk": previous,
                        "next_chunk": following,
                    },
                }
            )
    return snapshot, items


class IndexPreparation:
    def __init__(
        self, snapshot_path: Path, *, models=None, notify: Callable[[str], None] = print
    ) -> None:
        self.snapshot_path = snapshot_path
        self.snapshot, self.items = _load_passages(snapshot_path)
        self.root = snapshot_path.parents[2]
        self.models = models if models is not None else ProductionModels(notify)
        self.notify = notify
        self.signatures: dict[str, dict] = {}
        self.active_phase: str | None = None
        self.contexts: dict[str, dict] = {}
        self.vectors: dict[str, dict] = {}
        self.costs = {}

    def _start(self, phase: str) -> dict:
        self.notify(f"Checking {phase} model identity")
        signature = self.models.start(phase)
        previous = self.signatures.get(phase)
        if previous is not None and signature != previous:
            raise CorpusError(f"{phase} model changed between indexing stages")
        self.signatures[phase] = signature
        self.active_phase = phase
        return signature

    def prepare_contexts(self, items: list[dict]) -> None:
        stage_started = time.monotonic()
        signature = self.signatures.get("context") or self._start("context")
        created = 0
        for i, item in enumerate(items, 1):
            # The key is the exact request that the model receives, not the client code.
            body = llama_client.context_request_body(**item["context_request"])
            request = {"model": signature, "body": body}
            key = _hash(request)
            path = self.root / "contexts" / key[:2] / f"{key}.json"
            artifact = _read_object(path, request)
            if artifact is None:
                if self.active_phase != "context":
                    self._start("context")
                self.models.verify()
                context = self.models.context(item["context_request"]).strip()
                if not context:
                    raise CorpusError("Context model returned empty text")
                self.models.verify()
                artifact = _save_object(
                    path, request, context=context, engine=self.models.engine
                )
                created += 1
            if (
                not isinstance(artifact.get("context"), str)
                or not artifact["context"].strip()
            ):
                raise CorpusError(f"Invalid context artifact {path}")
            self.contexts[item["passage_id"]] = artifact
            if created or i == len(items):
                self.notify(
                    f"Contexts: {i}/{len(items)} ({created} generated, {i - created} reused)"
                )

        self.costs["contexts"] = {
            "elapsed_seconds": time.monotonic() - stage_started,
            "created": created,
            "reused": len(items) - created,
        }

    @staticmethod
    def _vector(artifact: dict) -> bytes:
        try:
            blob = base64.b64decode(artifact["vector_f32"], validate=True)
            vector = db.blob_to_embedding(blob)
            if (
                len(vector) != config.EMBEDDING_DIM
                or not all(math.isfinite(value) for value in vector)
                or not any(vector)
            ):
                raise ValueError("invalid vector dimension, finite values or norm")
            if _digest(blob) != artifact["vector_sha256"]:
                raise ValueError("vector hash mismatch")
            if (
                not isinstance(artifact["effective_text"], str)
                or not artifact["effective_text"]
            ):
                raise ValueError("missing actual embedding input")
            return blob
        except (KeyError, TypeError, ValueError) as exc:
            raise CorpusError(f"Invalid embedding artifact: {exc}") from exc

    def prepare_vectors(self, items: list[dict]) -> None:
        stage_started = time.monotonic()
        signature = self.signatures.get("embedding") or self._start("embedding")
        pending: list[tuple[dict, dict, Path]] = []
        aliases: dict[Path, list[str]] = defaultdict(list)
        for item in items:
            context = self.contexts[item["passage_id"]]["context"]
            text = index_text.contextualized_text(
                item["document"], item["section_path"], item["text"], context
            )
            # Some embedders need a document prefix. Production uses none.
            text = signature.get("document_template", "{text}").replace("{text}", text)
            request = {"model": signature, "text": text}
            key = _hash(request)
            path = self.root / "embeddings" / key[:2] / f"{key}.json"
            artifact = _read_object(path, request)
            if artifact is None:
                if path not in aliases:
                    pending.append((item, request, path))
                aliases[path].append(item["passage_id"])
            else:
                self._vector(artifact)
                self.vectors[item["passage_id"]] = artifact
        reused = sum(item["passage_id"] in self.vectors for item in items)
        self.notify(f"Embeddings: {reused}/{len(items)} reused")
        effective = {}

        def on_inputs(indices, texts):
            if len(indices) != len(texts):
                raise CorpusError("Invalid effective embedding inputs")
            effective.update(zip(indices, texts, strict=True))

        def on_batch(indices, vectors, stats):
            if len(indices) != len(vectors) or not indices:
                raise CorpusError("Invalid embedding batch")
            self.models.verify()
            for index, vector in zip(indices, vectors, strict=True):
                _item, request, path = pending[index]
                if index not in effective:
                    raise CorpusError(
                        "Embedding backend did not report its actual input"
                    )
                blob = db.embedding_to_blob(vector)
                output = {
                    "vector_f32": base64.b64encode(blob).decode(),
                    "vector_sha256": _digest(blob),
                    "effective_text": effective[index],
                    "input_changed": effective[index] != request["text"],
                    "engine": self.models.engine,
                }
                self._vector(output)
                artifact = _save_object(path, request, **output)
                for passage_id in aliases[path]:
                    self.vectors[passage_id] = artifact
            self.notify(
                f"Embeddings: {sum(item['passage_id'] in self.vectors for item in items)}/{len(items)}"
            )

        if pending:
            if self.active_phase != "embedding":
                self._start("embedding")
            self.models.verify()
            self.models.embed(
                [request["text"] for _, request, _ in pending], on_batch, on_inputs
            )
        if any(item["passage_id"] not in self.vectors for item in items):
            raise CorpusError("Embedding backend returned an incomplete result")

        self.costs["embeddings"] = {
            "elapsed_seconds": time.monotonic() - stage_started,
            "created": len(pending),
            "reused": reused,
        }

    def build_database(self) -> Path:
        stage_started = time.monotonic()
        inputs = {
            "snapshot": self.snapshot["snapshot_sha256"],
            "models": self.signatures,
            "code": {
                name: _digest(Path(module.__file__).read_bytes())
                for name, module in (
                    ("db", db),
                    ("index_text", index_text),
                    ("search", search),
                )
            },
            "adapter_sha256": _digest(Path(__file__).read_bytes()),
            "sqlite_vector": importlib.metadata.version("sqliteai-vector"),
            "schema_version": db.SCHEMA_VERSION,
            "embedding_dim": config.EMBEDDING_DIM,
            "passages": [
                {
                    "id": item["passage_id"],
                    "document": item["document"],
                    "context": _hash(
                        {
                            key: self.contexts[item["passage_id"]][key]
                            for key in ("request", "context")
                        }
                    ),
                    "embedding": _hash(
                        {
                            key: self.vectors[item["passage_id"]][key]
                            for key in ("request", "effective_text", "vector_sha256")
                        }
                    ),
                }
                for item in self.items
            ],
        }
        output = self.root / "indexes" / _hash(inputs)
        output.mkdir(parents=True, exist_ok=True)
        ready = output / "manifest.json"
        if ready.exists():
            load_index(ready)
            self.notify("Verified complete search index")
            return ready
        path = output / "index.sqlite3"
        conn = db.init_db(path)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS benchmark_passages (chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id), passage_id TEXT UNIQUE NOT NULL)"
            )
            by_doc = defaultdict(list)
            for item in self.items:
                by_doc[item["source_id"]].append(item)
            # The database is fast to make and uses one transaction. Model outputs
            # do not change. They stay after an interrupted build or quantization.
            with conn:
                conn.execute("DELETE FROM benchmark_passages")
                conn.execute("DELETE FROM chunk_contexts")
                conn.execute("DELETE FROM chunks")
                conn.execute("DELETE FROM sections")
                conn.execute("DELETE FROM documents")
                for doc_id, items in enumerate(by_doc.values(), 1):
                    first = items[0]
                    conn.execute(
                        "INSERT INTO documents (id, name, path, line_count, expected_chunks, source_signature, chunk_signature) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            doc_id,
                            first["document"],
                            str(Path("../../reading") / first["markdown_file"]),
                            len(
                                (self.root / "reading" / first["markdown_file"])
                                .read_text()
                                .splitlines()
                            ),
                            len(items),
                            first["source_sha256"],
                            self.snapshot["preparation_id"],
                        ),
                    )
                    for item in items:
                        pid = item["passage_id"]
                        section_id = None
                        if item["section_path"]:
                            section_id = conn.execute(
                                "INSERT INTO sections (doc_id, title, level, start_line, end_line) VALUES (?, ?, ?, ?, ?)",
                                (
                                    doc_id,
                                    item["section_path"],
                                    item["section_path"].count(" > ") + 1,
                                    item["start_line"],
                                    item["end_line"],
                                ),
                            ).lastrowid
                        context = self.contexts[pid]["context"]
                        chunk_id = conn.execute(
                            "INSERT INTO chunks (doc_id, section_id, start_line, end_line, word_count, text, embedding, embedding_signature, section_path, search_prefix, is_index) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                doc_id,
                                section_id,
                                item["start_line"],
                                item["end_line"],
                                item["word_count"],
                                item["text"],
                                self._vector(self.vectors[pid]),
                                _hash(self.signatures["embedding"]),
                                item["section_path"],
                                index_text.contextualized_prefix(
                                    item["document"], item["section_path"], context
                                ),
                                int(search.is_index_like(item["text"])),
                            ),
                        ).lastrowid
                        conn.execute(
                            "INSERT INTO chunk_contexts (chunk_id, context, model, pipeline_signature) VALUES (?, ?, ?, ?)",
                            (
                                chunk_id,
                                context,
                                self.signatures["context"].get(
                                    "request_model", "synthetic"
                                ),
                                _hash(self.signatures["context"]),
                            ),
                        )
                        conn.execute(
                            "INSERT INTO benchmark_passages VALUES (?, ?)",
                            (chunk_id, pid),
                        )
            self.notify("Building production vector and full-text indexes")
            db.quantize_vectors(conn)
            digest = _logical_digest(conn)
        finally:
            conn.close()
        manifest = {
            "schema_version": 1,
            "status": "index_ready",
            "inputs": inputs,
            "database": path.name,
            "document_path_base": "database_directory",
            "logical_sha256": digest,
            "passages": len(self.items),
            "preparation_cost": self.costs
            | {"database": {"elapsed_seconds": time.monotonic() - stage_started}},
            "changed_embedding_inputs": sum(
                self.vectors[item["passage_id"]]["input_changed"] for item in self.items
            ),
            "context_bodies_sha256": context_bodies_digest(self.snapshot_path),
        }
        manifest["sha256"] = _hash(manifest)
        _write_atomic(ready, _json(manifest))
        return ready


def context_bodies_digest(snapshot_path: Path) -> str:
    """Return the hash of all context requests. The model does not start."""
    _, items = _load_passages(snapshot_path)
    return _hash(
        [llama_client.context_request_body(**item["context_request"]) for item in items]
    )


def _logical_digest(conn: sqlite3.Connection) -> str:
    # The digest does not use page layout, row IDs, timestamps or the query cache.
    records = conn.execute(
        "SELECT b.passage_id, d.name, c.start_line, c.end_line, c.text, c.section_path, c.search_prefix, c.is_index, x.context, c.embedding FROM benchmark_passages b JOIN chunks c ON c.id=b.chunk_id JOIN documents d ON d.id=c.doc_id JOIN chunk_contexts x ON x.chunk_id=c.id ORDER BY b.passage_id"
    ).fetchall()
    return _hash([list(row[:-1]) + [_digest(row[-1])] for row in records])


def load_index(manifest_path: Path) -> dict:
    try:
        manifest = json.loads(manifest_path.read_bytes())
        if (
            manifest["status"] != "index_ready"
            or manifest["database"] != "index.sqlite3"
            or manifest["sha256"]
            != _hash({key: value for key, value in manifest.items() if key != "sha256"})
        ):
            raise ValueError("invalid index manifest")
        conn = db.connect_readonly(manifest_path.parent / manifest["database"])
        try:
            if (
                conn.execute("PRAGMA quick_check").fetchone()[0] != "ok"
                or _logical_digest(conn) != manifest["logical_sha256"]
            ):
                raise ValueError("database content differs from ready manifest")
            counts = [
                conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("chunks", "chunk_contexts", "benchmark_passages")
            ]
            if (
                any(count != manifest["passages"] for count in counts)
                or db.get_meta(conn, db.VECTOR_INDEX_STATE_KEY) != "ready"
            ):
                raise ValueError("incomplete search index")
        finally:
            conn.close()
        return manifest
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
        raise CorpusError(f"Invalid search index {manifest_path}: {exc}") from exc


def prepare_index(
    snapshot_path: Path, *, models=None, notify: Callable[[str], None] = print
) -> Path:
    preparation = IndexPreparation(snapshot_path, models=models, notify=notify)
    lock_path = preparation.root / "index-preparation.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CorpusError("Another benchmark indexing process is running") from exc
        try:
            preparation.prepare_contexts(preparation.items)
            preparation.prepare_vectors(preparation.items)
            return preparation.build_database()
        except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
            raise CorpusError(
                f"Index preparation failed; rerun bench prepare to resume: {exc}"
            ) from exc
        finally:
            preparation.models.close()
