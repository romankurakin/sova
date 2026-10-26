"""Run an experiment embedder in a private llama-server.

Sova keeps its embedding settings in module constants. While the private server
runs, those constants refer to it, so the production text preparation, vector
storage and search code stay the same.
"""

import os
import shutil
import socket
import subprocess
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from benchmarks.corpus import CorpusError
from benchmarks.experiments import Embedder
from benchmarks.models import ENGINE_FIELDS, ModelFingerprint
from sova import config, db, llama_client, search

# The same as the production embedding service in sova/install.py.
SERVER_ENV = {"MallocNanoZone": "0", "GGML_METAL_NO_RESIDENCY": "1"}


@contextmanager
def embedding_scope(url: str, model: str, dimensions: int, context_size: int):
    """Make the Sova embedding code use another server, model and vector size."""
    with ExitStack() as stack:
        for module in (config, db, search, llama_client):
            stack.enter_context(patch.object(module, "EMBEDDING_DIM", dimensions))
        stack.enter_context(patch.object(llama_client, "EMBEDDING_SERVER_URL", url))
        stack.enter_context(patch.object(llama_client, "EMBEDDING_MODEL", model))
        stack.enter_context(
            patch.object(llama_client, "_configured_ctx_size", lambda _: context_size)
        )
        # Separator token IDs belong to one tokenizer.
        llama_client._TOKENIZE_SEPARATOR_IDS.clear()
        stack.callback(llama_client._TOKENIZE_SEPARATOR_IDS.clear)
        yield


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class PrivateEmbedder:
    """Models for one experiment: recorded contexts and a private embedder."""

    def __init__(
        self, embedder: Embedder, root: Path, *, context: dict, notify=print
    ) -> None:
        self.embedder = embedder
        self.context_signature = context
        self.log = root / "logs" / "embedder.log"
        self.notify = notify
        self.url = f"http://127.0.0.1:{_free_port()}"
        self.process: subprocess.Popen | None = None
        self.fingerprint: ModelFingerprint | None = None
        self.engine: dict | None = None
        self.scope = ExitStack()

    def settings(self) -> dict[str, str]:
        return {
            "--pooling": self.embedder.pooling,
            "--flash-attn": "off",
            "--ctx-size": str(self.embedder.context_size),
            # Bidirectional embedders such as EmbeddingGemma must see a whole
            # input in one micro-batch; llama-server defaults to 512 tokens.
            "--batch-size": str(self.embedder.context_size),
            "--ubatch-size": str(self.embedder.context_size),
            "--parallel": "1",
        }

    def command(self, executable: str) -> list[str]:
        port = self.url.rsplit(":", 1)[1]
        args = [executable, "-hf", self.embedder.repository]
        args += ["-hff", self.embedder.file, "--host", "127.0.0.1", "--port", port]
        args += ["--embedding", *(x for pair in self.settings().items() for x in pair)]
        # Experiments embed text only. Without this flag, -hf also downloads and
        # loads a vision projector from multimodal repos such as EmbeddingGemma 2.
        return args + ["--no-mmproj", "--cache-ram", "0"]

    def _launch(self) -> None:
        executable = shutil.which("llama-server")
        if executable is None:
            raise CorpusError("llama-server is not installed")
        # Free the memory that the production models use.
        for url in (config.EMBEDDING_SERVER_URL, config.CONTEXT_SERVER_URL):
            llama_client.stop_server(url, suppress_interrupt=True)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.notify(
            f"Starting {self.embedder.model}. The first start downloads the model."
        )
        with self.log.open("ab") as log:
            self.process = subprocess.Popen(
                self.command(executable),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=os.environ | SERVER_ENV,
            )
        while not llama_client._health_ok(self.url, timeout=3):
            if self.process.poll() is not None:
                tail = self.log.read_text(errors="replace").splitlines()[-5:]
                raise CorpusError(
                    f"Embedder server stopped; see {self.log}: " + " | ".join(tail)
                )
            time.sleep(1)

    def start(self, phase: str) -> dict:
        if phase == "context":
            # Experiments use the contexts that bench prepare made.
            return self.context_signature
        if phase != "embedding":
            raise ValueError(f"Unknown model phase: {phase}")
        if self.process is None or self.process.poll() is not None:
            self.close()
            self._launch()
            self.scope.enter_context(
                embedding_scope(
                    self.url,
                    self.embedder.model,
                    self.embedder.dimensions,
                    self.embedder.context_size,
                )
            )
        self.fingerprint = ModelFingerprint(
            self.url, settings=self.settings(), repository=self.embedder.repository
        )
        signature = self.fingerprint.signature
        self.engine = {key: signature[key] for key in ENGINE_FIELDS}
        return {
            key: value for key, value in signature.items() if key not in ENGINE_FIELDS
        } | {
            "method": "sova.llama_client.get_embeddings_batch",
            "request_model": self.embedder.model,
            "embedding_dim": self.embedder.dimensions,
            "token_budget": llama_client._embedding_token_budget(),
            "query_template": self.embedder.query_template,
            "document_template": self.embedder.document_template,
        }

    def verify(self) -> None:
        if self.fingerprint is None or self.process is None:
            raise CorpusError("Embedder has not started")
        if self.process.poll() is not None:
            raise CorpusError(f"Embedder server stopped; see {self.log}")
        self.fingerprint.verify()

    def context(self, request: dict) -> str:
        raise CorpusError("A context is missing. Run bench prepare first.")

    def embed(self, texts: list[str], on_batch, on_inputs) -> None:
        llama_client.get_embeddings_batch(texts, on_batch=on_batch, on_inputs=on_inputs)

    def query_embedding(self, query: str) -> list[float]:
        prompt = self.embedder.query_template.replace("{task}", llama_client.QUERY_TASK)
        prompt = prompt.replace("{query}", query)
        return llama_client._embed_inputs_via_server([prompt], timeout=45.0)[0]

    def close(self) -> None:
        self.scope.close()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None
