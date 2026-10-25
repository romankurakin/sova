"""Call the production models and record their local identities."""

import hashlib
import json
import plistlib
import re
import urllib.request
from pathlib import Path

from benchmarks.corpus import CorpusError
from benchmarks.passages import _hash
from sova import config, llama_client

# These fields record how an output was made. They are not part of the key. The key
# uses the weights' hash, the chat template and the server settings that Sova sets.
ENGINE_FIELDS = ("build", "default_generation_settings", "model")


class ModelFingerprint:
    def __init__(
        self,
        url: str,
        *,
        settings: dict[str, str] | None = None,
        repository: str | None = None,
    ) -> None:
        self.url = url
        # A private server has no launchd file, so its caller gives the settings.
        self.launch = None if settings is None else (settings, repository)
        self._files: dict[Path, tuple[tuple, str]] = {}
        self.signature = self.read()

    def _hash_file(self, path: Path) -> str:
        stat = path.stat()
        stamp = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        cached = self._files.get(path)
        if cached and cached[0] == stamp:
            return cached[1]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = path.stat()
        if stamp != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise CorpusError("Model file changed while hashing")
        self._files[path] = (stamp, digest)
        return digest

    def read(self) -> dict:
        with urllib.request.urlopen(self.url + "/props", timeout=10) as response:
            props = json.load(response)
        if (
            not isinstance(props, dict)
            or not props.get("model_path")
            or not props.get("build_info")
        ):
            raise CorpusError("Model server must expose model_path and build_info")
        path = Path(props["model_path"])
        split = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", path.name)
        files = (
            [path]
            if split is None
            else [
                path.with_name(f"{split[1]}-{i:05d}-of-{int(split[3]):05d}.gguf")
                for i in range(1, int(split[3]) + 1)
            ]
        )
        label = llama_client._SERVICE_LABELS.get(self.url)
        launch = Path.home() / "Library/LaunchAgents" / f"{label}.plist"
        settings, repository = self.launch or ({}, None)
        if self.launch is None and launch.exists():
            args = plistlib.loads(launch.read_bytes()).get("ProgramArguments", [])
            for i, arg in enumerate(args[:-1]):
                if arg == "-hf":
                    repository = args[i + 1]
                if arg in {
                    "--pooling",
                    "--embd-normalize",
                    "--ctx-size",
                    "--flash-attn",
                    "--batch-size",
                    "--ubatch-size",
                    "--parallel",
                }:
                    settings[arg] = args[i + 1]
        return {
            "engine": "llama.cpp",
            "build": props["build_info"],
            "model_sha256": [self._hash_file(file) for file in files],
            "default_generation_settings": props.get("default_generation_settings"),
            "chat_template_sha256": _hash(props.get("chat_template")),
            "managed_service_settings": settings,
            "model": f"{repository}/{path.name}" if repository else path.name,
        }

    def verify(self) -> None:
        if self.read() != self.signature:
            raise CorpusError(
                "Model or server settings changed during indexing; rerun bench prepare"
            )


class ProductionModels:
    """Load one model at a time and keep the production inference settings."""

    def __init__(self, notify=print) -> None:
        self.notify = notify
        self.phase: str | None = None
        self.fingerprint: ModelFingerprint | None = None
        self.engine: dict | None = None

    def start(self, phase: str) -> dict:
        if phase not in {"context", "embedding"}:
            raise ValueError(f"Unknown model phase: {phase}")
        url = (
            config.CONTEXT_SERVER_URL
            if phase == "context"
            else config.EMBEDDING_SERVER_URL
        )
        other = (
            config.EMBEDDING_SERVER_URL
            if phase == "context"
            else config.CONTEXT_SERVER_URL
        )
        if self.phase != phase:
            llama_client.stop_server(other, suppress_interrupt=True)
            self.phase = phase
        ok, message = llama_client.check_servers(
            mode="index_context" if phase == "context" else "index_embed",
            on_status=self.notify,
        )
        if not ok:
            raise CorpusError(message)
        self.fingerprint = ModelFingerprint(url)
        if phase == "embedding":
            llama_client._TOKENIZE_SEPARATOR_IDS.clear()
        signature = self.fingerprint.signature
        self.engine = {key: signature[key] for key in ENGINE_FIELDS}
        return {
            key: value for key, value in signature.items() if key not in ENGINE_FIELDS
        } | {
            "method": f"sova.llama_client.{'generate_context' if phase == 'context' else 'get_embeddings_batch'}",
            "request_model": config.CONTEXT_MODEL
            if phase == "context"
            else config.EMBEDDING_MODEL,
            "embedding_dim": config.EMBEDDING_DIM if phase == "embedding" else None,
            "token_budget": llama_client._embedding_token_budget()
            if phase == "embedding"
            else None,
        }

    def verify(self) -> None:
        if self.fingerprint is None:
            raise CorpusError("Model phase has not started")
        self.fingerprint.verify()

    def context(self, request: dict) -> str:
        try:
            return llama_client.generate_context(**request)
        except Exception as exc:
            raise CorpusError(f"Production context generation failed: {exc}") from exc

    def embed(self, texts: list[str], on_batch, on_inputs) -> None:
        llama_client.get_embeddings_batch(texts, on_batch=on_batch, on_inputs=on_inputs)

    def query_embedding(self, query: str) -> list[float]:
        return llama_client.get_query_embedding(query)

    def close(self) -> None:
        if self.phase is not None:
            url = (
                config.CONTEXT_SERVER_URL
                if self.phase == "context"
                else config.EMBEDDING_SERVER_URL
            )
            llama_client.stop_server(url, suppress_interrupt=True)
            self.phase = None
