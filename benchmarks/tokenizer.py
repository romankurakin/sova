"""Identify and use the local production tokenizer. Embeddings are not made."""

import hashlib
import json
import re
import urllib.request
from pathlib import Path

from benchmarks.corpus import CorpusError
from sova import llama_client
from sova.external import ExternalToolError


class ProductionTokenizer:
    """Record the loaded GGUF. It must not change during preparation."""

    def __init__(self) -> None:
        self._files: dict[Path, tuple[tuple[int, int, int, int], str]] = {}
        try:
            if not llama_client._ensure_server(llama_client.EMBEDDING_SERVER_URL):
                raise CorpusError(
                    "Passage preparation needs Sova's local llama-server tokenizer. "
                    "Complete the model setup in the main README, then rerun bench prepare. "
                    "Prepared PDFs and Markdown are kept."
                )
            self.signature = self._identity()
        except (
            OSError,
            ValueError,
            llama_client.ServerError,
            ExternalToolError,
        ) as exc:
            raise CorpusError(f"Cannot initialize production tokenizer: {exc}") from exc
        # Production keeps separator token IDs in a cache.
        # They belong to this tokenizer.
        llama_client._TOKENIZE_SEPARATOR_IDS.clear()

    def _file_hash(self, path: Path) -> str:
        stat = path.stat()
        key = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        cached = self._files.get(path)
        if cached is not None and cached[0] == key:
            return cached[1]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = path.stat()
        if (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != key:
            raise CorpusError(
                "Tokenizer model changed while hashing; rerun preparation"
            )
        self._files[path] = (key, digest)
        return digest

    def _identity(self) -> dict:
        with urllib.request.urlopen(
            llama_client.EMBEDDING_SERVER_URL + "/props", timeout=10
        ) as response:
            props = json.load(response)
        if not isinstance(props, dict):
            raise CorpusError("Invalid tokenizer server properties")
        model = props.get("model_path")
        if not isinstance(model, str) or not model:
            raise CorpusError("Tokenizer server must expose model_path")
        path = Path(model)
        # The identity of a split GGUF contains all shards, in shard order.
        split = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", path.name)
        files = (
            [
                path.with_name(f"{split[1]}-{i:05d}-of-{int(split[3]):05d}.gguf")
                for i in range(1, int(split[3]) + 1)
            ]
            if split
            else [path]
        )
        return {
            "method": "sova.llama_client.get_token_counts_batch",
            "engine": "llama.cpp",
            "model_sha256": [self._file_hash(file) for file in files],
            "add_special": False,
            "parse_special": True,
        }

    def verify(self) -> None:
        try:
            if self._identity() != self.signature:
                raise CorpusError(
                    "Tokenizer changed during preparation; rerun bench prepare"
                )
        except (OSError, ValueError) as exc:
            raise CorpusError(f"Cannot verify production tokenizer: {exc}") from exc

    def count(self, texts: list[str]) -> list[int]:
        try:
            return llama_client.get_token_counts_batch(texts)
        except llama_client.ServerError as exc:
            raise CorpusError(f"Passage tokenization failed: {exc}") from exc
