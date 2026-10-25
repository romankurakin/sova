"""Load experiment files: search settings and a one-sentence description."""

import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from benchmarks.corpus import CorpusError
from benchmarks.records import Identifier, Record
from sova import search

EXPERIMENT_DIR = Path(__file__).with_name("experiments")


class Embedder(Record):
    """A GGUF embedding model that runs in a private llama-server."""

    repository: Identifier
    file: Identifier
    dimensions: int = Field(ge=1)
    pooling: Literal["last", "mean", "cls"] = "last"
    context_size: int = Field(default=4096, ge=512)
    # {task} is the production retrieval task. {query} is the search text.
    query_template: str = "Instruct: {task}\nQuery: {query}"
    document_template: str = "{text}"

    @model_validator(mode="after")
    def templates(self):
        if "{query}" not in self.query_template:
            raise ValueError("query_template must contain {query}")
        if "{text}" not in self.document_template:
            raise ValueError("document_template must contain {text}")
        return self

    @property
    def model(self) -> str:
        return f"{self.repository}/{self.file}"


class Experiment(Record):
    schema_version: Literal[1] = 1
    description: str = Field(min_length=1)
    method: Literal["hybrid", "dense", "bm25"] = "hybrid"
    output_k: int = Field(default=10, ge=10)
    depth: int | None = Field(default=None, ge=10)
    ranking: dict[str, int | float] = Field(default_factory=dict)
    embedder: Embedder | None = None
    cache_modes: tuple[Literal["warm", "cached", "cold"], ...] = (
        "warm",
        "cached",
        "cold",
    )
    passes: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def settings_are_valid(self):
        if "\n" in self.description:
            raise ValueError("The description must be one sentence on one line")
        if (
            not self.cache_modes
            or len(set(self.cache_modes)) != len(self.cache_modes)
            or "warm" not in self.cache_modes
        ):
            raise ValueError("Cache modes must be unique and include warm")
        for key, value in self.ranking.items():
            if (
                not key.startswith("SEARCH_")
                or not hasattr(search, key)
                or not math.isfinite(value)
            ):
                raise ValueError(f"Unsupported ranking setting: {key}")
            if type(getattr(search, key)) is int and type(value) is not int:
                raise ValueError(f"{key} requires an integer")
        if self.depth is not None and self.depth < self.output_k:
            raise ValueError("Retrieval depth cannot be less than output_k")
        if self.embedder is not None and self.method == "bm25":
            raise ValueError("An embedder has no effect on bm25 search")
        return self

    @property
    def native(self) -> bool:
        """The production CLI can run this setup without changes."""
        return self.method == "hybrid" and self.depth is None and not self.embedder

    def settings(self) -> dict:
        # The description is not a setting. A new text does not start a new run.
        return self.model_dump(mode="json", exclude={"description"})


def load_experiment(name: str) -> Experiment:
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in name):
        raise CorpusError(
            "Experiment name must contain lowercase letters, numbers, hyphens or underscores"
        )
    try:
        return Experiment.model_validate_json(
            (EXPERIMENT_DIR / f"{name}.json").read_bytes()
        )
    except (OSError, ValueError) as exc:
        raise CorpusError(f"Cannot load experiment {name}: {exc}") from exc
