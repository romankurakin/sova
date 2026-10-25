"""Run records that the runner, the candidate pool and the reports use."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from benchmarks.passages import _hash

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Identifier = Annotated[str, Field(min_length=1)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    @property
    def sha256(self) -> str:
        return _hash(self.model_dump(mode="json"))


class SearchRun(Record):
    schema_version: Literal[2] = 2
    query_id: Identifier
    query_sha256: (
        Digest  # Hash of Query.judge_input(), which contains the text and the intent.
    )
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
    hits: tuple[
        Identifier, ...
    ]  # Passage IDs in rank order. Repeated IDs stay in the list.

    @model_validator(mode="after")
    def error_state(self) -> Self:
        if (self.status == "error") != bool(self.error):
            raise ValueError("Failed runs must record an error; successful runs cannot")
        return self
