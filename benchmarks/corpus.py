"""Download the PDFs in the source lock and check their size and hash."""

import hashlib
import http.client
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

import pymupdf
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_LOCK = Path(__file__).with_name("sources.lock.json")
_BLOCK_SIZE = 1024 * 1024


class CorpusError(RuntimeError):
    """The benchmark cannot prepare or check the source documents."""


class Source(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    version: str
    license: str


class SourceFile(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    source_id: str
    filename: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*\.pdf$")
    url: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bytes: int = Field(gt=0)
    pages: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_url(self) -> Self:
        url = urlsplit(self.url)
        if url.scheme not in {"http", "https"} or not url.hostname:
            raise ValueError("source URL must be HTTP(S)")
        return self


class SourceLock(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    schema_version: Literal[1]
    corpus_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    sources: list[Source] = Field(min_length=1)
    files: list[SourceFile] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_membership(self) -> Self:
        ids = [source.id for source in self.sources]
        file_ids = [file.source_id for file in self.files]
        names = [file.filename.casefold() for file in self.files]
        if len(set(ids)) != len(ids) or sorted(ids) != sorted(file_ids):
            raise ValueError("each source must have exactly one PDF")
        if len(set(names)) != len(names):
            raise ValueError("PDF filenames must be unique")
        return self


def _inspect_pdf(path: Path, expected: SourceFile) -> None:
    if path.stat().st_size != expected.bytes:
        raise CorpusError(f"{expected.filename}: byte size differs from source lock")
    with path.open("rb") as stream:
        signature = stream.read(5)
        stream.seek(0)
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected.sha256:
        raise CorpusError(f"{expected.filename}: SHA-256 differs from source lock")
    if signature != b"%PDF-":
        raise CorpusError(f"{expected.filename}: not a PDF")
    try:
        with pymupdf.open(path) as document:
            if not document.is_pdf or document.is_encrypted:
                raise CorpusError(f"{expected.filename}: unreadable or encrypted PDF")
            if document.page_count != expected.pages:
                raise CorpusError(
                    f"{expected.filename}: page count differs from source lock"
                )
    except CorpusError:
        raise
    except Exception as exc:
        raise CorpusError(f"{expected.filename}: cannot read PDF: {exc}") from exc


def _fetch_file(
    expected: SourceFile,
    destination: Path,
    *,
    notify: Callable[[str], None],
) -> None:
    if destination.exists():
        _inspect_pdf(destination, expected)
        notify(f"Verified cached {expected.filename} ({expected.pages} pages)")
        return

    notify(f"Downloading {expected.filename}")
    request = urllib.request.Request(
        expected.url, headers={"User-Agent": "sova-benchmark/2"}
    )
    incoming = urllib.request.urlopen(request, timeout=30)

    temporary: Path | None = None
    try:
        with (
            incoming,
            tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{expected.filename}.",
                suffix=".part",
                delete=False,
            ) as output,
        ):
            temporary = Path(output.name)
            size = 0
            while block := incoming.read(_BLOCK_SIZE):
                size += len(block)
                if size > expected.bytes:
                    raise CorpusError(
                        f"{expected.filename}: download exceeds locked size"
                    )
                output.write(block)
        _inspect_pdf(temporary, expected)
        temporary.replace(destination)
        notify(f"Verified {expected.filename} ({expected.pages} pages)")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_atomic(path: Path, data: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_sources(
    root: Path,
    *,
    lock_path: Path = DEFAULT_LOCK,
    notify: Callable[[str], None] = print,
) -> None:
    """Download the missing PDFs. Check all PDFs and copy the lock to the root."""
    try:
        lock_bytes = lock_path.read_bytes()
        lock = SourceLock.model_validate_json(lock_bytes)
    except (OSError, ValueError) as exc:
        raise CorpusError(f"Invalid source lock: {exc}") from exc
    sources_dir = root / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    for expected in lock.files:
        try:
            _fetch_file(expected, sources_dir / expected.filename, notify=notify)
        except (OSError, ValueError, http.client.HTTPException) as exc:
            raise CorpusError(f"{expected.filename}: {exc}") from exc
    _write_atomic(root / "sources.lock.json", lock_bytes)
