"""Make passage snapshots with the Sova production chunker.

A snapshot does not change after it is written.
"""

import fcntl
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from benchmarks.corpus import CorpusError, SourceLock, _write_atomic
from benchmarks.reading import _digest
from sova import config, extract


class Tokenizer(Protocol):
    signature: dict

    def count(self, texts: list[str]) -> list[int]: ...

    def verify(self) -> None: ...


def _json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _hash(value: object) -> str:
    return _digest(_json(value))


def _load_inputs(manifest_path: Path) -> tuple[dict, dict[str, bytes]]:
    """Check all reading copies before the tokenizer starts or an output is written."""
    try:
        manifest = json.loads(manifest_path.read_bytes())
        lock_bytes = (manifest_path.parent.parent / "sources.lock.json").read_bytes()
        lock = SourceLock.model_validate_json(lock_bytes)
        if (
            manifest["status"] != "reading_copies_ready"
            or manifest["schema_version"] != 1
            or manifest["source_lock_sha256"] != _digest(lock_bytes)
            or manifest["corpus_id"] != lock.corpus_id
        ):
            raise ValueError("reading manifest does not match the source lock")
        documents = manifest["documents"]
        ids = [doc["source_id"] for doc in documents]
        sources = {source.source_id: source for source in lock.files}
        if sorted(ids) != sorted(sources):
            raise ValueError(
                "reading manifest must contain every locked source exactly once"
            )
        inputs = []
        markdowns = {}
        for doc in sorted(documents, key=lambda doc: doc["source_id"]):
            source_id = doc["source_id"]
            if (
                doc["markdown"] != f"{source_id}.md"
                or doc["sections_file"] != f"{source_id}.sections.json"
            ):
                raise ValueError(f"invalid reading filenames for {source_id}")
            markdown = (manifest_path.parent / doc["markdown"]).read_bytes()
            section_bytes = (manifest_path.parent / doc["sections_file"]).read_bytes()
            record = json.loads(section_bytes)
            if "structure_file" in doc:
                if doc["structure_file"] != f"{source_id}.structure.json":
                    raise ValueError("invalid structure filename")
                structure_bytes = (
                    manifest_path.parent / doc["structure_file"]
                ).read_bytes()
                if _digest(structure_bytes) != doc["structure_sha256"] or json.loads(
                    structure_bytes
                )["markdown_sha256"] != _digest(markdown):
                    raise ValueError("structure does not match reading copy")
            lines = markdown.decode().splitlines()
            sections = extract.parse_sections(lines)
            if (
                doc["source_sha256"] != sources[source_id].sha256
                or doc["markdown_sha256"] != _digest(markdown)
                or doc["sections_sha256"] != _digest(section_bytes)
                or {key: value for key, value in record.items() if key != "sections"}
                != {
                    key: value
                    for key, value in doc.items()
                    if key not in {"sections_file", "sections_sha256"}
                }
                or record["sections"] != sections
                or doc["lines"] != len(lines)
                or doc["characters"] != len(markdown.decode())
                or doc["section_count"] != len(sections)
                or not markdown.strip()
            ):
                raise ValueError(
                    f"reading content or sections do not match for {source_id}"
                )
            inputs.append(
                {
                    key: doc[key]
                    for key in (
                        "source_id",
                        "source_sha256",
                        "markdown_sha256",
                        "extractor",
                    )
                }
            )
            markdowns[source_id] = markdown
        return {"corpus_id": lock.corpus_id, "documents": inputs}, markdowns
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CorpusError(
            f"Invalid reading inputs; rerun bench prepare: {exc}"
        ) from exc


def _passage_id(passage: dict) -> str:
    return _hash(
        {
            key: passage[key]
            for key in (
                "source_id",
                "source_sha256",
                "markdown_sha256",
                "start_line",
                "end_line",
                "section_path",
                "text_sha256",
            )
        }
        | {"schema_version": 1}
        | {
            key: passage[key]
            for key in ("source_ranges", "context_scope")
            if key in passage
        }
    )


def _statistics(passages: list[dict], target: int) -> dict:
    counts = sorted(passage["token_count"] for passage in passages)
    return {
        "passages": len(counts),
        "tokens": sum(counts),
        "p50_tokens": counts[math.ceil(len(counts) * 0.50) - 1],
        "p95_tokens": counts[math.ceil(len(counts) * 0.95) - 1],
        "max_tokens": counts[-1],
        "over_target": sum(count > target for count in counts),
        "without_section": sum(not passage["section_path"] for passage in passages),
    }


def _validate_document(
    record: dict, source: dict, markdown: bytes, preparation_id: str, target: int
) -> None:
    if record["input"] != source or record["preparation_id"] != preparation_id:
        raise ValueError("passage checkpoint belongs to different inputs")
    lines = markdown.decode().splitlines()
    previous_end = 0
    ids = set()
    for passage in record["passages"]:
        start, end = passage["start_line"], passage["end_line"]
        if (
            type(start) is not int
            or type(end) is not int
            or not previous_end < start <= end <= len(lines)
        ):
            raise ValueError("invalid or overlapping passage range")
        previous_end = end
        # The recorded parser version made these section paths. Do not parse them again
        # with the current parser, because older snapshots then become unreadable.
        # Production changes blank separators, but it must keep the source words.
        ranges = extract.source_ranges(passage)
        if not ranges or any(
            not isinstance(span, list)
            or len(span) != 2
            or any(type(value) is not int for value in span)
            or not 1 <= span[0] <= span[1] <= len(lines)
            for span in ranges
        ):
            raise ValueError("invalid source fragment ranges")
        source_text = " ".join(
            "\n".join("\n".join(lines[a - 1 : b]) for a, b in ranges).split()
        )
        if (
            passage["source_id"] != source["source_id"]
            or passage["source_sha256"] != source["source_sha256"]
            or passage["markdown_sha256"] != source["markdown_sha256"]
            or not isinstance(passage["section_path"], str)
            or not passage["text"].strip()
            or " ".join(passage["text"].split()) != source_text
            or passage["text_sha256"] != _digest(passage["text"].encode())
            or passage["passage_id"] != _passage_id(passage)
            or passage["passage_id"] in ids
            or passage["word_count"] != len(passage["text"].split())
            or type(passage["token_count"]) is not int
            or passage["token_count"] <= 0
        ):
            raise ValueError("invalid passage content, provenance or identity")
        ids.add(passage["passage_id"])
    if not ids or record["statistics"] != _statistics(record["passages"], target):
        raise ValueError("empty or inconsistent passage checkpoint")
    if record["sha256"] != _hash(
        {key: value for key, value in record.items() if key != "sha256"}
    ):
        raise ValueError("passage checkpoint hash mismatch")


def load_snapshot(manifest_path: Path) -> dict:
    """Check a complete snapshot against its reading copies. Models do not start."""
    try:
        manifest = json.loads(manifest_path.read_bytes())
        inputs = manifest["inputs"]
        preparation_id = _hash(inputs)
        if (
            manifest["status"] != "passages_ready"
            or manifest["schema_version"] != 1
            or manifest["preparation_id"] != preparation_id
        ):
            raise ValueError("invalid passage manifest")
        if manifest["snapshot_sha256"] != _hash(
            {key: value for key, value in manifest.items() if key != "snapshot_sha256"}
        ):
            raise ValueError("passage manifest hash mismatch")
        if len(manifest["documents"]) != len(inputs["documents"]):
            raise ValueError("incomplete passage snapshot")
        all_passages = []
        target = inputs["pipeline"]["target_tokens"]
        for doc, source in zip(manifest["documents"], inputs["documents"], strict=True):
            source_id = source["source_id"]
            if doc["file"] != f"{source_id}.json":
                raise ValueError("invalid snapshot filenames")
            data = (manifest_path.parent / doc["file"]).read_bytes()
            markdown = (
                manifest_path.parents[2] / "reading" / f"{source_id}.md"
            ).read_bytes()
            if (
                _digest(data) != doc["sha256"]
                or _digest(markdown) != source["markdown_sha256"]
            ):
                raise ValueError(f"snapshot file hash mismatch for {source_id}")
            record = json.loads(data)
            _validate_document(record, source, markdown, preparation_id, target)
            all_passages.extend(record["passages"])
        if manifest["statistics"] != _statistics(all_passages, target):
            raise ValueError("inconsistent snapshot statistics")
        return manifest
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise CorpusError(f"Invalid passage snapshot {manifest_path}: {exc}") from exc


def prepare_passages(
    reading_manifest: Path,
    *,
    tokenizer: Tokenizer | None = None,
    target_tokens: int = config.CHUNK_TARGET_TOKENS,
    notify: Callable[[str], None] = print,
) -> Path:
    """Write each variant to its own snapshot. Save each document atomically."""
    inputs, markdowns = _load_inputs(reading_manifest)
    if target_tokens < 1:
        raise CorpusError("Passage target must be positive")
    if tokenizer is None:
        from benchmarks.tokenizer import ProductionTokenizer

        notify(
            "Checking the local production tokenizer (no embeddings or context generation)"
        )
        tokenizer = ProductionTokenizer()
    inputs["pipeline"] = {
        "schema_version": 1,
        "method": "sova.extract.chunk_text",
        "section_method": "sova.extract.find_section",
        "code_sha256": _digest(Path(extract.__file__).read_bytes()),
        "adapter_sha256": _digest(Path(__file__).read_bytes()),
        "target_tokens": target_tokens,
        "tokenizer": tokenizer.signature,
    }
    preparation_id = _hash(inputs)
    output = reading_manifest.parent.parent / "passages" / preparation_id
    output.mkdir(parents=True, exist_ok=True)
    # The OS releases the lock after Ctrl-C or a crash, so stale locks need no cleanup.
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CorpusError(
                "This passage snapshot is being prepared by another process"
            ) from exc
        ready = output / "manifest.json"
        if ready.exists():
            if load_snapshot(ready)["inputs"] != inputs:
                raise CorpusError(
                    "Passage snapshot directory contains different inputs"
                )
            notify(f"Verified passage snapshot {preparation_id}")
            return ready
        documents, all_passages = [], []
        for source in inputs["documents"]:
            source_id = source["source_id"]
            markdown = markdowns[source_id]
            path = output / f"{source_id}.json"
            record: dict | None = None
            if path.exists():
                try:
                    previous = json.loads(path.read_bytes())
                    _validate_document(
                        previous, source, markdown, preparation_id, target_tokens
                    )
                    record = previous
                except OSError, ValueError, KeyError, TypeError:
                    notify(f"Rebuilding incomplete or damaged checkpoint {source_id}")
            if record is None:
                notify(f"Chunking {source_id}")
                tokenizer.verify()
                lines = markdown.decode().splitlines()
                sections = extract.parse_sections(lines)
                try:
                    chunks = extract.chunk_text(
                        lines, target_tokens, count_tokens_batch=tokenizer.count
                    )
                    counts = tokenizer.count([chunk["text"] for chunk in chunks])
                    if len(counts) != len(chunks) or not chunks:
                        raise ValueError("empty chunks or invalid tokenizer response")
                    passages = []
                    for chunk, count in zip(chunks, counts, strict=True):
                        section = extract.find_section(
                            sections, chunk["start_line"], chunk["end_line"]
                        )
                        passage = (
                            dict(chunk)
                            | {
                                key: source[key]
                                for key in (
                                    "source_id",
                                    "source_sha256",
                                    "markdown_sha256",
                                )
                            }
                            | {
                                "section_path": sections[section]["path"]
                                if section is not None
                                else "",
                                "text_sha256": _digest(chunk["text"].encode()),
                                "token_count": count,
                                "context_scope": extract.context_scope(sections, chunk),
                            }
                        )
                        passage["passage_id"] = _passage_id(passage)
                        passages.append(passage)
                    record = {
                        "input": source,
                        "preparation_id": preparation_id,
                        "passages": passages,
                        "statistics": _statistics(passages, target_tokens),
                    }
                    record["sha256"] = _hash(record)
                    _validate_document(
                        record, source, markdown, preparation_id, target_tokens
                    )
                except (ValueError, RuntimeError) as exc:
                    raise CorpusError(
                        f"Cannot prepare passages for {source_id}: {exc}"
                    ) from exc
                tokenizer.verify()
                _write_atomic(path, _json(record))
            else:
                notify(f"Verified passages for {source_id}")
            documents.append({"file": path.name, "sha256": _digest(path.read_bytes())})
            all_passages.extend(record["passages"])
            stats = record["statistics"]
            notify(
                f"  {stats['passages']} passages; {stats['over_target']} over target; max {stats['max_tokens']} tokens"
            )
        tokenizer.verify()
        manifest = {
            "schema_version": 1,
            "status": "passages_ready",
            "preparation_id": preparation_id,
            "inputs": inputs,
            "coordinates": "1-based inclusive Markdown lines; not PDF pages",
            "documents": documents,
            "statistics": _statistics(all_passages, target_tokens),
        }
        manifest["snapshot_sha256"] = _hash(manifest)
        _write_atomic(ready, _json(manifest))
        return ready
