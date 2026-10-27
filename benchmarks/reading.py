"""Make full reading copies with the Sova production PDF extractor."""

import hashlib
import importlib.metadata
import json
from collections.abc import Callable
from pathlib import Path

from benchmarks.corpus import CorpusError, SourceLock, _inspect_pdf, _write_atomic
from sova import extract, pdf_structure


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _extractor_signature() -> dict:
    return {
        "code_sha256": _digest(Path(extract.__file__).read_bytes()),
        "structure_code_sha256": _digest(Path(pdf_structure.__file__).read_bytes()),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("PyMuPDF", "pymupdf4llm", "pymupdf-layout", "indra-ocr")
        },
        "method": "sova.extract.extract_pdf",
        "section_parser": "sova.extract.parse_sections",
    }


def prepare_reading(root: Path, *, notify: Callable[[str], None] = print) -> Path:
    """Extract Markdown from each checked PDF. Use an earlier copy if it matches."""
    try:
        lock_bytes = (root / "sources.lock.json").read_bytes()
        lock = SourceLock.model_validate_json(lock_bytes)
    except (OSError, ValueError) as exc:
        raise CorpusError(f"Invalid source lock copy: {exc}") from exc

    output = root / "reading"
    output.mkdir(exist_ok=True)
    ready = output / "manifest.json"
    ready.unlink(missing_ok=True)
    signature = _extractor_signature()
    documents = []
    for source in lock.files:
        pdf = root / "sources" / source.filename
        # Check the exact bytes, also when the run uses an existing reading copy.
        _inspect_pdf(pdf, source)
        md_path = output / f"{source.source_id}.md"
        sections_path = output / f"{source.source_id}.sections.json"
        structure_path = pdf_structure.structure_path(md_path)
        record = None
        try:
            previous = json.loads(sections_path.read_bytes())
            if (
                previous.get("source_id") == source.source_id
                and previous.get("source_sha256") == source.sha256
                and previous.get("extractor") == signature
                and previous.get("markdown_sha256") == _digest(md_path.read_bytes())
                and isinstance(previous.get("sections"), list)
                and previous.get("structure_sha256")
                == _digest(structure_path.read_bytes())
                and pdf_structure.valid_structure(md_path)
            ):
                record = previous
        except OSError, ValueError, AttributeError:
            pass
        if record is None:
            notify(f"Extracting {source.filename}")
            try:
                document = extract.extract_pdf_document(pdf)
                markdown = document.markdown
                if not markdown.strip():
                    raise CorpusError(f"No text extracted from {source.filename}")
                lines = markdown.splitlines()
                sections = extract.parse_sections(lines)
            except CorpusError:
                raise
            except Exception as exc:
                raise CorpusError(
                    f"Extraction failed for {source.filename}: {exc}"
                ) from exc
            record = {
                "source_id": source.source_id,
                "source_sha256": source.sha256,
                "extractor": signature,
                "markdown": md_path.name,
                "markdown_sha256": _digest(markdown.encode()),
                "lines": len(lines),
                "characters": len(markdown),
                "section_count": len(sections),
                "sections": sections,
                "structure_file": structure_path.name,
                "structure_sha256": _digest(
                    pdf_structure.structure_json(document).encode()
                ),
            }
            _write_atomic(md_path, markdown.encode())
            _write_atomic(
                structure_path, pdf_structure.structure_json(document).encode()
            )
            _write_atomic(
                sections_path,
                (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode(),
            )
        else:
            notify(f"Verified reading copy {md_path.name}")
        documents.append(
            {key: value for key, value in record.items() if key != "sections"}
            | {
                "sections_file": sections_path.name,
                "sections_sha256": _digest(sections_path.read_bytes()),
            }
        )

    result = {
        "schema_version": 1,
        "corpus_id": lock.corpus_id,
        "source_lock_sha256": _digest(lock_bytes),
        "status": "reading_copies_ready",
        "documents": documents,
    }
    _write_atomic(
        ready, (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode()
    )
    return ready
