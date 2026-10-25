"""Reading-copy resume and invalidation without book downloads or model calls."""

import hashlib
import json

import pymupdf
import pytest

from benchmarks import corpus, reading
from sova.pdf_structure import ExtractedDocument


def document(markdown):
    return ExtractedDocument(
        markdown,
        {
            "schema_version": 1,
            "markdown_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
            "blocks": [],
            "toc": [],
            "diagnostics": [],
        },
    )


@pytest.fixture
def prepared_sources(tmp_path):
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), "Synthetic document")
        pdf = document.tobytes()
    sources_dir = tmp_path / "data" / "sources"
    sources_dir.mkdir(parents=True)
    files = []
    for name in ("first", "second"):
        (sources_dir / f"{name}.pdf").write_bytes(pdf)
        files.append(
            {
                "source_id": name,
                "filename": f"{name}.pdf",
                "url": f"https://example.invalid/{name}.pdf",
                "sha256": hashlib.sha256(pdf).hexdigest(),
                "bytes": len(pdf),
                "pages": 1,
            }
        )
    lock = {
        "schema_version": 1,
        "corpus_id": "synthetic",
        "sources": [
            {"id": name, "version": "1", "license": "Test"}
            for name in ("first", "second")
        ],
        "files": files,
    }
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock))
    corpus.prepare_sources(
        tmp_path / "data", lock_path=lock_path, notify=lambda _: None
    )
    return tmp_path / "data"


def test_complete_reading_resume_and_changed_extractor(prepared_sources, monkeypatch):
    calls = []

    def extract(path):
        calls.append(path.name)
        return document("# Parent\n\n## Child\n\nSynthetic text.\n")

    monkeypatch.setattr(reading.extract, "extract_pdf_document", extract)
    result = reading.prepare_reading(prepared_sources, notify=lambda _: None)
    assert calls == ["first.pdf", "second.pdf"]
    record = json.loads((result.parent / "first.sections.json").read_bytes())
    assert record["sections"][1]["path"] == "Parent > Child"
    before = result.read_bytes()
    reading.prepare_reading(prepared_sources, notify=lambda _: None)
    assert calls == ["first.pdf", "second.pdf"]
    assert result.read_bytes() == before
    # A damaged Markdown copy is rebuilt; the other document is reused.
    (result.parent / "first.md").write_text("broken")
    reading.prepare_reading(prepared_sources, notify=lambda _: None)
    assert calls == ["first.pdf", "second.pdf", "first.pdf"]
    monkeypatch.setattr(
        reading, "_extractor_signature", lambda: {"code_sha256": "changed"}
    )
    reading.prepare_reading(prepared_sources, notify=lambda _: None)
    assert calls[-2:] == ["first.pdf", "second.pdf"]


def test_interrupted_reading_resumes_completed_documents(prepared_sources, monkeypatch):
    def extract(path):
        if path.name == "second.pdf":
            raise KeyboardInterrupt
        return document("# First\n\nSynthetic text.\n")

    monkeypatch.setattr(reading.extract, "extract_pdf_document", extract)
    with pytest.raises(KeyboardInterrupt):
        reading.prepare_reading(prepared_sources, notify=lambda _: None)
    output = prepared_sources / "reading"
    assert (output / "first.sections.json").exists()
    assert not (output / "manifest.json").exists()
    calls = []

    def resume(path):
        calls.append(path.name)
        return document("# Second\n\nSynthetic text.\n")

    monkeypatch.setattr(reading.extract, "extract_pdf_document", resume)
    reading.prepare_reading(prepared_sources, notify=lambda _: None)
    assert calls == ["second.pdf"]


def test_reading_rechecks_source_bytes(prepared_sources, monkeypatch):
    monkeypatch.setattr(
        reading.extract,
        "extract_pdf_document",
        lambda _: pytest.fail("Must verify PDF first"),
    )
    (prepared_sources / "sources/first.pdf").write_bytes(b"bad PDF")
    with pytest.raises(corpus.CorpusError, match="byte size"):
        reading.prepare_reading(prepared_sources)
    assert not (prepared_sources / "reading/manifest.json").exists()
