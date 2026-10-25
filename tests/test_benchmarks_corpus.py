"""Corpus preparation uses synthetic PDFs; ordinary tests never download books."""

import hashlib
import http.client
import io
import json

import pymupdf
import pytest
from typer.testing import CliRunner

from benchmarks import cli, corpus


@pytest.fixture
def fixture_corpus(tmp_path):
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), "Synthetic source")
        document.new_page().insert_text((72, 72), "Memory translation")
        document.set_toc([[1, "Memory", 1], [2, "Translation", 2]])
        pdf = document.tobytes()
    lock = {
        "schema_version": 1,
        "corpus_id": "test-corpus",
        "sources": [{"id": "test", "version": "1", "license": "Test only"}],
        "files": [
            {
                "source_id": "test",
                "filename": "test.pdf",
                "url": "https://example.invalid/test.pdf",
                "sha256": hashlib.sha256(pdf).hexdigest(),
                "bytes": len(pdf),
                "pages": 2,
            }
        ],
    }
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock))
    return pdf, lock, lock_path, tmp_path / "data"


def prepare(fixture):
    _, _, lock_path, data_dir = fixture
    corpus.prepare_sources(data_dir, lock_path=lock_path, notify=lambda _: None)


def no_network(*args, **kwargs):
    pytest.fail("Unexpected network access")


def test_download_verify_and_automatic_reuse(fixture_corpus, monkeypatch):
    pdf, _, lock_path, root = fixture_corpus
    calls = []

    def download(request, timeout):
        calls.append(request.full_url)
        assert timeout == 30
        return io.BytesIO(pdf)

    monkeypatch.setattr(corpus.urllib.request, "urlopen", download)
    prepare(fixture_corpus)
    assert calls == ["https://example.invalid/test.pdf"]
    assert (root / "sources.lock.json").read_bytes() == lock_path.read_bytes()
    destination = root / "sources/test.pdf"
    assert destination.read_bytes() == pdf
    modified = destination.stat().st_mtime_ns
    monkeypatch.setattr(corpus.urllib.request, "urlopen", no_network)
    prepare(fixture_corpus)
    assert destination.stat().st_mtime_ns == modified


def test_existing_sources_are_reused_without_network(fixture_corpus, monkeypatch):
    pdf, _, _, root = fixture_corpus
    sources_dir = root / "sources"
    sources_dir.mkdir(parents=True)
    (sources_dir / "test.pdf").write_bytes(pdf)
    monkeypatch.setattr(corpus.urllib.request, "urlopen", no_network)
    prepare(fixture_corpus)
    assert (root / "sources/test.pdf").read_bytes() == pdf


@pytest.mark.parametrize(
    "problem", ["hash", "truncated", "oversize", "not_pdf", "page_count"]
)
def test_reject_invalid_downloads(fixture_corpus, monkeypatch, problem):
    pdf, lock, lock_path, root = fixture_corpus
    if problem == "hash":
        pdf = pdf[:-1] + bytes([pdf[-1] ^ 1])
    elif problem == "truncated":
        pdf = pdf[:100]
    elif problem == "oversize":
        pdf += b"extra"
    elif problem == "not_pdf":
        pdf = b"<html>upstream error</html>"
        lock["files"][0].update(bytes=len(pdf), sha256=hashlib.sha256(pdf).hexdigest())
    elif problem == "page_count":
        lock["files"][0]["pages"] = 3
    lock_path.write_text(json.dumps(lock))
    monkeypatch.setattr(
        corpus.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(pdf)
    )
    with pytest.raises(corpus.CorpusError):
        prepare(fixture_corpus)
    assert not (root / "sources/test.pdf").exists()
    assert not list((root / "sources").iterdir())


def test_corrupt_cache_is_reported(fixture_corpus, monkeypatch):
    pdf, _, _, root = fixture_corpus
    monkeypatch.setattr(
        corpus.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(pdf)
    )
    prepare(fixture_corpus)
    (root / "sources/test.pdf").write_bytes(b"broken")
    monkeypatch.setattr(corpus.urllib.request, "urlopen", no_network)
    with pytest.raises(corpus.CorpusError, match="byte size"):
        prepare(fixture_corpus)


def test_network_failure_leaves_no_partial_file(fixture_corpus, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(corpus.urllib.request, "urlopen", unavailable)
    with pytest.raises(corpus.CorpusError, match="network unavailable"):
        prepare(fixture_corpus)
    root = fixture_corpus[3]
    assert not list((root / "sources").iterdir())


def test_dropped_http_response_is_reported_and_cleaned(fixture_corpus, monkeypatch):
    class BrokenResponse(io.BytesIO):
        def read(self, size=-1):
            raise http.client.IncompleteRead(b"partial", 100)

    monkeypatch.setattr(
        corpus.urllib.request, "urlopen", lambda *a, **k: BrokenResponse()
    )
    with pytest.raises(corpus.CorpusError, match="test.pdf"):
        prepare(fixture_corpus)
    root = fixture_corpus[3]
    assert not list((root / "sources").iterdir())


def test_interruption_preserves_finished_files_and_resumes(fixture_corpus, monkeypatch):
    pdf, lock, lock_path, root = fixture_corpus
    lock["sources"].append({"id": "second", "version": "1", "license": "Test only"})
    lock["files"].append(
        {
            **lock["files"][0],
            "source_id": "second",
            "filename": "second.pdf",
            "url": "https://example.invalid/second.pdf",
        }
    )
    lock_path.write_text(json.dumps(lock))

    class Interrupted(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise KeyboardInterrupt
            return super().read(100)

    def download(request, **kwargs):
        return (
            Interrupted(pdf)
            if request.full_url.endswith("second.pdf")
            else io.BytesIO(pdf)
        )

    monkeypatch.setattr(corpus.urllib.request, "urlopen", download)
    with pytest.raises(KeyboardInterrupt):
        prepare(fixture_corpus)
    assert (root / "sources/test.pdf").read_bytes() == pdf
    assert not (root / "sources/second.pdf").exists()
    assert not list((root / "sources").glob("*.part"))
    calls = []

    def retry(request, **kwargs):
        calls.append(request.full_url)
        return io.BytesIO(pdf)

    monkeypatch.setattr(corpus.urllib.request, "urlopen", retry)
    prepare(fixture_corpus)
    assert calls == ["https://example.invalid/second.pdf"]


@pytest.mark.parametrize("problem", ["path", "duplicate", "source", "version"])
def test_invalid_lock_is_rejected_before_writing(fixture_corpus, problem):
    _, lock, lock_path, root = fixture_corpus
    if problem == "path":
        lock["files"][0]["filename"] = "../outside.pdf"
    elif problem == "duplicate":
        lock["files"].append(lock["files"][0])
    elif problem == "source":
        lock["files"][0]["source_id"] = "unknown"
    elif problem == "version":
        lock["schema_version"] = 99
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(corpus.CorpusError, match="Invalid source lock"):
        prepare(fixture_corpus)
    assert not root.exists()


def test_cli_reports_failure_without_traceback(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(corpus.urllib.request, "urlopen", unavailable)
    monkeypatch.setattr(cli, "DEFAULT_DATA", tmp_path / "empty")
    result = CliRunner().invoke(cli.app, ["prepare"])
    assert result.exit_code == 1
    assert "network unavailable" in result.output
    assert "Traceback" not in result.output


def test_pinned_corpus_totals():
    lock = corpus.SourceLock.model_validate_json(corpus.DEFAULT_LOCK.read_bytes())
    assert len(lock.files) == 5
    assert sum(file.pages for file in lock.files) == 1835
    assert sum(file.bytes for file in lock.files) == 24420633
