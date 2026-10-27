"""Snapshot identity, provenance and crash recovery without models or book text."""

import json
import shutil
from pathlib import Path

import pytest
from click import unstyle
from typer.testing import CliRunner

from benchmarks import cli, passages, reading, tokenizer
from benchmarks.corpus import CorpusError


class CountingTokenizer:
    def __init__(self, revision="one"):
        self.signature = {"synthetic_tokenizer_revision": revision}
        self.calls = 0

    def count(self, texts):
        self.calls += 1
        return [max(1, len(text.split())) for text in texts]

    def verify(self):
        pass


def make_reading_inputs(tmp_path, markdown=None):
    root = tmp_path / "data"
    output = root / "reading"
    output.mkdir(parents=True)
    if markdown is None:
        markdown = "# Parent\n\n## Mechanism\n\n###### First field\n\nSome useful text.\n\n###### Second field\n\nMore useful text.\n"
    files, documents = [], []
    for name in ("first", "second"):
        source_hash = reading._digest(name.encode())
        files.append(
            {
                "source_id": name,
                "filename": f"{name}.pdf",
                "url": f"https://example.invalid/{name}.pdf",
                "sha256": source_hash,
                "bytes": 1,
                "pages": 1,
            }
        )
        lines = list[str](markdown.splitlines())
        sections = passages.extract.parse_sections(lines)
        record = {
            "source_id": name,
            "source_sha256": source_hash,
            "extractor": {"test_revision": 1},
            "markdown": f"{name}.md",
            "markdown_sha256": reading._digest(markdown.encode()),
            "lines": len(markdown.splitlines()),
            "characters": len(markdown),
            "section_count": len(sections),
            "sections": sections,
        }
        (output / f"{name}.md").write_text(markdown)
        data = passages._json(record)
        (output / f"{name}.sections.json").write_bytes(data)
        documents.append(
            {key: value for key, value in record.items() if key != "sections"}
            | {
                "sections_file": f"{name}.sections.json",
                "sections_sha256": reading._digest(data),
            }
        )
    lock = {
        "schema_version": 1,
        "corpus_id": "synthetic",
        "sources": [
            {"id": name, "version": "1", "license": "Synthetic test"}
            for name in ("first", "second")
        ],
        "files": files,
    }
    lock_bytes = passages._json(lock)
    (root / "sources.lock.json").write_bytes(lock_bytes)
    manifest = output / "manifest.json"
    manifest.write_bytes(
        passages._json(
            {
                "schema_version": 1,
                "status": "reading_copies_ready",
                "corpus_id": "synthetic",
                "source_lock_sha256": reading._digest(lock_bytes),
                "documents": documents,
            }
        )
    )
    return manifest


@pytest.fixture
def inputs(tmp_path):
    return make_reading_inputs(tmp_path)


def prepare(inputs, counter=None, **kwargs):
    return passages.prepare_passages(
        inputs,
        tokenizer=counter or CountingTokenizer(),
        notify=lambda _: None,
        **kwargs,
    )


def records(manifest):
    metadata = passages.load_snapshot(manifest)
    return [
        json.loads((manifest.parent / doc["file"]).read_bytes())
        for doc in metadata["documents"]
    ]


def test_production_boundaries_and_full_span_section(inputs):
    result = prepare(inputs)
    first, second = records(result)
    actual = first["passages"]
    expected = passages.extract.chunk_text(
        (inputs.parent / "first.md").read_text().splitlines(),
        count_tokens_batch=CountingTokenizer().count,
    )
    assert [
        {key: row[key] for key in ("start_line", "end_line", "text", "word_count")}
        for row in actual
    ] == expected
    assert len(actual) == 1
    assert actual[0]["section_path"] == "Parent > Mechanism"
    assert actual[0]["text"] == second["passages"][0]["text"]
    assert actual[0]["passage_id"] != second["passages"][0]["passage_id"]


def test_portable_order_independent_snapshot_and_model_free_read(inputs, tmp_path):
    result = prepare(inputs)
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(inputs.parent.parent, elsewhere)
    copied = elsewhere / "reading/manifest.json"
    manifest = json.loads(copied.read_bytes())
    manifest["documents"].reverse()
    copied.write_bytes(passages._json(manifest))
    shutil.rmtree(elsewhere / "passages")
    relocated = prepare(copied)
    assert relocated.read_bytes() == result.read_bytes()
    assert passages.load_snapshot(relocated)["statistics"]["passages"] == 2
    # Passages point into the reading copy, so a changed copy is rejected.
    (elsewhere / "reading/first.md").write_text("changed")
    with pytest.raises(CorpusError, match="hash mismatch"):
        passages.load_snapshot(relocated)


def test_variants_resume_without_tokenization_and_preserve_previous(inputs):
    a = prepare(inputs)
    original = a.read_bytes()
    b = prepare(inputs, target_tokens=12)
    c = prepare(inputs, CountingTokenizer("two"))
    assert len({a.parent, b.parent, c.parent}) == 3
    counter = CountingTokenizer()
    assert prepare(inputs, counter) == a
    assert counter.calls == 0
    assert a.read_bytes() == original
    assert (
        records(a)[0]["passages"][0]["passage_id"]
        == records(c)[0]["passages"][0]["passage_id"]
    )


def test_frozen_snapshot_does_not_reinterpret_headings_with_new_code(
    inputs, monkeypatch
):
    result = prepare(inputs)
    monkeypatch.setattr(
        passages.extract,
        "parse_sections",
        lambda _: pytest.fail(
            "Frozen metadata must remain readable after parser changes"
        ),
    )
    assert passages.load_snapshot(result)["statistics"]["passages"] == 2


def test_interrupted_variant_keeps_other_snapshot_and_resumes(inputs, monkeypatch):
    a = prepare(inputs)
    original = a.read_bytes()
    real_chunk = passages.extract.chunk_text
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real_chunk(*args, **kwargs)

    monkeypatch.setattr(passages.extract, "chunk_text", interrupted)
    with pytest.raises(KeyboardInterrupt):
        prepare(inputs, target_tokens=12)
    assert a.read_bytes() == original
    pending = next(path for path in a.parent.parent.iterdir() if path != a.parent)
    assert (pending / "first.json").is_file()
    assert not (pending / "manifest.json").exists()
    calls.clear()

    def resumed(*args, **kwargs):
        calls.append(1)
        return real_chunk(*args, **kwargs)

    monkeypatch.setattr(passages.extract, "chunk_text", resumed)
    prepare(inputs, target_tokens=12)
    assert len(calls) == 1


@pytest.mark.parametrize("damage", ["markdown", "sections", "membership"])
def test_bad_reading_inputs_rejected_before_tokenization(inputs, damage):
    if damage == "markdown":
        (inputs.parent / "first.md").write_text("damaged")
    elif damage == "sections":
        (inputs.parent / "first.sections.json").write_text("{}")
    else:
        manifest = json.loads(inputs.read_bytes())
        manifest["documents"] = manifest["documents"][:1]
        inputs.write_bytes(passages._json(manifest))
    counter = CountingTokenizer()
    with pytest.raises(CorpusError, match="Invalid reading inputs"):
        prepare(inputs, counter)
    assert counter.calls == 0
    assert not (inputs.parent.parent / "passages").exists()


def test_complete_damaged_snapshot_is_not_silently_overwritten(inputs):
    result = prepare(inputs)
    bad = result.parent / "first.json"
    bad.write_text("broken")
    with pytest.raises(CorpusError, match="hash mismatch"):
        prepare(inputs)
    assert bad.read_text() == "broken"


def test_changed_provenance_changes_passage_id():
    record = {
        "source_id": "first",
        "source_sha256": "source",
        "markdown_sha256": "md",
        "start_line": 1,
        "end_line": 2,
        "section_path": "Parent",
        "text_sha256": "text",
    }
    original = passages._passage_id(record)
    for key in record:
        assert passages._passage_id(record | {key: "changed"}) != original


def test_table_continuations_repeat_the_header_in_the_snapshot(tmp_path):
    markdown = "# Registers\n\n|Register|Description|\n|---|---|\n" + "".join(
        f"|r{i}|Description of register number {i}|\n" for i in range(12)
    )
    inputs = make_reading_inputs(tmp_path, markdown)
    snapshot = prepare(inputs, target_tokens=20)
    continuations = [
        p for p in records(snapshot)[0]["passages"] if "source_ranges" in p
    ]
    assert len(continuations) > 1
    last = continuations[-1]
    assert "Register|Description" in last["text"]
    assert "r11" in last["text"]
    assert "r0|" not in last["text"]
    assert last["context_scope"] == 1


def test_tokenizer_change_before_checkpoint_does_not_publish(inputs):
    class Changing(CountingTokenizer):
        def verify(self):
            if self.calls:
                raise CorpusError("Tokenizer changed")

    with pytest.raises(CorpusError, match="Tokenizer changed"):
        prepare(inputs, Changing())
    assert not list((inputs.parent.parent / "passages").glob("*/*.json"))


def test_invalid_final_token_counts_do_not_publish(inputs):
    class Invalid(CountingTokenizer):
        def count(self, texts):
            return [True] * len(texts)

    with pytest.raises(CorpusError, match="invalid passage"):
        prepare(inputs, Invalid())
    assert not list((inputs.parent.parent / "passages").glob("*/*.json"))


def test_prepare_cli_runs_stages_automatically(tmp_path, monkeypatch):
    calls = []

    def stage(name):
        def run(*args, **kwargs):
            calls.append(name)
            return tmp_path / name / "manifest.json"

        return run

    for name in ("sources", "reading", "passages", "index", "pool"):
        monkeypatch.setattr(cli, f"prepare_{name}", stage(name))

    def frozen(root):
        from benchmarks.storage import write

        path = root / "suite-test.json"
        write(path, {"status": "ready"})
        return path

    monkeypatch.setattr(cli, "freeze", frozen)
    monkeypatch.setattr(cli, "build_report", lambda _: {})
    monkeypatch.setattr(cli, "DEFAULT_DATA", tmp_path)
    result = CliRunner().invoke(cli.app, ["prepare"])
    assert result.exit_code == 0, result.output
    assert calls == ["sources", "reading", "passages", "index", "pool"]
    assert "Queries: 207 (187 en, 10 ru, 10 zh)" in result.output
    help_text = unstyle(CliRunner().invoke(cli.app, ["prepare", "--help"]).output)
    assert "--extract" not in help_text
    assert "--chunk" not in help_text


@pytest.mark.parametrize(
    "change", [None, "chunker", "markdown", "context_prompt", "index_missing"]
)
def test_prepare_reuses_frozen_boundaries_but_applies_code_changes(
    tmp_path, monkeypatch, change
):
    from benchmarks.storage import read, write

    root = tmp_path
    paths = {
        name: root / name / "manifest.json"
        for name in (
            "sources",
            "reading",
            "old_passages",
            "new_passages",
            "old_index",
            "new_index",
        )
    }
    state_path = root / "preparation.json"
    write(
        state_path,
        {
            "passages": "old_passages/manifest.json",
            "index": "old_index/manifest.json",
        },
    )
    # The index is local only: a fresh clone has none and must rebuild it.
    if change != "index_missing":
        paths["old_index"].parent.mkdir()
        paths["old_index"].write_text("{}")
    pipeline = {
        "code_sha256": reading._digest(Path(cli.extract.__file__).read_bytes()),
        "adapter_sha256": reading._digest(
            Path(cli.passage_module.__file__).read_bytes()
        ),
    }
    if change == "chunker":
        pipeline["code_sha256"] = "old-extract-code"
    documents = [{"source_id": "first", "markdown_sha256": "md"}]
    monkeypatch.setattr(
        cli,
        "load_snapshot",
        lambda _: {"inputs": {"documents": documents, "pipeline": pipeline}},
    )
    reading_documents = (
        documents
        if change != "markdown"
        else [documents[0] | {"markdown_sha256": "new"}]
    )
    monkeypatch.setattr(
        cli.passage_module,
        "_load_inputs",
        lambda _: ({"documents": reading_documents}, {}),
    )
    monkeypatch.setattr(cli, "context_bodies_digest", lambda _: "current-requests")
    monkeypatch.setattr(
        cli,
        "load_index",
        lambda _: {
            "context_bodies_sha256": "old-prompt"
            if change == "context_prompt"
            else "current-requests"
        },
    )
    for name in ("sources", "reading"):
        monkeypatch.setattr(
            cli, f"prepare_{name}", lambda *a, _name=name, **k: paths[_name]
        )
    calls = []

    def new_passages(*a, **k):
        calls.append("passages")
        return paths["new_passages"]

    def new_index(*a, **k):
        # A failed restart must not leave a stale index selected for a later run.
        assert read(state_path)["index"] is None
        calls.append("index")
        return paths["new_index"]

    monkeypatch.setattr(cli, "prepare_passages", new_passages)
    monkeypatch.setattr(cli, "prepare_index", new_index)
    monkeypatch.setattr(cli, "prepare_pool", lambda *a, **k: None)
    suite = root / "suite.json"
    write(suite, {"status": "ready"})
    monkeypatch.setattr(cli, "freeze", lambda _: suite)
    monkeypatch.setattr(cli, "build_report", lambda _: {})
    cli._prepare_workflow(root)
    rebuilt = change in ("chunker", "markdown")
    assert calls == (["passages", "index"] if rebuilt else ["index"] if change else [])
    state = read(state_path)
    assert state["passages"] == (
        "new_passages/manifest.json" if rebuilt else "old_passages/manifest.json"
    )


def test_tokenizer_fingerprints_weights_and_engine_without_embeddings(
    tmp_path, monkeypatch
):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"synthetic weights")
    props = tmp_path / "props.json"
    props.write_text(json.dumps({"model_path": str(model), "build_info": "test-build"}))
    monkeypatch.setattr(tokenizer.llama_client, "_ensure_server", lambda _: True)
    monkeypatch.setattr(
        tokenizer.urllib.request, "urlopen", lambda *a, **k: props.open("rb")
    )
    monkeypatch.setattr(
        tokenizer.llama_client, "get_token_counts_batch", lambda texts: [2] * len(texts)
    )
    counter = tokenizer.ProductionTokenizer()
    assert counter.signature["model_sha256"] == [reading._digest(model.read_bytes())]
    assert str(tmp_path) not in json.dumps(counter.signature)
    assert counter.count(["hello"]) == [2]
    counter.verify()
    model.write_bytes(b"different weights")
    with pytest.raises(CorpusError, match="Tokenizer changed"):
        counter.verify()


def test_tokenizer_setup_failure_is_actionable(monkeypatch):
    monkeypatch.setattr(tokenizer.llama_client, "_ensure_server", lambda _: False)
    with pytest.raises(
        CorpusError, match="model setup.*Prepared PDFs and Markdown are kept"
    ):
        tokenizer.ProductionTokenizer()
