"""Synthetic end-to-end CLI run without production data or model services."""

import json
import sys

from sova import cli, config, projects


def test_index_resume_and_doctor_end_to_end(monkeypatch, tmp_path, capsys):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "manual.md").write_text(
        "# Account access\n\n"
        "Administrators grant account access after identity verification and "
        "record the approval in the audit ledger for future reviews.\n",
        encoding="utf-8",
    )

    project_root = tmp_path / "projects"
    monkeypatch.setattr(projects, "_PROJECTS_ROOT", project_root)
    monkeypatch.setattr(projects, "_REGISTRY_PATH", project_root / "registry.json")
    monkeypatch.setattr(cli, "check_servers", lambda **_kwargs: (True, "Model ready"))
    monkeypatch.setattr(cli, "stop_server", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "run_embedding_canary", lambda **_kwargs: None)
    monkeypatch.setattr(cli, "quantize_vectors", lambda _conn: None)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    context_calls: list[str] = []

    def generate_context(_doc, _section, text, _previous, _next):
        context_calls.append(text)
        return (
            "Account access requires verified identity, recorded approval, "
            "and the retrieval-only provenance sentinel."
        )

    monkeypatch.setattr(cli, "generate_context", generate_context)

    def embed(texts, on_batch=None):
        vectors = [[0.01] * config.EMBEDDING_DIM for _ in texts]
        if on_batch:
            on_batch(
                list(range(len(texts))),
                vectors,
                {"batch_size": len(texts), "workers": 1, "duration_s": 0.001},
            )
        return vectors

    monkeypatch.setattr(cli, "get_embeddings_batch", embed)
    monkeypatch.setattr(
        cli,
        "get_token_counts_batch",
        lambda texts: [len(text.split()) for text in texts],
    )

    monkeypatch.setattr(sys, "argv", ["sova", "--json", "index", str(docs)])
    cli.main()
    first_events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line
    ]

    assert first_events[0]["type"] == "run_started"
    assert first_events[-1]["type"] == "completed"
    assert len(context_calls) == 1

    project = projects.get_project("documents")
    assert project is not None
    conn = cli.sqlite3.connect(project.db_path)
    row = conn.execute(
        """
        SELECT cc.pipeline_signature, c.embedding_signature, c.section_path,
               c.text, c.search_text
        FROM chunk_contexts cc JOIN chunks c ON c.id = cc.chunk_id
        """
    ).fetchone()
    assert row[0:3] == (
        cli._context_pipeline_signature(),
        cli._embedding_pipeline_signature(),
        "Account access",
    )
    assert "retrieval-only provenance sentinel" not in row[3]
    assert "retrieval-only provenance sentinel" in row[4]
    fts_hit = conn.execute(
        """
        SELECT rowid FROM chunks_fts
        WHERE chunks_fts MATCH '"retrieval" "provenance" "sentinel"'
        """
    ).fetchone()
    assert fts_hit is not None
    conn.close()

    monkeypatch.setattr(sys, "argv", ["sova", "--json", "index", "documents"])
    cli.main()
    second_events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line
    ]

    assert second_events[-1]["type"] == "completed"
    assert len(context_calls) == 1

    (docs / "manual.md").write_text(
        "# Account access\n\n"
        "Security administrators now require two reviewers, verified identity, "
        "and an immutable approval record before granting any account access.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["sova", "--json", "index", "documents"])
    cli.main()
    changed_events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line
    ]

    assert changed_events[-1]["type"] == "completed"
    assert len(context_calls) == 2

    monkeypatch.setattr(sys, "argv", ["sova", "--json", "doctor", "documents"])
    cli.main()
    audit_events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line
    ]

    assert audit_events == [
        {
            "data": {"findings": 0},
            "level": "info",
            "message": "Database checks passed",
            "type": "audit_completed",
        }
    ]
