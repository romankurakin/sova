"""Tests for cli module."""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

from sova.cli import fmt_duration, fmt_size


class TestFmtSize:
    def test_zero_bytes(self):
        assert fmt_size(0) == "-"

    def test_zero_bytes_repeated(self):
        assert fmt_size(0) == "-"

    def test_bytes(self):
        assert fmt_size(500) == "500 B"

    def test_kilobytes(self):
        result = fmt_size(2048)
        assert "KB" in result
        assert "2.0" in result

    def test_megabytes(self):
        result = fmt_size(2 * 1024 * 1024)
        assert "MB" in result
        assert "2.0" in result

    def test_just_over_kb_boundary(self):
        result = fmt_size(1024)
        assert "KB" in result

    def test_just_over_mb_boundary(self):
        result = fmt_size(1024 * 1024)
        assert "MB" in result


class TestFmtDuration:
    def test_seconds(self):
        result = fmt_duration(30.0)
        assert "s" in result
        assert "30.0" in result

    def test_minutes(self):
        result = fmt_duration(120.0)
        assert "m" in result
        assert "2.0" in result

    def test_hours(self):
        result = fmt_duration(7200.0)
        assert "h" in result
        assert "2.0" in result

    def test_just_under_minute(self):
        result = fmt_duration(59.9)
        assert "s" in result

    def test_just_at_minute(self):
        result = fmt_duration(60.0)
        assert "m" in result

    def test_just_at_hour(self):
        result = fmt_duration(3600.0)
        assert "h" in result


class TestInterruptHandling:
    def test_search_ctrl_c_during_server_check_does_not_stop_services(
        self, monkeypatch
    ):
        from sova import cli

        stops: list[tuple[str, bool]] = []

        class DummyProject:
            project_id = "proj"

        monkeypatch.setattr(sys, "argv", ["sova", "proj", "q"])
        monkeypatch.setattr(
            cli,
            "_activate_project_from_ref",
            lambda _ref, allow_create_from_dir=False: DummyProject(),
        )
        monkeypatch.setattr(
            cli,
            "check_servers",
            lambda **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        monkeypatch.setattr(
            cli,
            "stop_server",
            lambda url, suppress_interrupt=False: stops.append(
                (url, suppress_interrupt)
            ),
        )

        with pytest.raises(SystemExit) as exc:
            cli.main()

        assert exc.value.code == 130
        assert stops == []

    def test_index_ctrl_c_returns_130_and_stops_services(self, monkeypatch):
        from sova import cli

        class DummyConn:
            def close(self):
                return None

        class DummyCache:
            def clear(self):
                return None

        stops: list[tuple[str, bool]] = []

        class DummyProject:
            project_id = "proj"

        monkeypatch.setattr(sys, "argv", ["sova", "index", "proj"])
        monkeypatch.setattr(
            cli,
            "_activate_project_from_ref",
            lambda _ref, allow_create_from_dir=False: DummyProject(),
        )
        monkeypatch.setattr(cli.config, "get_docs_dir", lambda: Path("/tmp"))
        monkeypatch.setattr(cli, "check_servers", lambda **kwargs: (True, "ready"))
        monkeypatch.setattr(cli, "_report_phase_runtime", lambda *args, **kwargs: None)
        monkeypatch.setattr(cli, "init_db", lambda: DummyConn())
        monkeypatch.setattr(
            cli,
            "_sync_index_signatures",
            lambda conn: cli._IndexSignatureState(False, False, "c", "e", "k"),
        )
        monkeypatch.setattr(
            cli,
            "find_docs",
            lambda: [{"name": "doc1", "pdf": None, "md": Path("/tmp/doc1.md")}],
        )
        monkeypatch.setattr(
            cli,
            "_prepare_source",
            lambda name, *_args, **_kwargs: cli._PreparedSource(
                name, Path(f"/tmp/{name}.md"), "sig"
            ),
        )
        monkeypatch.setattr(
            cli,
            "_tokenize_doc",
            lambda *_args, **_kwargs: (1, [{"start_line": 1, "text": "x"}], []),
        )
        monkeypatch.setattr(cli, "_current_tokenized_doc_id", lambda *_args: None)
        monkeypatch.setattr(cli, "_context_work_pending", lambda *_args, **_kwargs: True)
        monkeypatch.setattr(
            cli, "_embedding_work_pending", lambda *_args, **_kwargs: True
        )
        monkeypatch.setattr(
            cli,
            "_generate_contexts",
            lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        monkeypatch.setattr(
            cli,
            "_load_prepared_doc",
            lambda *_args: ([{"start_line": 1, "text": "x"}], []),
        )
        monkeypatch.setattr(cli, "get_cache", lambda: DummyCache())
        monkeypatch.setattr(cli, "_prune_missing_documents", lambda *args: 0)
        monkeypatch.setattr(
            cli,
            "stop_server",
            lambda url, suppress_interrupt=False: stops.append(
                (url, suppress_interrupt)
            ),
        )

        with pytest.raises(SystemExit) as exc:
            cli.main()

        assert exc.value.code == 130
        assert (
            cli.config.CONTEXT_SERVER_URL,
            True,
        ) in stops
        assert (
            cli.config.EMBEDDING_SERVER_URL,
            True,
        ) in stops

    def test_search_reports_server_status_changes(self, monkeypatch):
        from sova import cli

        statuses: list[str] = []

        class DummyProject:
            project_id = "proj"

        def fake_check_servers(on_status=None, mode="search", **kwargs):
            if kwargs.get("fast_only"):
                return False, "warm check failed"
            if on_status:
                on_status("embedding: downloading (0.5 GB)")
                on_status("embedding: loading")
            return True, "ready"

        monkeypatch.setattr(sys, "argv", ["sova", "proj", "q"])
        monkeypatch.setattr(
            cli,
            "_activate_project_from_ref",
            lambda _ref, allow_create_from_dir=False: DummyProject(),
        )
        monkeypatch.setattr(cli, "check_servers", fake_check_servers)
        monkeypatch.setattr(cli, "search_semantic", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            cli, "status", lambda message, phase=None: statuses.append(message)
        )

        cli.main()

        assert statuses.count("embedding downloading (0.5 GB)") == 1
        assert statuses.count("embedding loading") == 1
        assert "ready" in statuses

    def test_search_uses_single_embedding_service_path(self, monkeypatch):
        from sova import cli

        check_calls: list[dict[str, object]] = []
        search_calls: list[tuple[str, int]] = []

        class DummyProject:
            project_id = "proj"

        def fake_check_servers(**kwargs):
            check_calls.append(kwargs)
            return True, "ready"

        def fake_search_semantic(query: str, limit: int, verbose: bool = False):
            del verbose
            search_calls.append((query, limit))

        monkeypatch.setattr(sys, "argv", ["sova", "proj", "q"])
        monkeypatch.setattr(
            cli,
            "_activate_project_from_ref",
            lambda _ref, allow_create_from_dir=False: DummyProject(),
        )
        monkeypatch.setattr(cli, "check_servers", fake_check_servers)
        monkeypatch.setattr(cli, "search_semantic", fake_search_semantic)

        cli.main()

        assert check_calls[0] == {"mode": "search", "fast_only": True}
        assert search_calls == [("q", 10)]


def test_json_missing_arguments_are_one_structured_error(monkeypatch, capsys):
    from sova import cli

    monkeypatch.setattr(sys, "argv", ["sova", "--json", "search"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2
    captured = capsys.readouterr()
    events = [json.loads(line) for line in captured.out.splitlines()]
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert "Missing argument" in events[0]["data"]["cause"]
    assert captured.err == ""


def test_json_help_is_one_structured_event(monkeypatch, capsys):
    from sova import cli

    monkeypatch.setattr(sys, "argv", ["sova", "--json", "--help"])

    cli.main()

    captured = capsys.readouterr()
    events = [json.loads(line) for line in captured.out.splitlines()]
    assert len(events) == 1
    assert events[0]["type"] == "help"
    assert "Commands:" in events[0]["data"]["text"]
    assert captured.err == ""


def test_main_propagates_command_exit_code(monkeypatch):
    from sova import cli

    class DummyCommand:
        def main(self, **_kwargs):
            return 7

    monkeypatch.setattr(sys, "argv", ["sova", "projects"])
    monkeypatch.setattr(cli.typer.main, "get_command", lambda _app: DummyCommand())

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 7


def test_index_reserved_token_fails_before_project_lookup(monkeypatch):
    from sova import cli

    captured: dict[str, str] = {}

    monkeypatch.setattr(sys, "argv", ["sova", "index", "list"])
    monkeypatch.setattr(
        cli,
        "_report_error_block",
        lambda summary, **kw: captured.update(
            {
                "summary": summary,
                "cause": kw.get("cause", ""),
                "action": kw.get("action", ""),
            }
        ),
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2
    assert captured["summary"] == "project name is reserved"
    assert "conflicts with a CLI command" in captured["cause"]


def test_help_command_prints_global_help(monkeypatch, capsys):
    from sova import cli

    monkeypatch.setattr(sys, "argv", ["sova", "help"])
    cli.main()

    out = capsys.readouterr().out
    assert "Usage: sova" in out
    for command in ("help", "projects", "download", "remove", "list", "index"):
        assert command in out
    assert "Download all model files" in out
    assert "search" in out
    assert "--json" in out


def test_help_flag_prints_global_help(monkeypatch, capsys):
    from sova import cli

    monkeypatch.setattr(sys, "argv", ["sova", "--help"])
    cli.main()

    out = capsys.readouterr().out
    assert "Usage: sova" in out
    assert "Download all model files" in out


def test_subcommand_help_flag_prints_command_help(monkeypatch, capsys):
    from sova import cli

    monkeypatch.setattr(sys, "argv", ["sova", "projects", "--help"])
    cli.main()

    out = capsys.readouterr().out
    assert "Usage: sova projects" in out


class TestRuntimeReporting:
    def test_report_phase_runtime_uses_named_budget_fields(self, monkeypatch):
        from sova import cli

        headroom: list[float | None] = []

        monkeypatch.setattr(cli.config, "get_effective_available_gib", lambda: 7.4)
        monkeypatch.setattr(cli.config, "get_memory_reserve_gib", lambda _mode: 4.0)
        monkeypatch.setattr(
            cli,
            "report_runtime",
            lambda memory_headroom_gib: headroom.append(memory_headroom_gib),
        )

        cli._report_phase_runtime("index.context", "chat", mode="index")

        assert headroom == [3.4]

    def test_runtime_reporter_re_emits_after_refresh_interval(self, monkeypatch):
        from sova import cli

        calls: list[tuple[str, str, str]] = []
        times = iter([0.0, 5.0, 21.0, 22.0, 43.0])

        monkeypatch.setattr(cli.time, "monotonic", lambda: next(times))
        monkeypatch.setattr(
            cli,
            "_report_phase_runtime",
            lambda phase, service_name, mode="index": calls.append(
                (phase, service_name, mode)
            ),
        )

        tick = cli._make_runtime_reporter("index.context", "chat", mode="index")
        tick(True)
        tick(False)
        tick(False)
        tick(False)
        tick(False)

        assert calls == [
            ("index.context", "chat", "index"),
            ("index.context", "chat", "index"),
            ("index.context", "chat", "index"),
        ]


class TestProgressReporter:
    def test_progress_reporter_throttles_and_always_emits_final(self, monkeypatch):
        from sova import cli

        updates: list[tuple[str, int, int]] = []
        monkeypatch.setattr(
            cli,
            "progress",
            lambda phase, done, total, **kwargs: updates.append((phase, done, total)),
        )
        monkeypatch.setattr(cli.time, "monotonic", lambda: 100.0)

        emit = cli._make_progress_reporter("context", 1000)
        for done in range(1, 1001):
            emit(done)

        # The event stream is lossless; renderers own throttling.
        assert len(updates) == 1000
        assert updates[-1] == ("context", 1000, 1000)


def test_generate_contexts_is_idempotent_on_duplicate_chunk_start_lines(monkeypatch):
    from sova import cli

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL,
            start_line INTEGER NOT NULL,
            embedding BLOB
        );
        CREATE TABLE chunk_contexts (
            chunk_id INTEGER PRIMARY KEY,
            context TEXT NOT NULL,
            model TEXT NOT NULL
        );
        """
    )
    conn.execute("INSERT INTO chunks (id, doc_id, start_line) VALUES (1, 1, 10)")
    conn.commit()

    chunks = [
        {"start_line": 10, "text": "first"},
        {"start_line": 10, "text": "first duplicate"},
    ]
    sections: list[dict] = []

    monkeypatch.setattr(cli, "generate_context", lambda *args, **kwargs: "ctx")

    cli._generate_contexts("doc", 1, chunks, sections, conn)
    count = conn.execute("SELECT COUNT(*) FROM chunk_contexts").fetchone()[0]
    assert count == 1

    # Retry should not fail and should keep a single row.
    cli._generate_contexts("doc", 1, chunks, sections, conn)
    count_after = conn.execute("SELECT COUNT(*) FROM chunk_contexts").fetchone()[0]
    assert count_after == 1

    conn.close()


def test_tokenize_doc_updates_changed_chunk_text_and_clears_context_and_embedding(
    monkeypatch, tmp_path
):
    from sova import cli

    monkeypatch.setattr(
        cli,
        "get_token_counts_batch",
        lambda texts: [len(text.split()) for text in texts],
    )

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            path TEXT NOT NULL,
            line_count INTEGER,
            expected_chunks INTEGER
        );
        CREATE TABLE sections (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            level INTEGER NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL,
            section_id INTEGER,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            word_count INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB,
            embedding_signature TEXT,
            section_path TEXT NOT NULL DEFAULT '',
            search_text TEXT NOT NULL DEFAULT '',
            is_index INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE chunk_contexts (
            chunk_id INTEGER PRIMARY KEY,
            context TEXT NOT NULL,
            model TEXT NOT NULL
        );
        """
    )

    md = tmp_path / "doc.md"
    md.write_text("\n".join(["alpha"] * 12) + "\n", encoding="utf-8")
    prepared = cli._tokenize_doc(cli._PreparedSource("doc", md, "sig-1"), conn)
    assert prepared is not None
    doc_id, chunks, _ = prepared
    assert doc_id == 1
    assert len(chunks) == 1
    loaded_chunks, loaded_sections = cli._load_prepared_doc(conn, doc_id)
    assert loaded_chunks == chunks
    assert loaded_sections == []
    chunk_id = conn.execute("SELECT id FROM chunks WHERE doc_id = 1").fetchone()[0]

    conn.execute(
        "INSERT INTO chunk_contexts (chunk_id, context, model) VALUES (?, ?, ?)",
        (chunk_id, "old context", "model-v1"),
    )
    conn.execute("UPDATE chunks SET embedding = ? WHERE id = ?", (b"old-emb", chunk_id))
    conn.commit()

    md.write_text("\n".join(["changed"] * 12) + "\n", encoding="utf-8")
    cli._tokenize_doc(cli._PreparedSource("doc", md, "sig-2"), conn)

    row = conn.execute(
        "SELECT text, embedding FROM chunks WHERE id = ?", (chunk_id,)
    ).fetchone()
    assert row[0].startswith("changed")
    assert row[1] is None
    ctx_count = conn.execute(
        "SELECT COUNT(*) FROM chunk_contexts WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()[0]
    assert ctx_count == 0
    conn.close()


def test_tokenize_doc_prunes_stale_chunks_when_chunk_boundaries_shift(
    monkeypatch, tmp_path
):
    from sova import cli

    monkeypatch.setattr(
        cli,
        "get_token_counts_batch",
        lambda texts: [len(text.split()) for text in texts],
    )

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            path TEXT NOT NULL,
            line_count INTEGER,
            expected_chunks INTEGER
        );
        CREATE TABLE sections (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            level INTEGER NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL,
            section_id INTEGER,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            word_count INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB,
            embedding_signature TEXT,
            section_path TEXT NOT NULL DEFAULT '',
            search_text TEXT NOT NULL DEFAULT '',
            is_index INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE chunk_contexts (
            chunk_id INTEGER PRIMARY KEY,
            context TEXT NOT NULL,
            model TEXT NOT NULL
        );
        """
    )

    md = tmp_path / "doc.md"
    first_lines = ["# H"] + ["w"] * 820 + [""] + ["tail"] * 20
    md.write_text("\n".join(first_lines) + "\n", encoding="utf-8")
    cli._tokenize_doc(cli._PreparedSource("doc", md, "sig-1"), conn)
    starts_before = [
        r[0]
        for r in conn.execute(
            "SELECT start_line FROM chunks ORDER BY start_line"
        ).fetchall()
    ]
    assert starts_before == [1, 768, 823]

    second_lines = ["intro"] * 30 + first_lines
    md.write_text("\n".join(second_lines) + "\n", encoding="utf-8")
    prepared = cli._tokenize_doc(cli._PreparedSource("doc", md, "sig-2"), conn)
    assert prepared is not None
    _, parsed_chunks, _ = prepared
    starts_after = [
        r[0]
        for r in conn.execute(
            "SELECT start_line FROM chunks ORDER BY start_line"
        ).fetchall()
    ]
    assert starts_after == [1, 31, 798, 853]
    expected = conn.execute(
        "SELECT expected_chunks FROM documents WHERE name = 'doc'"
    ).fetchone()[0]
    assert expected == len(parsed_chunks)
    conn.close()


def test_tokenize_doc_keeps_context_and_embedding_when_only_section_ids_reorder(
    monkeypatch, tmp_path
):
    from sova import cli

    monkeypatch.setattr(
        cli,
        "get_token_counts_batch",
        lambda texts: [len(text.split()) for text in texts],
    )

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            path TEXT NOT NULL,
            line_count INTEGER,
            expected_chunks INTEGER
        );
        CREATE TABLE sections (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            level INTEGER NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL,
            section_id INTEGER,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            word_count INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB,
            embedding_signature TEXT,
            section_path TEXT NOT NULL DEFAULT '',
            search_text TEXT NOT NULL DEFAULT '',
            is_index INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE chunk_contexts (
            chunk_id INTEGER PRIMARY KEY,
            context TEXT NOT NULL,
            model TEXT NOT NULL
        );
        """
    )

    md1 = tmp_path / "doc1.md"
    md2 = tmp_path / "doc2.md"
    md1.write_text("# A\n\n" + "\n".join(["x"] * 20) + "\n", encoding="utf-8")
    md2.write_text("# B\n\n" + "\n".join(["y"] * 20) + "\n", encoding="utf-8")

    cli._tokenize_doc(cli._PreparedSource("doc1", md1, "sig-1"), conn)
    cli._tokenize_doc(cli._PreparedSource("doc2", md2, "sig-2"), conn)

    row = conn.execute(
        "SELECT id FROM chunks WHERE doc_id = (SELECT id FROM documents WHERE name = 'doc1')"
    ).fetchone()
    chunk_id = row[0]
    conn.execute("UPDATE chunks SET embedding = ? WHERE id = ?", (b"emb", chunk_id))
    conn.execute(
        "INSERT INTO chunk_contexts (chunk_id, context, model) VALUES (?, ?, ?)",
        (chunk_id, "ctx", "m"),
    )
    conn.commit()

    cli._tokenize_doc(cli._PreparedSource("doc1", md1, "sig-1"), conn)

    chunk = conn.execute(
        "SELECT embedding FROM chunks WHERE id = ?", (chunk_id,)
    ).fetchone()
    assert chunk[0] == b"emb"
    ctx_count = conn.execute(
        "SELECT COUNT(*) FROM chunk_contexts WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()[0]
    assert ctx_count == 1
    conn.close()


def test_generate_contexts_retries_on_empty_response(monkeypatch):
    from sova import cli

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL,
            start_line INTEGER NOT NULL,
            embedding BLOB
        );
        CREATE TABLE chunk_contexts (
            chunk_id INTEGER PRIMARY KEY,
            context TEXT NOT NULL,
            model TEXT NOT NULL
        );
        """
    )
    conn.execute("INSERT INTO chunks (id, doc_id, start_line) VALUES (1, 1, 10)")
    conn.commit()

    chunks = [{"start_line": 10, "text": "first"}]
    sections: list[dict] = []

    responses = iter(["   ", "ctx after retry"])
    stops: list[str] = []
    monkeypatch.setattr(
        cli, "generate_context", lambda *args, **kwargs: next(responses)
    )
    monkeypatch.setattr(cli, "stop_server", lambda url, **kwargs: stops.append(url))

    cli._generate_contexts("doc", 1, chunks, sections, conn)

    row = conn.execute(
        "SELECT context FROM chunk_contexts WHERE chunk_id = 1"
    ).fetchone()
    assert row == ("ctx after retry",)
    assert len(stops) == 1
    conn.close()


def test_forced_context_migration_resumes_rows_already_written_by_new_pipeline(
    monkeypatch,
):
    from sova import cli

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL,
            start_line INTEGER NOT NULL, embedding BLOB,
            embedding_signature TEXT
        );
        CREATE TABLE chunk_contexts (
            chunk_id INTEGER PRIMARY KEY, context TEXT NOT NULL, model TEXT NOT NULL,
            pipeline_signature TEXT NOT NULL
        );
        INSERT INTO chunks (id, doc_id, start_line) VALUES (1, 1, 10), (2, 1, 20);
        """
    )
    current = cli._context_pipeline_signature()
    conn.execute(
        "INSERT INTO chunk_contexts VALUES (1, 'already done', ?, ?)",
        (cli.config.CONTEXT_MODEL, current),
    )
    conn.commit()
    generated: list[str] = []
    monkeypatch.setattr(
        cli,
        "generate_context",
        lambda _name, _section, text, _prev, _next: (
            generated.append(text) or "new context"
        ),
    )

    cli._generate_contexts(
        "doc",
        1,
        [{"start_line": 10, "text": "first"}, {"start_line": 20, "text": "second"}],
        [],
        conn,
        force_rebuild_context=True,
    )

    assert generated == ["second"]
    assert conn.execute("SELECT COUNT(*) FROM chunk_contexts").fetchone()[0] == 2
    conn.close()


def test_prune_missing_documents_removes_stale_search_rows():
    from sova import cli

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE documents (id INTEGER PRIMARY KEY, name TEXT UNIQUE);
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL,
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
        );
        INSERT INTO documents VALUES (1, 'keep'), (2, 'deleted');
        INSERT INTO chunks VALUES (1, 1), (2, 2);
        """
    )

    removed = cli._prune_missing_documents(conn, {"keep"})

    assert removed == 1
    assert conn.execute("SELECT name FROM documents").fetchall() == [("keep",)]
    assert conn.execute("SELECT id FROM chunks").fetchall() == [(1,)]
    conn.close()


def test_prepare_source_reuses_extraction_checkpoint_before_tokenization(
    monkeypatch, tmp_path
):
    from sova import cli

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY, name TEXT UNIQUE, source_signature TEXT
        );
        CREATE TABLE index_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"stable pdf bytes")
    data_dir = tmp_path / "data"
    extractions: list[Path] = []
    monkeypatch.setattr(cli.config, "get_data_dir", lambda: data_dir)
    monkeypatch.setattr(
        cli,
        "extract_pdf",
        lambda path: extractions.append(path) or "# Extracted\n\nBody\n",
    )
    monkeypatch.setattr(cli, "status", lambda *_args, **_kwargs: None)

    first = cli._prepare_source("manual", pdf, None, conn)
    assert first is not None
    assert first.markdown_path.read_text(encoding="utf-8") == "# Extracted\n\nBody\n"
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0

    second = cli._prepare_source("manual", pdf, first.markdown_path, conn)

    assert second == first
    assert extractions == [pdf]
    assert cli.get_meta(conn, cli._source_checkpoint_key("manual")) == first.source_signature
    conn.close()


def test_current_tokenized_doc_requires_complete_matching_checkpoint(tmp_path):
    from sova import cli

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY, name TEXT UNIQUE,
            expected_chunks INTEGER, source_signature TEXT, chunk_signature TEXT
        );
        CREATE TABLE chunks (id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL);
        INSERT INTO documents
        VALUES (1, 'manual', 2, 'source-v1', 'chunks-v1');
        INSERT INTO chunks VALUES (1, 1), (2, 1);
        """
    )
    source = cli._PreparedSource("manual", tmp_path / "manual.md", "source-v1")

    assert cli._current_tokenized_doc_id(conn, source, "chunks-v1") == 1
    conn.execute("DELETE FROM chunks WHERE id = 2")
    assert cli._current_tokenized_doc_id(conn, source, "chunks-v1") is None
    conn.close()


def test_index_with_empty_source_preserves_existing_database_rows(
    monkeypatch, tmp_path
):
    from sova import cli

    db_path = tmp_path / "indexed.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE documents (id INTEGER PRIMARY KEY, name TEXT UNIQUE);
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL,
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
        );
        INSERT INTO documents VALUES (1, 'deleted');
        INSERT INTO chunks VALUES (1, 1);
        """
    )
    conn.commit()
    monkeypatch.setattr(cli.config, "get_docs_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "init_db", lambda: conn)
    monkeypatch.setattr(
        cli,
        "_sync_index_signatures",
        lambda _conn: cli._IndexSignatureState(False, False, "c", "e", "k"),
    )
    monkeypatch.setattr(cli, "find_docs", list)

    with pytest.raises(SystemExit) as exc:
        cli._run_index_mode()

    assert exc.value.code == 1
    check = sqlite3.connect(db_path)
    assert check.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    assert check.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
    check.close()


def test_index_prepares_every_document_before_loading_models(monkeypatch, tmp_path):
    from sova import cli

    events: list[str] = []
    conn = sqlite3.connect(":memory:")
    docs = [
        {"name": "one", "pdf": tmp_path / "one.pdf", "md": None},
        {"name": "two", "pdf": tmp_path / "two.pdf", "md": None},
    ]

    monkeypatch.setattr(cli.config, "get_docs_dir", lambda: tmp_path)
    monkeypatch.setattr(cli.config, "get_active_project_id", lambda: "test")
    monkeypatch.setattr(cli, "init_db", lambda: conn)
    monkeypatch.setattr(
        cli,
        "_sync_index_signatures",
        lambda _conn: cli._IndexSignatureState(False, False, "c", "e", "k"),
    )
    monkeypatch.setattr(cli, "find_docs", lambda: docs)
    monkeypatch.setattr(
        cli,
        "_prepare_source",
        lambda name, *_args: (
            events.append(f"prepare:{name}")
            or cli._PreparedSource(name, tmp_path / f"{name}.md", "sig")
        ),
    )
    monkeypatch.setattr(
        cli,
        "_tokenize_doc",
        lambda source, *_args: (
            events.append(f"tokenize:{source.name}")
            or (1, [{"start_line": 1, "text": source.name}], [])
        ),
    )
    monkeypatch.setattr(cli, "_current_tokenized_doc_id", lambda *_args: None)
    monkeypatch.setattr(cli, "_context_work_pending", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(cli, "_embedding_work_pending", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        cli,
        "check_servers",
        lambda **kwargs: events.append(f"load:{kwargs['mode']}") or (True, "ready"),
    )
    monkeypatch.setattr(
        cli,
        "_generate_contexts",
        lambda name, *_args, **_kwargs: events.append(f"context:{name}"),
    )
    monkeypatch.setattr(
        cli,
        "_load_prepared_doc",
        lambda *_args: ([{"start_line": 1, "text": "x"}], []),
    )
    monkeypatch.setattr(
        cli,
        "_embed_doc",
        lambda name, *_args, **_kwargs: events.append(f"embed:{name}"),
    )
    monkeypatch.setattr(
        cli,
        "stop_server",
        lambda url, **_kwargs: events.append(f"stop:{url}"),
    )
    monkeypatch.setattr(cli, "run_embedding_canary", lambda: events.append("canary"))
    monkeypatch.setattr(
        cli, "quantize_vectors", lambda _conn: events.append("finalize")
    )
    monkeypatch.setattr(cli, "_prune_missing_documents", lambda *_args: 0)
    monkeypatch.setattr(cli, "_commit_index_signatures", lambda *_args: None)
    monkeypatch.setattr(
        cli, "get_cache", lambda: type("Cache", (), {"clear": lambda self: None})()
    )
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(cli, "emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "report_scope", lambda *_args: None)
    monkeypatch.setattr(cli, "_report_phase_runtime", lambda *_args, **_kwargs: None)

    cli._run_index_mode()

    assert events.index("prepare:one") < events.index("prepare:two")
    embed_loads = [i for i, event in enumerate(events) if event == "load:index_embed"]
    assert len(embed_loads) == 2
    assert events.index("prepare:two") < embed_loads[0]
    assert embed_loads[0] < events.index("tokenize:one")
    assert events.index("tokenize:two") < events.index("load:index_context")
    assert events.index("load:index_context") < events.index("context:one")
    assert events.index("context:two") < embed_loads[1]
    assert embed_loads[1] < events.index("embed:one")
    assert events.index("embed:two") < events.index("finalize")


def test_sync_index_signatures_marks_rebuild_without_immediate_data_clear(monkeypatch):
    from sova import cli

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            embedding BLOB
        );
        CREATE TABLE chunk_contexts (
            chunk_id INTEGER PRIMARY KEY,
            context TEXT NOT NULL,
            model TEXT NOT NULL
        );
        CREATE TABLE index_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.execute("INSERT INTO chunks (id, embedding) VALUES (1, ?)", (b"emb",))
    conn.execute(
        "INSERT INTO chunk_contexts (chunk_id, context, model) VALUES (1, 'ctx', 'm1')"
    )
    conn.execute(
        "INSERT INTO index_meta (key, value) VALUES (?, ?)",
        ("pipeline.context.signature", "old"),
    )
    conn.execute(
        "INSERT INTO index_meta (key, value) VALUES (?, ?)",
        ("pipeline.embedding.signature", "same-embed"),
    )
    conn.execute(
        "INSERT INTO index_meta (key, value) VALUES (?, ?)",
        ("pipeline.chunk.signature", "same-chunk"),
    )
    conn.commit()

    monkeypatch.setattr(cli, "_context_pipeline_signature", lambda: "new-context")
    monkeypatch.setattr(cli, "_embedding_pipeline_signature", lambda: "same-embed")
    monkeypatch.setattr(cli, "_chunk_pipeline_signature", lambda: "same-chunk")

    state = cli._sync_index_signatures(conn)

    embedding = conn.execute("SELECT embedding FROM chunks WHERE id = 1").fetchone()[0]
    assert embedding == b"emb"
    ctx_count = conn.execute("SELECT COUNT(*) FROM chunk_contexts").fetchone()[0]
    assert ctx_count == 1
    assert state.force_rebuild_context is True
    assert state.force_rebuild_embed is True

    stored = conn.execute(
        "SELECT value FROM index_meta WHERE key = 'pipeline.context.signature'"
    ).fetchone()[0]
    assert stored == "old"

    cli._commit_index_signatures(conn, state)
    stored = conn.execute(
        "SELECT value FROM index_meta WHERE key = 'pipeline.context.signature'"
    ).fetchone()[0]
    assert stored == "new-context"
    conn.close()


def test_sync_index_signatures_resumes_current_rows_without_false_rebuild(monkeypatch):
    from sova import cli

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY, expected_chunks INTEGER,
            chunk_signature TEXT
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY, doc_id INTEGER,
            embedding BLOB, embedding_signature TEXT
        );
        CREATE TABLE chunk_contexts (
            chunk_id INTEGER PRIMARY KEY, context TEXT NOT NULL,
            model TEXT NOT NULL, pipeline_signature TEXT
        );
        CREATE TABLE index_meta (
            key TEXT PRIMARY KEY, value TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO documents VALUES (1, 2, 'chunks-current');
        INSERT INTO chunks VALUES
            (1, 1, X'01', 'embed-current'),
            (2, 1, NULL, NULL);
        INSERT INTO chunk_contexts
        VALUES (1, 'saved context', 'model', 'context-current');
        """
    )
    events: list[tuple[str, str]] = []
    monkeypatch.setattr(cli, "_context_pipeline_signature", lambda: "context-current")
    monkeypatch.setattr(cli, "_embedding_pipeline_signature", lambda: "embed-current")
    monkeypatch.setattr(cli, "_chunk_pipeline_signature", lambda: "chunks-current")
    monkeypatch.setattr(
        cli,
        "emit",
        lambda event, message, **_kwargs: events.append((event, message)),
    )

    state = cli._sync_index_signatures(conn)

    assert state.force_rebuild_context is False
    assert state.force_rebuild_embed is False
    assert events == [
        ("pipeline_resuming", "Resuming interrupted index. Reusing completed work.")
    ]
    assert conn.execute("SELECT context FROM chunk_contexts").fetchone()[0] == (
        "saved context"
    )
    assert conn.execute("SELECT embedding FROM chunks WHERE id = 1").fetchone()[0] == (
        b"\x01"
    )
    conn.close()


def test_list_mode_reports_structured_error_on_sqlite_operational_error(monkeypatch):
    from sova import cli

    lines: list[tuple[str, dict]] = []

    monkeypatch.setattr(cli, "find_docs", list)
    monkeypatch.setattr(
        cli,
        "list_docs",
        lambda _docs: (_ for _ in ()).throw(sqlite3.OperationalError("")),
    )
    monkeypatch.setattr(
        cli,
        "report_error",
        lambda summary, **kwargs: lines.append((summary, kwargs)),
    )

    with pytest.raises(SystemExit) as exc:
        cli._run_list_mode()

    assert exc.value.code == 1
    assert lines[0][0] == "database extension unavailable"
    assert "sova-install" in lines[0][1]["action"]


def test_reset_is_not_a_command_and_fails_as_unknown_project(monkeypatch):
    from sova import cli

    monkeypatch.setattr(sys, "argv", ["sova", "reset", "proj"])
    monkeypatch.setattr(
        cli,
        "_activate_project_from_ref",
        lambda _ref, allow_create_from_dir=False: (_ for _ in ()).throw(SystemExit(1)),
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1


def test_embed_doc_persists_partial_window_progress_on_failure(monkeypatch):
    from sova import cli

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL,
            section_id INTEGER,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            word_count INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB,
            is_index INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE chunk_contexts (
            chunk_id INTEGER PRIMARY KEY,
            context TEXT NOT NULL,
            model TEXT NOT NULL
        );
        """
    )
    for line in range(1, 21):
        conn.execute(
            """
            INSERT INTO chunks
                (id, doc_id, section_id, start_line, end_line, word_count, text, embedding, is_index)
            VALUES (?, 1, NULL, ?, ?, ?, ?, NULL, 0)
            """,
            (line, line, line, 1, f"chunk {line}"),
        )
    conn.commit()

    chunks = [{"start_line": line, "text": f"chunk {line}"} for line in range(1, 21)]
    attempts = {"count": 0}

    def fake_get_embeddings_batch(texts, on_batch=None):
        attempts["count"] += 1
        if attempts["count"] == 1:
            batch = [[1.0, 0.0] for _ in range(12)]
            if on_batch:
                on_batch(
                    list(range(12)),
                    batch,
                    {"batch_size": 12, "workers": 1, "duration_s": 0.01},
                )
        raise RuntimeError("Remote end closed connection without response")

    monkeypatch.setattr(cli, "get_embeddings_batch", fake_get_embeddings_batch)
    monkeypatch.setattr(cli, "embedding_to_blob", lambda _emb: b"emb")
    monkeypatch.setattr(cli, "stop_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "run_embedding_canary", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli.time, "sleep", lambda *_: None)

    with pytest.raises(RuntimeError, match="embedding failed for doc"):
        cli._embed_doc("doc", 1, chunks, [], conn)

    embedded = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE doc_id = 1 AND embedding IS NOT NULL"
    ).fetchone()[0]
    assert embedded == 12
    conn.close()


def test_embed_doc_retry_only_embeds_unfinished_tail(monkeypatch):
    from sova import cli

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL,
            section_id INTEGER,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            word_count INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB,
            is_index INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE chunk_contexts (
            chunk_id INTEGER PRIMARY KEY,
            context TEXT NOT NULL,
            model TEXT NOT NULL
        );
        """
    )
    for line in range(1, 21):
        conn.execute(
            """
            INSERT INTO chunks
                (id, doc_id, section_id, start_line, end_line, word_count, text, embedding, is_index)
            VALUES (?, 1, NULL, ?, ?, ?, ?, NULL, 0)
            """,
            (line, line, line, 1, f"chunk {line}"),
        )
    conn.commit()

    chunks = [{"start_line": line, "text": f"chunk {line}"} for line in range(1, 21)]
    seen_lengths: list[int] = []
    calls = {"count": 0}

    def fake_get_embeddings_batch(texts, on_batch=None):
        calls["count"] += 1
        seen_lengths.append(len(texts))
        if calls["count"] == 1:
            first_batch = [[1.0, 0.0] for _ in range(12)]
            if on_batch:
                on_batch(
                    list(range(12)),
                    first_batch,
                    {"batch_size": 12, "workers": 1, "duration_s": 0.01},
                )
            raise RuntimeError("Remote end closed connection without response")
        full_batch = [[2.0, 0.0] for _ in texts]
        if on_batch:
            on_batch(
                list(range(len(texts))),
                full_batch,
                {"batch_size": len(texts), "workers": 1, "duration_s": 0.01},
            )
        return full_batch

    monkeypatch.setattr(cli, "get_embeddings_batch", fake_get_embeddings_batch)
    monkeypatch.setattr(cli, "embedding_to_blob", lambda _emb: b"emb")
    monkeypatch.setattr(cli, "stop_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "run_embedding_canary", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli.time, "sleep", lambda *_: None)

    cli._embed_doc("doc", 1, chunks, [], conn)

    assert seen_lengths == [20, 8]
    embedded = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE doc_id = 1 AND embedding IS NOT NULL"
    ).fetchone()[0]
    assert embedded == 20
    conn.close()


def test_embed_doc_reports_absolute_progress(monkeypatch):
    from sova import cli

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL,
            section_id INTEGER,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            word_count INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB,
            is_index INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE chunk_contexts (
            chunk_id INTEGER PRIMARY KEY,
            context TEXT NOT NULL,
            model TEXT NOT NULL
        );
        """
    )
    for line in range(1, 21):
        embedding = b"old" if line <= 10 else None
        conn.execute(
            """
            INSERT INTO chunks
                (id, doc_id, section_id, start_line, end_line, word_count, text, embedding, is_index)
            VALUES (?, 1, NULL, ?, ?, ?, ?, ?, 0)
            """,
            (line, line, line, 1, f"chunk {line}", embedding),
        )
    conn.commit()

    chunks = [{"start_line": line, "text": f"chunk {line}"} for line in range(1, 21)]
    reports: list[tuple[str, int, int]] = []

    def fake_get_embeddings_batch(texts, on_batch=None):
        batch = [[1.0, 0.0] for _ in texts]
        if on_batch:
            on_batch(
                list(range(len(texts))),
                batch,
                {"batch_size": len(texts), "workers": 1, "duration_s": 0.01},
            )
        return batch

    monkeypatch.setattr(cli, "get_embeddings_batch", fake_get_embeddings_batch)
    monkeypatch.setattr(cli, "embedding_to_blob", lambda _emb: b"emb")
    monkeypatch.setattr(
        cli,
        "progress",
        lambda phase, done, total, **kwargs: reports.append((phase, done, total)),
    )

    cli._embed_doc("doc", 1, chunks, [], conn)

    assert ("embed", 20, 20) in reports
    conn.close()


def test_context_pipeline_signature_tracks_exact_model_artifact(monkeypatch):
    from sova import cli

    current = cli._context_pipeline_signature()
    monkeypatch.setattr(cli.config, "CONTEXT_MODEL_HF_FILE", "another-model.gguf")

    assert cli._context_pipeline_signature() != current


def test_download_reports_models_as_structured_items(monkeypatch):
    from sova import cli

    events: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(cli, "is_service_installed", lambda _label: True)
    monkeypatch.setattr(cli, "is_model_cached", lambda _label: True)
    monkeypatch.setattr(
        cli,
        "emit",
        lambda event_type, message, **kwargs: events.append(
            (event_type, message, kwargs)
        ),
    )

    cli._run_download_mode()

    cached = [event for event in events if event[0] == "download_cached"]
    assert [(message, kwargs["item"]) for _, message, kwargs in cached] == [
        ("cached", "embedding"),
        ("cached", "chat"),
    ]
    assert events[-1][1] == "All models are cached"


def test_download_failure_surfaces_service_diagnostics(monkeypatch):
    from sova import cli

    errors: list[dict] = []
    monkeypatch.setattr(
        cli,
        "_DOWNLOAD_SERVICES",
        [("chat", "com.sova.chat", cli.config.CONTEXT_SERVER_URL)],
    )
    monkeypatch.setattr(cli, "_DOWNLOAD_STALL_TIMEOUT_S", -1.0)
    monkeypatch.setattr(cli, "is_service_installed", lambda _label: True)
    monkeypatch.setattr(cli, "is_model_cached", lambda _label: False)
    monkeypatch.setattr(cli, "is_service_running", lambda _label: False)
    monkeypatch.setattr(cli, "get_model_status", lambda _label: "starting")
    monkeypatch.setattr(cli, "get_service_diagnostics", lambda _url: "file not found")
    monkeypatch.setattr(cli, "start_service", lambda _label: None)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(cli, "emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli,
        "_report_error_block",
        lambda summary, **kwargs: errors.append({"summary": summary, **kwargs}),
    )

    with pytest.raises(SystemExit) as exc:
        cli._run_download_mode()

    assert exc.value.code == 1
    assert errors == [
        {
            "summary": "model download failed",
            "cause": "file not found",
            "action": "check the model configuration and re-run: sova-install",
            "detail": "log: ~/.sova/logs/chat.err.log",
        }
    ]
