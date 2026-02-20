"""Tests for cli module."""

import sys
import sqlite3
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
            lambda url, suppress_interrupt=False: stops.append((url, suppress_interrupt)),
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
            "find_docs",
            lambda: [{"name": "doc1", "pdf": None, "md": Path("/tmp/doc1.md")}],
        )
        monkeypatch.setattr(
            cli,
            "_prepare_doc",
            lambda *args, **kwargs: (1, [{"start_line": 1, "text": "x"}], []),
        )
        monkeypatch.setattr(
            cli,
            "_generate_contexts",
            lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        monkeypatch.setattr(cli, "show_stats", lambda mode="index": None)
        monkeypatch.setattr(cli, "get_cache", lambda: DummyCache())
        monkeypatch.setattr(
            cli,
            "stop_server",
            lambda url, suppress_interrupt=False: stops.append((url, suppress_interrupt)),
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

    def test_search_uses_live_server_status_without_log_spam(self, monkeypatch):
        from sova import cli

        reports: list[tuple[str, str]] = []
        live_updates: list[str] = []

        class DummyProject:
            project_id = "proj"

        class DummyLive:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def update(self, renderable):
                live_updates.append(str(renderable))

        def fake_check_servers(on_status=None, mode="search"):
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
        monkeypatch.setattr(cli, "Live", DummyLive)
        monkeypatch.setattr(cli, "check_servers", fake_check_servers)
        monkeypatch.setattr(cli, "search_semantic", lambda *args, **kwargs: None)
        monkeypatch.setattr(cli, "report", lambda name, msg: reports.append((name, msg)))

        cli.main()

        # Intermediate server states should go through live update, not report().
        assert any("downloading" in u for u in live_updates)
        assert any("loading" in u for u in live_updates)
        assert ("server", "ready") in reports
        assert not any(
            n == "server" and ("downloading" in m or "loading" in m) for n, m in reports
        )


def test_index_reserved_token_fails_before_project_lookup(monkeypatch):
    from sova import cli

    captured: dict[str, str] = {}

    monkeypatch.setattr(sys, "argv", ["sova", "index", "list"])
    monkeypatch.setattr(
        cli,
        "_report_error_block",
        lambda summary, **kw: captured.update(
            {"summary": summary, "cause": kw.get("cause", ""), "action": kw.get("action", "")}
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
    assert "usage: sova" in out
    assert "{help,projects,remove,list,index}" in out


def test_help_flag_is_unknown_option(monkeypatch):
    from sova import cli

    captured: dict[str, str] = {}
    monkeypatch.setattr(sys, "argv", ["sova", "--help"])
    monkeypatch.setattr(
        cli,
        "_report_error_block",
        lambda summary, **kw: captured.update(
            {"summary": summary, "cause": kw.get("cause", ""), "action": kw.get("action", "")}
        ),
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2
    assert captured["summary"] == "unknown option"
    assert captured["action"] == "use: sova help"


def test_subcommand_help_flag_is_unknown_option(monkeypatch):
    from sova import cli

    captured: dict[str, str] = {}
    monkeypatch.setattr(sys, "argv", ["sova", "projects", "--help"])
    monkeypatch.setattr(
        cli,
        "_report_error_block",
        lambda summary, **kw: captured.update(
            {"summary": summary, "cause": kw.get("cause", ""), "action": kw.get("action", "")}
        ),
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2
    assert captured["summary"] == "unknown option"
    assert captured["action"] == "use: sova help"


class TestRuntimeReporting:
    def test_report_phase_runtime_uses_named_budget_fields(self, monkeypatch):
        from sova import cli

        lines: list[tuple[str, str]] = []

        monkeypatch.setattr(cli.time, "strftime", lambda _fmt: "12:34:56")
        monkeypatch.setattr(cli.config, "get_effective_available_gib", lambda: 7.4)
        monkeypatch.setattr(cli.config, "get_memory_reserve_gib", lambda _mode: 4.0)
        monkeypatch.setattr(
            cli,
            "_service_status_line",
            lambda _service, with_memory=False: "chat [green]running[/green]",
        )
        monkeypatch.setattr(cli, "report", lambda name, msg: lines.append((name, msg)))

        cli._report_phase_runtime("index.context", "chat", mode="index")

        assert ("phase", "index.context (updated 12:34:56)") in lines
        assert (
            "runtime",
            "free-for-model 3.4 GiB | chat [green]running[/green]",
        ) in lines

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


class TestIndexLiveView:
    def test_replaces_context_progress_line_in_place(self):
        from sova import cli

        view = cli._IndexLiveView()
        view.emit("doc", "arm_profile_architecture_reference_manual")
        view.emit("extract", "603,867 lines")
        view.emit("context", "1/15,531 chunks")
        view.emit("context", "3/15,531 chunks")
        view.emit("context", "5/15,531 chunks")

        events = list(view._events)
        assert len(events) == 3
        assert events[0].endswith("arm_profile_architecture_reference_manual")
        assert events[1].endswith("603,867 lines")
        assert events[2].endswith("5/15,531 chunks")

    def test_replaces_embed_progress_line_in_place(self):
        from sova import cli

        view = cli._IndexLiveView()
        view.emit("doc", "arm_profile_architecture_reference_manual")
        view.emit("embed", "12/15,531 chunks")
        view.emit("embed", "24/15,531 chunks")
        view.emit("embed", "36/15,531 chunks")

        events = list(view._events)
        assert len(events) == 2
        assert events[0].endswith("arm_profile_architecture_reference_manual")
        assert events[1].endswith("36/15,531 chunks")

    def test_replaces_server_status_line_in_place(self):
        from sova import cli

        view = cli._IndexLiveView()
        view.emit("event", "starting services")
        view.emit("server", "chat: downloading (1.5 GB)")
        view.emit("server", "chat: downloading (2.0 GB)")
        view.emit("server", "chat: loading")

        events = list(view._events)
        assert len(events) == 2
        assert events[0].endswith("starting services")
        assert events[1].endswith("chat: loading")


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
    monkeypatch.setattr(cli, "report", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_ACTIVE_INDEX_VIEW", object())

    cli._generate_contexts("doc", 1, chunks, sections, conn)
    count = conn.execute("SELECT COUNT(*) FROM chunk_contexts").fetchone()[0]
    assert count == 1

    # Retry should not fail and should keep a single row.
    cli._generate_contexts("doc", 1, chunks, sections, conn)
    count_after = conn.execute("SELECT COUNT(*) FROM chunk_contexts").fetchone()[0]
    assert count_after == 1

    conn.close()


def test_list_mode_reports_structured_error_on_sqlite_operational_error(monkeypatch):
    from sova import cli

    lines: list[tuple[str, str]] = []

    monkeypatch.setattr(cli, "find_docs", lambda: [])
    monkeypatch.setattr(
        cli,
        "list_docs",
        lambda _docs: (_ for _ in ()).throw(sqlite3.OperationalError("")),
    )
    monkeypatch.setattr(cli, "report", lambda name, msg: lines.append((name, msg)))

    with pytest.raises(SystemExit) as exc:
        cli._run_list_mode()

    assert exc.value.code == 1
    assert any(name == "error" and "database extension unavailable" in msg for name, msg in lines)
    assert any(name == "action" and "sova-install" in msg for name, msg in lines)


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
