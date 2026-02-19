"""Tests for cli module."""

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
