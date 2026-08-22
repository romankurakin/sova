"""Tests for the renderer-independent terminal event contract."""

import io
import json

from rich.console import Console

from sova.ui import (
    Event,
    JsonRenderer,
    LiveRenderer,
    PlainRenderer,
    Reporter,
    _status_renderable,
)


def test_plain_renderer_separates_payload_from_operational_output():
    stdout = io.StringIO()
    stderr = io.StringIO()
    renderer = PlainRenderer(stdout=stdout, stderr=stderr, progress_interval_s=0)

    renderer.emit(Event("run_started", "Indexing docs  1 document"))
    renderer.emit(
        Event(
            "result",
            "doc.md:1-3",
            data={
                "location": "doc.md:1-3",
                "text": "Relevant text",
                "last": True,
            },
        )
    )

    assert stdout.getvalue() == "doc.md:1-3\n  Relevant text\n"
    assert stderr.getvalue() == "Indexing docs  1 document\n"
    assert "\x1b" not in stdout.getvalue() + stderr.getvalue()


def test_structured_work_item_keeps_activity_in_the_state_column():
    stderr = io.StringIO()
    renderer = PlainRenderer(stdout=io.StringIO(), stderr=stderr)
    event = Event(
        "status",
        "starting",
        phase="download",
        item="chat",
        data={"item_width": 9},
    )

    renderer.emit(event)
    assert stderr.getvalue() == "download  chat  starting\n"

    rendered = io.StringIO()
    Console(file=rendered, width=80, highlight=False).print(_status_renderable(event))
    line = rendered.getvalue()
    assert line.index("chat") < line.index("starting")


def test_plain_renderer_throttles_progress_but_keeps_final(monkeypatch):
    stdout = io.StringIO()
    stderr = io.StringIO()
    times = iter([100.0, 101.0, 102.0])
    monkeypatch.setattr("sova.ui.time.monotonic", lambda: next(times))
    renderer = PlainRenderer(stdout=stdout, stderr=stderr, progress_interval_s=30)

    renderer.emit(Event("progress", "", phase="context", current=1, total=3))
    renderer.emit(Event("progress", "", phase="context", current=2, total=3))
    renderer.emit(Event("progress", "", phase="context", current=3, total=3))

    lines = stderr.getvalue().splitlines()
    assert len(lines) == 2
    assert "1/3" in lines[0]
    assert "3/3" in lines[1]


def test_json_renderer_emits_one_valid_object_per_event():
    stdout = io.StringIO()
    renderer = JsonRenderer(stdout)

    renderer.emit(Event("status", "Preparing model", phase="context"))
    renderer.emit(
        Event(
            "progress",
            "",
            phase="context",
            item="manual",
            current=2,
            total=10,
            unit="chunks",
        )
    )

    rows = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert rows == [
        {
            "level": "info",
            "message": "Preparing model",
            "phase": "context",
            "type": "status",
        },
        {
            "current": 2,
            "item": "manual",
            "level": "info",
            "message": "",
            "phase": "context",
            "total": 10,
            "type": "progress",
            "unit": "chunks",
        },
    ]


def test_low_memory_warning_is_edge_triggered():
    stdout = io.StringIO()
    reporter = Reporter(JsonRenderer(stdout))

    reporter.runtime(memory_headroom_gib=0.4)
    reporter.runtime(memory_headroom_gib=0.3)
    reporter.runtime(memory_headroom_gib=1.2)
    reporter.runtime(memory_headroom_gib=0.2)

    events = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [event["type"] for event in events] == ["warning", "warning"]


def test_error_is_one_structured_json_event():
    stdout = io.StringIO()
    renderer = JsonRenderer(stdout)

    renderer.emit(
        Event(
            "error",
            "database not ready",
            level="error",
            data={"cause": "missing", "action": "run indexing"},
        )
    )

    payload = json.loads(stdout.getvalue())
    assert payload["message"] == "database not ready"
    assert payload["data"] == {"action": "run indexing", "cause": "missing"}


def test_plain_table_keeps_its_human_readable_title():
    stdout = io.StringIO()
    renderer = PlainRenderer(stdout=stdout, stderr=io.StringIO())

    renderer.emit(
        Event(
            "table",
            "Benchmark Results",
            data={
                "columns": ["Metric", "Value"],
                "alignments": ["left", "right"],
                "rows": [["nDCG", "0.9"], ["Latency", "12"]],
                "sections": [1],
            },
        )
    )

    assert stdout.getvalue().splitlines() == [
        "Benchmark Results",
        "Metric   Value",
        "nDCG       0.9",
        "",
        "Latency     12",
    ]


def test_live_renderer_keeps_one_status_region_across_events(monkeypatch):
    instances = []

    class FakeLive:
        def __init__(self, *, console, **_kwargs):
            self.console = console
            self.started = 0
            self.stopped = 0
            self.updates = []
            instances.append(self)

        def start(self):
            self.started += 1

        def stop(self):
            self.stopped += 1

        def update(self, value, *, refresh):
            self.updates.append((value, refresh))

    monkeypatch.setattr("sova.ui.Live", FakeLive)
    stderr = io.StringIO()
    renderer = LiveRenderer(stdout=io.StringIO(), stderr=stderr)

    renderer.emit(Event("status", "Loading", phase="context"))
    active = renderer._live
    renderer.emit(Event("warning", "Low memory", level="warning"))
    renderer.emit(Event("status", "Ready", phase="context"))

    assert len(instances) == 1
    assert renderer._live is active
    assert instances[0].stopped == 0

    renderer.emit(Event("completed", "Done"))
    assert instances[0].stopped == 1
    assert renderer._live is None
