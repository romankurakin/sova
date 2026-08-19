"""Tests for benchmarks CLI behaviors."""

import json
from pathlib import Path

import pytest


class _DummyLive:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        return False

    def update(self, _renderable):
        return None


def test_cmd_judge_closes_backend_on_early_stop(monkeypatch, tmp_path: Path):
    import benchmarks.__main__ as bench_cli
    from benchmarks import judge, search_interface
    from sova import config

    db_path = tmp_path / "sova.db"
    db_path.write_text("")
    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(bench_cli, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(bench_cli, "_BENCH_DIR", tmp_path)
    monkeypatch.setattr(bench_cli, "Live", _DummyLive)

    spec = judge.QuerySpec("t01", "test query", "conceptual", [])
    monkeypatch.setattr(judge, "QUERY_SET", [spec])
    monkeypatch.setattr(judge, "JUDGE_MODEL", "test-model")

    def _raise_judge_error(*_args, **_kwargs):
        raise judge.JudgeError("rate limited")

    monkeypatch.setattr(judge, "judge_query", _raise_judge_error)

    closed: list[bool] = []
    monkeypatch.setattr(search_interface, "close_backend", lambda: closed.append(True))

    with pytest.raises(SystemExit) as exc:
        bench_cli.cmd_judge()

    assert exc.value.code == 1
    assert closed == [True]


def test_cmd_run_closes_backend_on_interrupt(monkeypatch, tmp_path: Path):
    import benchmarks.__main__ as bench_cli
    from benchmarks import run_benchmark, search_interface

    bench_dir = tmp_path / "bench"
    bench_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(bench_cli, "_BENCH_DIR", bench_dir)

    ground_truth = {
        "queries": [
            {
                "id": "t01",
                "query": "test query",
                "category": "conceptual",
                "judgments": [{"chunk_id": 1, "score": 3}],
            }
        ]
    }
    ground_truth["suite"] = {
        "id": "search",
        "schema_version": 2,
        "corpus": {"database_sha256": None},
    }
    (bench_dir / bench_cli._SUITE_FILENAME).write_text(json.dumps(ground_truth))

    monkeypatch.setattr(search_interface, "clear_cache", lambda: None)
    monkeypatch.setattr(
        search_interface,
        "measure_latency",
        lambda _queries, **_kwargs: {"total_times": [100.0, 110.0, 90.0]},
    )

    closed: list[bool] = []
    monkeypatch.setattr(search_interface, "close_backend", lambda: closed.append(True))

    def _raise_interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(run_benchmark, "run_search", _raise_interrupt)

    with pytest.raises(KeyboardInterrupt):
        bench_cli.cmd_run(name="interrupt-case")

    assert closed == [True]


def test_cmd_run_fails_on_unjudged(monkeypatch, tmp_path: Path):
    import benchmarks.__main__ as bench_cli
    from benchmarks import run_benchmark, search_interface

    bench_dir = tmp_path / "bench"
    bench_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(bench_cli, "_BENCH_DIR", bench_dir)

    ground_truth = {
        "queries": [
            {
                "id": "t01",
                "query": "test query",
                "category": "conceptual",
                "judgments": [{"chunk_id": 1, "score": 3}],
            }
        ]
    }
    ground_truth["suite"] = {
        "id": "search",
        "schema_version": 2,
        "corpus": {"database_sha256": None},
    }
    gt_path = bench_dir / bench_cli._SUITE_FILENAME
    gt_path.write_text(json.dumps(ground_truth))

    monkeypatch.setattr(search_interface, "clear_cache", lambda: None)
    monkeypatch.setattr(
        search_interface,
        "measure_latency",
        lambda _queries, **_kwargs: {"total_times": [100.0, 110.0, 90.0]},
    )
    monkeypatch.setattr(
        run_benchmark,
        "run_search",
        lambda *_args, **_kwargs: [
            {"chunk_id": 2, "doc": "d", "text": "t", "score": 0.2, "section_id": None}
        ],
    )

    closed: list[bool] = []
    monkeypatch.setattr(search_interface, "close_backend", lambda: closed.append(True))

    with pytest.raises(RuntimeError, match="ground truth contains unjudged chunks"):
        bench_cli.cmd_run(name="missing-judgments")

    assert closed == [True]
    loaded = json.loads(gt_path.read_text())
    assert loaded["queries"][0]["judgments"] == [{"chunk_id": 1, "score": 3}]


def test_cmd_run_defaults_to_one_deterministic_pass(monkeypatch, tmp_path: Path):
    import benchmarks.__main__ as bench_cli
    from benchmarks import run_benchmark, search_interface

    bench_dir = tmp_path / "bench"
    bench_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(bench_cli, "_BENCH_DIR", bench_dir)

    ground_truth = {
        "queries": [
            {
                "id": "t01",
                "query": "test query",
                "category": "conceptual",
                "judgments": [{"chunk_id": 1, "score": 3}],
            }
        ]
    }
    ground_truth["suite"] = {
        "id": "search",
        "schema_version": 2,
        "corpus": {"database_sha256": None},
    }
    (bench_dir / bench_cli._SUITE_FILENAME).write_text(json.dumps(ground_truth))

    latency_samples = [
        [100.0, 110.0, 90.0],
        [200.0, 210.0, 190.0],
        [300.0, 310.0, 290.0],
    ]
    idx = {"value": 0}

    def _measure_latency(_queries, **_kwargs):
        sample = latency_samples[idx["value"]]
        idx["value"] += 1
        return {"total_times": sample}

    clear_calls: list[bool] = []
    close_calls: list[bool] = []
    monkeypatch.setattr(
        search_interface, "clear_cache", lambda: clear_calls.append(True)
    )
    monkeypatch.setattr(
        search_interface, "close_backend", lambda: close_calls.append(True)
    )
    monkeypatch.setattr(search_interface, "measure_latency", _measure_latency)
    monkeypatch.setattr(
        run_benchmark,
        "run_search",
        lambda *_args, **_kwargs: [
            {"chunk_id": 1, "doc": "d", "text": "t", "score": 1.0, "section_id": 1}
        ],
    )

    result_path = bench_dir / "results" / "three-pass-default.json"
    result_path.unlink(missing_ok=True)

    bench_cli.cmd_run(name="three-pass-default")

    data = json.loads(result_path.read_text())
    assert data["runs"] == 1
    # The first sample is an untimed warm-up; the second is measured.
    assert data["latency_ms"]["p50"] == pytest.approx(200.0)
    assert len(clear_calls) == 1
    assert len(close_calls) == 1
    result_path.unlink(missing_ok=True)


def test_cmd_run_respects_internal_runs_override(monkeypatch, tmp_path: Path):
    import benchmarks.__main__ as bench_cli
    from benchmarks import run_benchmark, search_interface

    bench_dir = tmp_path / "bench"
    bench_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(bench_cli, "_BENCH_DIR", bench_dir)

    ground_truth = {
        "queries": [
            {
                "id": "t01",
                "query": "test query",
                "category": "conceptual",
                "judgments": [{"chunk_id": 1, "score": 3}],
            }
        ]
    }
    ground_truth["suite"] = {
        "id": "search",
        "schema_version": 2,
        "corpus": {"database_sha256": None},
    }
    (bench_dir / bench_cli._SUITE_FILENAME).write_text(json.dumps(ground_truth))

    clear_calls: list[bool] = []
    close_calls: list[bool] = []
    monkeypatch.setattr(
        search_interface, "clear_cache", lambda: clear_calls.append(True)
    )
    monkeypatch.setattr(
        search_interface, "close_backend", lambda: close_calls.append(True)
    )
    monkeypatch.setattr(
        search_interface,
        "measure_latency",
        lambda _queries, **_kwargs: {"total_times": [150.0, 160.0, 140.0]},
    )
    monkeypatch.setattr(
        run_benchmark,
        "run_search",
        lambda *_args, **_kwargs: [
            {"chunk_id": 1, "doc": "d", "text": "t", "score": 1.0, "section_id": 1}
        ],
    )

    result_path = bench_dir / "results" / "runs-override.json"
    result_path.unlink(missing_ok=True)

    bench_cli.cmd_run(name="runs-override", runs=1)

    data = json.loads(result_path.read_text())
    assert data["runs"] == 1
    assert len(clear_calls) == 1
    assert len(close_calls) == 1
    result_path.unlink(missing_ok=True)


def test_build_ground_truth_uses_compact_schema():
    import benchmarks.__main__ as bench_cli

    payload = bench_cli._build_ground_truth(
        [
            {
                "id": "t01",
                "query": "q",
                "category": "conceptual",
                "subtopics": [],
                "judgments": [],
            }
        ]
    )
    assert payload == {"queries": payload["queries"]}
    assert set(payload.keys()) == {"queries"}


def test_load_ground_truth_preserves_current_envelope(tmp_path: Path):
    import benchmarks.__main__ as bench_cli

    path = tmp_path / "draft-suite.json"
    path.write_text(
        json.dumps(
            {
                "queries": [
                    {"id": "q1", "query": "x", "category": "c", "judgments": []}
                ],
            }
        )
    )

    loaded = bench_cli._load_ground_truth(path)
    assert loaded is not None
    assert set(loaded.keys()) == {"queries"}
    assert loaded["queries"][0]["id"] == "q1"
