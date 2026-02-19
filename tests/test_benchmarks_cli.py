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
    import benchmarks.judge as judge
    import benchmarks.search_interface as search_interface
    import sova.config as config

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
    import benchmarks.run_benchmark as run_benchmark
    import benchmarks.search_interface as search_interface

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
    (bench_dir / "ground_truth.json").write_text(json.dumps(ground_truth))

    monkeypatch.setattr(search_interface, "clear_cache", lambda: None)
    monkeypatch.setattr(
        search_interface,
        "measure_latency",
        lambda _queries: {"total_times": [100.0, 110.0, 90.0]},
    )

    closed: list[bool] = []
    monkeypatch.setattr(search_interface, "close_backend", lambda: closed.append(True))

    def _raise_interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(run_benchmark, "run_search", _raise_interrupt)

    with pytest.raises(KeyboardInterrupt):
        bench_cli.cmd_run(name="interrupt-case")

    assert closed == [True]
