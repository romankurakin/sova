"""Complete benchmark lifecycle with synthetic sources and fake model boundaries."""

import json
import subprocess

import pytest

from benchmarks import build_suite, judge, run_benchmark
from benchmarks.indexing import _load_passages
from benchmarks.queries import load_evaluation_queries
from benchmarks.report import build_report
from benchmarks.search_interface import SovaBackend
from benchmarks.storage import read, write
from tests.test_benchmarks_passages import make_reading_inputs, prepare


class FakeBackend(SovaBackend):
    def __init__(self, root, snapshot):
        self.snapshot, items = _load_passages(snapshot)
        self.items = {item["passage_id"]: item for item in items}
        self.manifest = {
            "sha256": "test-index",
            "database": "index.sqlite3",
            "inputs": {"models": {}},
        }
        self.index_path = root / "indexes/test/manifest.json"
        self.calls = 0
        self.models = self
        self.pool_size = None

    def start(self):
        pass

    def verify(self):
        pass

    def close(self):
        pass

    def query_embedding(self, query):
        self.calls += 1
        return [1.0, 0.0]

    def retrieve(self, query, **kwargs):
        return list(self.items)[: self.pool_size]


def fake_judge(job):
    response = {
        "grades": [
            {
                "passage_id": p["passage_id"],
                "grade": 0 if p["passage_id"] == "negative" else 3,
                "reason": "Synthetic grade",
                "quote": p["text"],
                "architecture_tags": [],
            }
            for p in job["payload"]["passages"]
        ]
    }
    return {"answer": response, "usage": None, "identity": {"synthetic": True}}


def make_workflow(tmp_path, pool_size=None):
    inputs = make_reading_inputs(tmp_path)
    snapshot = prepare(inputs)
    root = inputs.parent.parent
    write(
        root / "preparation.json",
        {
            "passages": str(snapshot.relative_to(root)),
            "index": "indexes/test/manifest.json",
        },
    )
    backend = FakeBackend(root, snapshot)
    backend.pool_size = pool_size
    queries = load_evaluation_queries()[:2]
    build_suite.prepare_pool(
        root, backend=backend, queries=queries, notify=lambda _: None
    )
    return root, backend, queries


@pytest.fixture
def workflow(tmp_path):
    return make_workflow(tmp_path)


def cli_results(items):
    return [
        {
            "rank": i,
            "document": item["document"],
            "location": f"{item['document']}.md:{item['start_line']}-{item['end_line']}",
            "start_line": item["start_line"],
            "end_line": item["end_line"],
            "text": item["text"],
        }
        for i, item in enumerate(items, 1)
    ]


def complete_judgments(root):
    for provider in ("codex", "claude", "codex", "claude"):
        judge.run_judge(root, provider, caller=fake_judge, notify=lambda _: None)
    return build_suite.freeze(root)


def test_complete_workflow_reuses_judgments_and_reports_without_models(
    workflow, monkeypatch
):
    root, backend, queries = workflow
    assert read(build_suite.freeze(root))["status"] == "reference_incomplete"
    suite = complete_judgments(root)
    assert read(suite)["status"] == "ready"
    receipts = list((root / "judgments").rglob("*.json"))
    calls = []
    judge.run_judge(
        root, "codex", caller=lambda job: calls.append(job), notify=lambda _: None
    )
    assert calls == []
    assert len(receipts) == len(list((root / "judgments").rglob("*.json")))

    def executor(request):
        return cli_results(backend.items.values()), {"elapsed_seconds": 0.1}

    monkeypatch.setattr(run_benchmark.llama_client, "stop_server", lambda *a, **k: None)
    saved = run_benchmark.run_benchmark(
        root, backend=backend, queries=queries, executor=executor, notify=lambda _: None
    )
    # Three cache modes and three passes leave nine local checkpoints per query.
    assert len(list(saved.with_suffix("").glob("*/*.json"))) == 18
    run = read(saved)
    assert set(run["queries"]) == {q.id for q in queries}
    assert all(value["stable"] for value in run["queries"].values())
    assert set(run["queries"][queries[0].id]["seconds"]) == {"warm", "cached", "cold"}
    assert (
        run_benchmark.run_benchmark(
            root,
            backend=backend,
            queries=queries,
            executor=lambda _: pytest.fail("Completed runs must be reused"),
            notify=lambda _: None,
        )
        == saved
    )
    report = build_report(root)
    row = report["experiments"][0]
    assert row["name"] == "baseline"
    assert row["ndcg@10"]["en"]["mean"] == 1
    result = read(root / row["result"])
    assert result["quality"]["en"]["hit@10"] == 1
    assert result["run"] == f"runs/{saved.name}"
    assert "| [baseline](results/baseline/" in (root / "EXPERIMENTS.md").read_text()
    assert str(root) not in json.dumps(report) + json.dumps(result)
    # The reference keeps the grades and what is needed to check them, nothing more.
    reference = read(suite)
    assert set(reference) == {
        "queries",
        "passage_snapshot",
        "qrels",
        "passage_isa",
        "gaps",
        "pilot",
        "status",
        "sha256",
    }
    assert reference["queries"] == {
        q.id: run["queries"][q.id]["query_sha256"] for q in queries
    }
    assert reference["gaps"] == []
    assert result["protocol"]["name"] == "passage-grades-v2"


def test_one_judge_finishes_its_queue_but_never_produces_qrels(workflow):
    root, _, _ = workflow
    result = judge.run_judge(root, "codex", caller=fake_judge, notify=lambda _: None)
    assert result["status"] == "completed"
    queue = read(root / "queue.json")["jobs"]
    assert len(list((root / "judgments/codex").glob("*.json"))) == len(queue)
    suite = read(build_suite.freeze(root))
    assert suite["status"] == "reference_incomplete"
    assert suite["pilot"]["status"] == "pending"
    assert not any(suite["qrels"].values())


def test_failed_control_blocks_the_queue(workflow):
    root, _, _ = workflow
    calls = []

    def lenient(job):
        calls.append(job["kind"])
        value = fake_judge(job)
        for grade in value["answer"]["grades"]:
            grade["grade"] = 3
            grade["quote"] = job["payload"]["passages"][0]["text"][:10]
        if job["kind"] == "control":
            for grade, passage in zip(
                value["answer"]["grades"], job["payload"]["passages"], strict=True
            ):
                grade["quote"] = passage["text"]
        return value

    result = judge.run_judge(root, "codex", caller=lenient, notify=lambda _: None)
    assert result["status"] == "failed_controls"
    assert calls == ["control"]


def test_quota_is_preserved_and_next_explicit_run_can_probe(workflow):
    root, _, _ = workflow

    def quota(job):
        raise judge.JudgeFailure("paused_quota", "usage limit")

    with pytest.raises(judge.JudgeFailure):
        judge.run_judge(root, "codex", caller=quota, notify=lambda _: None)
    assert not list((root / "judgments").rglob("*.json"))
    judge.run_judge(
        root, "codex", caller=fake_judge, max_pairs=2, notify=lambda _: None
    )
    assert list((root / "judgments/codex").glob("*.json"))


def test_judges_run_in_parallel_but_not_twice(workflow):
    from benchmarks.corpus import CorpusError
    from benchmarks.storage import locked

    root, _, _ = workflow
    with locked(root, "judge-claude"):
        result = judge.run_judge(root, "codex", caller=fake_judge, notify=print)
        assert result["status"] == "completed"
        with pytest.raises(CorpusError, match="judge-claude"):
            judge.run_judge(root, "claude", caller=fake_judge, notify=print)


def test_quote_may_leave_out_pdf_icon_glyphs():
    job = {
        "kind": "grade",
        "payload": {
            "passages": [{"passage_id": "a", "text": "must be used to_ \uf05a _ensure"}]
        },
    }

    def grade(quote):
        return {
            "grades": [
                {
                    "passage_id": "a",
                    "grade": 3,
                    "reason": "Synthetic grade",
                    "quote": quote,
                    "architecture_tags": [],
                }
            ]
        }

    build_suite.validate_answer(job, grade("used to_  _ensure"))
    with pytest.raises(build_suite.CorpusError, match="exact source quote"):
        build_suite.validate_answer(job, grade("used to ensure"))


def test_bad_quotes_exhaust_only_format_retries(workflow):
    root, _, _ = workflow
    calls = []

    def bad(job):
        calls.append(job)
        value = fake_judge(job)
        if job["kind"] in ("grade", "control"):
            value["answer"]["grades"][0]["quote"] = "not in source"
        return value

    # The control fails every retry, so the provider stops without judgments.
    result = judge.run_judge(root, "codex", caller=bad, notify=lambda _: None)
    assert result["status"] == "failed_controls"
    assert len(calls) == 3
    assert not list((root / "judgments").rglob("*.json"))


def test_timeout_is_retried(workflow):
    root, _, _ = workflow
    calls = []

    def slow_once(job):
        calls.append(job)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired("codex", 900)
        return fake_judge(job)

    result = judge.run_judge(root, "codex", caller=slow_once, notify=lambda _: None)
    assert result["status"] == "completed"
    queue = read(root / "queue.json")["jobs"]
    assert len(list((root / "judgments/codex").glob("*.json"))) == len(queue)


def test_judge_partial_batch_budget_and_resume(workflow):
    root, _, _ = workflow
    calls = []

    def record(job):
        calls.append(job)
        return fake_judge(job)

    result = judge.run_judge(
        root, "codex", caller=record, max_pairs=1, notify=lambda _: None
    )
    assert result["status"] == "paused_budget"
    assert [p["passage_id"] for p in calls[0]["payload"]["passages"]] == ["positive"]
    assert not list((root / "judgments/codex").glob("*.json"))
    judge.run_judge(root, "codex", caller=record, max_pairs=1, notify=lambda _: None)
    assert [p["passage_id"] for p in calls[1]["payload"]["passages"]] == ["negative"]
    receipt = read(next((root / "judgments/codex").glob("*.json")))
    assert {g["passage_id"] for g in receipt["answer"]["grades"]} == {
        "positive",
        "negative",
    }
    assert receipt["attempts"] == 2


def test_interrupted_judge_resumes_and_only_a_model_change_is_rejected(workflow):
    root, _, _ = workflow

    def interrupt(_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        judge.run_judge(root, "codex", caller=interrupt, notify=lambda _: None)
    states = [read(p) for p in (root / "judge-state/codex").glob("*.json")]
    assert states[0]["status"] == "interrupted"
    judge.run_judge(
        root, "codex", caller=fake_judge, max_pairs=1, notify=lambda _: None
    )

    def upgraded_cli(job):
        return fake_judge(job) | {"identity": {"cli": "newer", "synthetic": True}}

    judge.run_judge(
        root, "codex", caller=upgraded_cli, max_pairs=1, notify=lambda _: None
    )
    assert len(list((root / "judgments/codex").glob("*.json"))) == 1

    def changed(job):
        return fake_judge(job) | {"resolved_models": ["another-model"]}

    with pytest.raises(judge.JudgeFailure, match="model changed"):
        judge.run_judge(
            root, "codex", caller=changed, max_pairs=1, notify=lambda _: None
        )
    assert len(list((root / "judgments/codex").glob("*.json"))) == 1
    assert len(list((root / "judge-attempts/codex").glob("*/*.json"))) == 3


def test_failed_search_is_checkpointed_and_resumed(workflow, monkeypatch):
    root, backend, queries = workflow
    complete_judgments(root)
    monkeypatch.setattr(run_benchmark.llama_client, "stop_server", lambda *a, **k: None)

    def fail(_):
        raise RuntimeError("synthetic timeout")

    with pytest.raises(RuntimeError, match="saved output"):
        run_benchmark.run_benchmark(
            root, backend=backend, queries=queries, executor=fail, notify=lambda _: None
        )
    assert len(list((root / "runs").glob("*/*/*.json"))) == 1
    # An incomplete run has no compact file, so it has no result yet.
    assert build_report(root)["experiments"] == []
    saved = run_benchmark.run_benchmark(
        root,
        backend=backend,
        queries=queries,
        executor=lambda _: ([], {"elapsed_seconds": 0.1}),
        notify=lambda _: None,
    )
    assert len(list((saved.with_suffix("") / "failed-attempts").glob("*.json"))) == 1
    assert build_report(root)["experiments"][0]["ndcg@10"]["en"]["mean"] == 0


def test_importing_new_run_reuses_pool_queries_and_existing_labels(
    workflow, monkeypatch
):
    root, backend, queries = workflow
    complete_judgments(root)
    receipts = {p.name for p in (root / "judgments/codex").glob("*.json")}
    calls = backend.calls
    monkeypatch.setattr(run_benchmark.llama_client, "stop_server", lambda *a, **k: None)
    run_benchmark.run_benchmark(
        root,
        backend=backend,
        queries=queries,
        executor=lambda _: ([], {"elapsed_seconds": 0.1}),
        notify=lambda _: None,
    )
    build_suite.prepare_pool(
        root, backend=backend, queries=queries, notify=lambda _: None
    )
    assert backend.calls == calls
    assert receipts == {p.name for p in (root / "judgments/codex").glob("*.json")}
    judge.run_judge(
        root,
        "codex",
        caller=lambda _: pytest.fail("Compatible labels must be reused"),
        notify=lambda _: None,
    )


def test_new_run_extends_the_pool_without_rebatching_judged_pairs(
    tmp_path, monkeypatch
):
    root, backend, queries = make_workflow(tmp_path, pool_size=1)
    complete_judgments(root)
    before = read(root / "queue.json")["by_query"]
    monkeypatch.setattr(run_benchmark.llama_client, "stop_server", lambda *a, **k: None)
    run_benchmark.run_benchmark(
        root,
        backend=backend,
        queries=queries,
        executor=lambda _: (
            cli_results(backend.items.values()),
            {"elapsed_seconds": 0.1},
        ),
        notify=lambda _: None,
    )
    row = build_report(root)["experiments"][0]
    assert row["ndcg@10"]["en"]["mean"] is None
    assert row["unjudged_queries"] == len(queries)
    build_suite.prepare_pool(
        root, backend=backend, queries=queries, notify=lambda _: None
    )
    after = read(root / "queue.json")["by_query"]
    added = []
    for qid, jobs in before.items():
        assert after[qid][: len(jobs)] == jobs
        added.extend(after[qid][len(jobs) :])
    assert len(added) == len(queries)
    calls = []

    def record(job):
        calls.append(job["sha256"])
        return fake_judge(job)

    for provider in ("codex", "claude"):
        judge.run_judge(root, provider, caller=record, notify=lambda _: None)
    assert sorted(set(calls)) == sorted(added)
    report = build_report(root)
    assert report["status"] == "ready"
    assert report["experiments"][0]["ndcg@10"]["en"]["mean"] == 1


def test_new_index_keeps_judged_passages_in_the_pool(workflow):
    root, backend, queries = workflow
    complete_judgments(root)
    before = read(root / "queue.json")["by_query"]
    backend.manifest = backend.manifest | {"sha256": "rebuilt-index"}
    backend.pool_size = 1
    build_suite.prepare_pool(
        root, backend=backend, queries=queries, notify=lambda _: None
    )
    assert read(root / "queue.json")["by_query"] == before
    judge.run_judge(
        root,
        "codex",
        caller=lambda _: pytest.fail("Judged passages must stay pooled"),
        notify=lambda _: None,
    )


def test_judge_run_settings_do_not_change_jobs(workflow, monkeypatch):
    root, _, _ = workflow
    jobs = build_suite.refresh_jobs(root)
    config = build_suite.judge_config()
    monkeypatch.setattr(
        build_suite,
        "judge_config",
        lambda: config | {"timeout_seconds": 1800, "retry_limit": 5},
    )
    assert build_suite.refresh_jobs(root) == jobs


def test_real_search_worker_is_isolated_and_reports_measurements(
    workflow, monkeypatch, capsys
):
    import io

    from benchmarks import telemetry, worker
    from sova import cache, cli, config, db
    from tests.test_benchmarks_indexing import build

    root, fake, queries = workflow
    index = build(root / read(root / "preparation.json")["passages"])
    personal = root / "personal.sqlite3"
    personal.write_bytes(b"personal sentinel")
    monkeypatch.setattr(config, "DB_PATH", personal)
    monkeypatch.setattr(config, "DATA_DIR", root / "personal")
    monkeypatch.setattr(config, "_ACTIVE_PROJECT_DB_PATH", personal)
    monkeypatch.setattr(config, "_ACTIVE_PROJECT_DATA_DIR", root / "personal")
    monkeypatch.setattr(
        cli, "get_query_embedding", lambda _: [1.0] + [0.0] * (config.EMBEDDING_DIM - 1)
    )
    monkeypatch.setattr(telemetry, "model_rss", list)
    monkeypatch.setattr(cache, "_cache", None)
    request = {
        "database": str(index.parent / "index.sqlite3"),
        "cache_directory": str(root / "query-cache"),
        "ranking": {},
        "query": "useful",
        "limit": 10,
        "cache": True,
    }
    for attempt in range(2):
        monkeypatch.setattr(worker.sys, "stdin", io.StringIO(json.dumps(request)))
        worker.main()
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        results = [e["data"] for e in events if e["type"] == "result"]
        assert len(results) == 2
        assert len(fake.from_cli(queries[0], results).hits) == 2
        diagnostic = next(
            e["data"] for e in events if e["type"] == "benchmark_diagnostics"
        )
        assert diagnostic["cli_peak_rss_bytes"] > 0
        assert diagnostic["candidate_counts"]["final"] == 2
        assert ("get_vector_candidates" not in diagnostic["stage_seconds"]) == (
            attempt == 1
        )
    assert personal.read_bytes() == b"personal sentinel"
    conn = db.connect_readonly(index.parent / "index.sqlite3")
    assert conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0] == 1
    conn.close()


def test_unknown_judge_failure_does_not_block_the_job(workflow):
    root, _, _ = workflow

    def failing(job):
        raise judge.JudgeFailure(judge.classify_error("service hiccup"), "hiccup")

    with pytest.raises(judge.JudgeFailure):
        judge.run_judge(root, "codex", caller=failing, notify=lambda _: None)
    result = judge.run_judge(root, "codex", caller=fake_judge, notify=lambda _: None)
    assert result["status"] == "completed"


def test_benchmark_data_never_changes_run_identity(tmp_path):
    from benchmarks import telemetry

    def git(*args):
        subprocess.run(
            # The user's global hooks and signing must not apply to a scratch repo.
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    (tmp_path / "code.py").write_text("print(1)\n")
    data = tmp_path / "benchmarks/results/suite"
    data.mkdir(parents=True)
    (data / "judgment.json").write_text("{}")
    git("add", ".")
    git("commit", "-qm", "test: start")
    identity = telemetry.environment_identity(tmp_path)["code_sha256"]
    (data / "judgment.json").write_text('{"grade": 3}')
    (data / "new.json").write_text("{}")
    assert telemetry.environment_identity(tmp_path)["code_sha256"] == identity
    git("add", ".")
    git("commit", "-qm", "test: results")
    assert telemetry.environment_identity(tmp_path)["code_sha256"] == identity
    (tmp_path / "code.py").write_text("print(2)\n")
    assert telemetry.environment_identity(tmp_path)["code_sha256"] != identity


def test_cached_artifacts_are_revalidated_and_cannot_be_mutated_in_memory(tmp_path):
    from benchmarks.corpus import CorpusError

    path = tmp_path / "state.json"
    write(path, {"nested": {"status": "ready"}})
    value = read(path)
    value["nested"]["status"] = "corrupt"
    assert read(path)["nested"]["status"] == "ready"
    path.write_text(path.read_text().replace("ready", "wrong"))
    with pytest.raises(CorpusError, match="checksum"):
        read(path)
