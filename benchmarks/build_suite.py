"""Make the candidate pool, the grade jobs, the pilot audit and reference revisions."""

import json
from functools import lru_cache
from pathlib import Path

from benchmarks.corpus import CorpusError
from benchmarks.judging_schema import INSTRUCTION, SCHEMAS, SYSTEM
from benchmarks.passages import _hash
from benchmarks.queries import Query, load_evaluation_queries
from benchmarks.search_interface import SovaBackend
from benchmarks.storage import locked, read, write

BATCH_SIZE = 4
CONTROL_GRADES = {"positive": 3, "negative": 0}
# The pilot contains the first eight pairs of each pilot query.
PILOT_PAIRS = 8
CONTROL = {
    "query": {
        "id": "control",
        "text": "In synthetic ISA A, what does register X enable?",
        "intent": "Effect of X in ISA A.",
    },
    "passages": [
        {
            "passage_id": "positive",
            "text": "In synthetic ISA A, register X enables timer T.",
            "source_id": "synthetic",
            "section_path": "",
        },
        {
            "passage_id": "negative",
            "text": "In synthetic ISA B, register X selects a color.",
            "source_id": "synthetic",
            "section_path": "",
        },
    ],
}


@lru_cache(maxsize=8)
def _config_at(stamp):
    return json.loads(Path(__file__).with_name("judges.json").read_bytes())


def judge_config():
    st = Path(__file__).with_name("judges.json").stat()
    return _config_at((st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns))


def judged_by():
    """Return the model and effort of each judge. Timeouts and retries are not here."""
    config = judge_config()
    return {provider: config[provider] for provider in ("codex", "claude")}


@lru_cache(maxsize=8)
def _schema(kind):
    return SCHEMAS[kind].model_json_schema()


_VERIFIED_JOBS: dict[Path, tuple] = {}


def pilot_ids(queries):
    # The selection does not use results. It has six English queries and three
    # queries for each other language.
    return {
        q.id
        for lang, n in [("en", 6), ("ru", 3), ("zh", 3)]
        for q in sorted(
            (q for q in queries if q.language == lang), key=lambda q: _hash(q.id)
        )[:n]
    }


def _request(kind, payload, pilot):
    # The job key contains all that the judge sees and the judge configuration.
    # It does not contain the adapter code.
    return {
        "schema_version": 3,
        "kind": kind,
        "payload": payload,
        "pilot": pilot,
        "judge_config": judged_by(),
        "schema": _schema(kind),
        "system_sha256": _hash(SYSTEM),
        "instruction_sha256": _hash(INSTRUCTION),
    }


def add_job(root, kind, payload, *, pilot=False):
    request = _request(kind, payload, pilot)
    job_id = _hash(request)
    path = root / "jobs" / f"{job_id}.json"
    if path.exists():
        st = path.stat()
        stamp = (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        if _VERIFIED_JOBS.get(path) == stamp:
            return job_id
    write(path, request, immutable=True)
    st = path.stat()
    _VERIFIED_JOBS[path] = (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    return job_id


def answer(root, job_id, provider):
    path = root / "judgments" / provider / f"{job_id}.json"
    return read(path) if path.exists() else None


def prepare_pool(root, *, backend=None, queries=None, notify=print):
    state = read(root / "preparation.json")
    queries = queries if queries is not None else load_evaluation_queries()
    own = backend is None
    if own:
        backend = SovaBackend(root / state["index"], root / state["passages"])
    snapshot = backend.snapshot["snapshot_sha256"]
    imported: dict[str, set[str]] = {}
    imported_hashes = []
    query_hashes = {q.id: _hash(q.judge_input()) for q in queries}
    for run_path in sorted((root / "runs").glob("*.json")):
        run = read(run_path)
        # Grades refer to passage IDs. Thus only runs on this snapshot add to its pool.
        if run["identity"]["passage_snapshot"] != snapshot:
            continue
        imported_hashes.append(run["sha256"])
        for qid, value in run["queries"].items():
            if query_hashes.get(qid) == value["query_sha256"]:
                imported.setdefault(qid, set()).update(value["hits"][:10])
    # A pool does not become smaller. Passages from earlier pools keep their grades
    # when the rankings change.
    carried: dict[str, set[str]] = {}
    for path in sorted((root / "pools").glob("*/manifest.json")):
        earlier = read(path)
        for qid in earlier["queries"]:
            record = read(path.parent / "queries" / f"{qid}.json")
            carried.setdefault(qid, set()).update(
                p["passage_id"]
                for p in record["passages"]
                if p["passage_id"] in backend.items
            )
    prior = None
    if (root / "pool.json").exists():
        prior_manifest, prior_records = pool_records(root)
        if prior_manifest["inputs"]["index"] == backend.manifest["sha256"]:
            prior = {record["query"]["id"]: record for record in prior_records}
    inputs = {
        "passages": snapshot,
        "index": backend.manifest["sha256"],
        "queries": [q.model_dump() for q in queries],
        "pool_protocol": "cumulative-union-10-passages-v3",
        "imported_runs": imported_hashes,
    }
    directory = root / "pools" / _hash(inputs)
    try:
        with locked(root, "pool"):
            selected_pilot = pilot_ids(queries)
            ids = []
            for query in queries:
                target = directory / "queries" / f"{query.id}.json"
                if target.exists():
                    read(target)
                    ids.append(query.id)
                    continue
                cached = prior.get(query.id) if prior else None
                if cached and cached["query"] == query.model_dump():
                    rankings = cached["rankings"]
                else:
                    rankings = {}
                    emb = backend.query_embedding(query.text)
                    for method in ("bm25", "dense", "hybrid"):
                        rankings[method] = backend.retrieve(
                            query.text, method=method, limit=10, embedding=emb
                        )
                pool = {pid for hits in rankings.values() for pid in hits}
                pool |= imported.get(query.id, set()) | carried.get(query.id, set())
                record = {
                    "query": query.model_dump(),
                    "passages": [
                        {
                            k: backend.items[pid][k]
                            for k in ("passage_id", "text", "source_id", "section_path")
                        }
                        for pid in sorted(pool)
                    ],
                    "rankings": rankings,
                    "pilot": query.id in selected_pilot,
                }
                write(target, record, immutable=True)
                ids.append(query.id)
                notify(f"Pool: {len(ids)}/{len(queries)} queries")
            manifest = write(
                directory / "manifest.json",
                {"inputs": inputs, "queries": ids},
                immutable=True,
            )
            write(
                root / "pool.json",
                {
                    "path": str((directory / "manifest.json").relative_to(root)),
                    "sha256_pool": manifest["sha256"],
                },
            )
    finally:
        if own:
            backend.close()
    refresh_jobs(root)
    return directory / "manifest.json"


def pool_records(root):
    pointer = read(root / "pool.json")
    path = root / pointer["path"]
    manifest = read(path)
    if manifest["sha256"] != pointer["sha256_pool"]:
        raise CorpusError("Pool pointer checksum mismatch")
    return manifest, [
        read(path.parent / "queries" / f"{qid}.json") for qid in manifest["queries"]
    ]


def refresh_jobs(root):
    if not (root / "pool.json").exists():
        return []
    _manifest, records = pool_records(root)
    queue_path = root / "queue.json"
    previous = read(queue_path).get("by_query", {}) if queue_path.exists() else {}
    by_query = {}
    for record in records:
        query = {k: record["query"][k] for k in ("id", "text", "intent")}
        passages = {p["passage_id"]: p for p in record["passages"]}
        jobs, covered = [], set()
        # Batches do not change when the pool becomes larger, so their judgments
        # stay valid. Only new passages go into new batches.
        for jid in previous.get(query["id"], []):
            path = root / "jobs" / f"{jid}.json"
            if not path.exists():
                continue
            job = read(path)
            payload = job["payload"]
            if (
                job["kind"] == "grade"
                and payload.get("query") == query
                and all(passages.get(p["passage_id"]) == p for p in payload["passages"])
                and _hash(_request("grade", payload, job["pilot"])) == jid
            ):
                jobs.append(jid)
                covered.update(p["passage_id"] for p in payload["passages"])
        first = not jobs
        remaining = [p for pid, p in sorted(passages.items()) if pid not in covered]
        for i in range(0, len(remaining), BATCH_SIZE):
            payload = {"query": query, "passages": remaining[i : i + BATCH_SIZE]}
            pilot = record["pilot"] and first and i < PILOT_PAIRS
            jobs.append(add_job(root, "grade", payload, pilot=pilot))
        by_query[query["id"]] = jobs
    control = add_job(root, "control", CONTROL, pilot=True)
    jobs = [jid for record in records for jid in by_query[record["query"]["id"]]]
    jobs.append(control)
    write(queue_path, {"jobs": jobs, "by_query": by_query})
    return jobs


def _visible(text):
    # PDF icon glyphs are private-use characters. A judge cannot copy them.
    return "".join(c for c in text if not 0xE000 <= ord(c) <= 0xF8FF)


def validate_answer(job, data):
    result = SCHEMAS[job["kind"]].model_validate(data).model_dump()
    passages = {p["passage_id"]: p for p in job["payload"]["passages"]}
    ids = [g["passage_id"] for g in result["grades"]]
    if sorted(ids) != sorted(passages):
        raise CorpusError("Grade answer must include every passage exactly once")
    for grade in result["grades"]:
        if grade["grade"] and (
            not grade["quote"].strip()
            or _visible(grade["quote"])
            not in _visible(passages[grade["passage_id"]]["text"])
        ):
            raise CorpusError("Nonzero grade needs an exact source quote")
    return result


def pilot_audit(root, jobs):
    """Return the control results and the judge agreement on the pilot pairs."""
    pending, errors, differences, agreements = [], [], [], []
    for job in jobs:
        if not job["pilot"]:
            continue
        grades = {}
        for provider in ("codex", "claude"):
            response = answer(root, job["sha256"], provider)
            if response is not None:
                grades[provider] = {
                    g["passage_id"]: g["grade"] for g in response["answer"]["grades"]
                }
        if job["kind"] == "control":
            errors.extend(
                f"{provider}: synthetic control failed"
                for provider, value in grades.items()
                if value != CONTROL_GRADES
            )
        if len(grades) < 2:
            pending.append(job["sha256"])
            continue
        if job["kind"] == "grade":
            left, right = grades["codex"], grades["claude"]
            for pid in left:
                differences.append(abs(left[pid] - right[pid]))
                agreements.append((left[pid] >= 2) == (right[pid] >= 2))
    return {
        "status": "failed_controls" if errors else "pending" if pending else "passed",
        "pending_jobs": len(pending),
        "errors": errors,
        "pairs": len(differences),
        "mean_judge_grade_difference": sum(differences) / len(differences)
        if differences
        else None,
        "useful_grade_agreement": sum(agreements) / len(agreements)
        if agreements
        else None,
    }


def freeze(root):
    manifest, records = pool_records(root)
    jobs = [read(root / "jobs" / f"{jid}.json") for jid in refresh_jobs(root)]
    by_query: dict[str, list[dict]] = {}
    for job in jobs:
        by_query.setdefault(job["payload"]["query"]["id"], []).append(job)
    queries, qrels, isa, gaps = {}, {}, {}, set()
    for record in records:
        qid = record["query"]["id"]
        # A grade is valid only for the text and intent that the judges saw.
        queries[qid] = _hash(
            Query.model_validate_json(json.dumps(record["query"])).judge_input()
        )
        qrels[qid], isa[qid] = {}, {}
        for job in by_query.get(qid, []):
            responses = [
                answer(root, job["sha256"], provider)
                for provider in ("codex", "claude")
            ]
            if any(response is None for response in responses):
                gaps.add(qid)
                continue
            left, right = [response["answer"] for response in responses]
            rg = {g["passage_id"]: g for g in right["grades"]}
            for grade in left["grades"]:
                other = rg[grade["passage_id"]]
                tags = set(grade["architecture_tags"])
                isa[qid][grade["passage_id"]] = (
                    sorted(tags) if tags == set(other["architecture_tags"]) else None
                )
                qrels[qid][grade["passage_id"]] = (grade["grade"] + other["grade"]) / 2
    pilot = pilot_audit(root, jobs)
    value = {
        "queries": queries,
        "passage_snapshot": manifest["inputs"]["passages"],
        "qrels": qrels,
        "passage_isa": isa,
        # Queries that still wait for a grade from one of the judges.
        "gaps": sorted(gaps),
        "pilot": pilot,
        "status": "ready"
        if not gaps and pilot["status"] == "passed"
        else "reference_incomplete",
    }
    # Each revision stays as history. suite.json refers to the current revision.
    revision = _hash(value)
    path = root / "suites" / revision / "manifest.json"
    write(path, value, immutable=True)
    write(
        root / "suite.json", {"path": str(path.relative_to(root)), "revision": revision}
    )
    return path
