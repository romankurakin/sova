"""Public questions are portable, source-pinned and independent of local drafts."""

import json

import pytest

from benchmarks.corpus import CorpusError
from benchmarks.queries import (
    DEFAULT_PROBES,
    DEFAULT_QUERIES,
    QUERY_DIR,
    load_evaluation_queries,
    load_query_banks,
    questions_markdown,
)


def test_bundled_questions_and_safe_judge_input():
    core, probes = load_query_banks()
    assert len(core.queries) == 187
    assert len({q.comparison_group for q in core.queries}) == 99
    assert len(probes.queries) == 20
    assert {q.language for q in probes.queries} == {"ru", "zh"}
    for query in core.queries + probes.queries:
        assert query.judge_input() == {
            "id": query.id,
            "text": query.text,
            "intent": query.intent,
        }


@pytest.mark.parametrize(
    "problem",
    [
        "excerpt",
        "page",
        "hash",
        "duplicate",
        "lock",
        "anchor",
        "anchor_intent",
        "anchor_category",
    ],
)
def test_reject_unportable_or_stale_inputs(tmp_path, problem):
    core = json.loads(DEFAULT_QUERIES.read_bytes())
    probes = json.loads(DEFAULT_PROBES.read_bytes())
    if problem == "excerpt":
        core["queries"][0]["source_refs"][0]["quote"] = "Synthetic corpus excerpt"
    elif problem == "page":
        core["queries"][0]["source_refs"][0]["pdf_page"] = 100000
    elif problem == "hash":
        core["queries"][0]["source_refs"][0]["source_sha256"] = "0" * 64
    elif problem == "duplicate":
        core["queries"].append(core["queries"][0])
    elif problem == "lock":
        core["source_lock_sha256"] = "0" * 64
    elif problem == "anchor":
        probes["queries"][0]["anchor_text_sha256"] = "0" * 64
    elif problem == "anchor_intent":
        probes["queries"][0]["intent"] = "A different information need."
    elif problem == "anchor_category":
        probes["queries"][0]["category"] = "conceptual"
    core_path, probes_path = tmp_path / "core.json", tmp_path / "probes.json"
    core_path.write_text(json.dumps(core))
    probes_path.write_text(json.dumps(probes))
    with pytest.raises(CorpusError, match="Invalid query bank"):
        load_query_banks(core_path=core_path, probes_path=probes_path)


def test_evaluation_uses_both_repository_banks():
    queries = load_evaluation_queries()
    assert len(queries) == 207
    assert sorted({query.language for query in queries}) == ["en", "ru", "zh"]
    core, probes = load_query_banks()
    assert [query.id for query in queries] == [
        q.id for q in (*core.queries, *probes.queries)
    ]


def test_reading_list_matches_current_banks():
    core, probes = load_query_banks()
    rendered = questions_markdown(core, probes=probes)
    assert (QUERY_DIR / "questions.md").read_text() == rendered
    for query in core.queries:
        assert rendered.count(f"### {query.id} — {query.text}\n") == 1
    for probe in probes.queries:
        anchor_heading = f"### {probe.anchor_id} — "
        anchor_section = rendered.split(anchor_heading, 1)[1].split("\n### ", 1)[0]
        assert rendered.count(f"#### {probe.id} (") == 1
        assert probe.text in anchor_section
    assert "translations included by default" in rendered
    assert "disabled pending" not in rendered


def test_all_languages_are_enabled():
    queries = load_evaluation_queries()
    assert len(queries) == len({query.id for query in queries}) == 207
    assert {
        language: sum(query.language == language for query in queries)
        for language in ("en", "ru", "zh")
    } == {"en": 187, "ru": 10, "zh": 10}
