from types import SimpleNamespace as NS

import pytest

from agentic_ai.config import Settings
from agentic_ai.remediation import reconciler
from agentic_ai.remediation.dedupe import find_open_parent, fingerprint, record_repeat
from agentic_ai.telemetry.run_context import RunContext, TaskError

ERROR = "ValueError: probe v2: read 10 incident rows, then failed on purpose"


def _run(error=ERROR, run_id="7464260318731", job_id="720091003927156", task="read_then_fail"):
    return RunContext(job_id=job_id, run_id=run_id, job_name="probe v2",
                      task_errors=[TaskError(task, "1", "FAILED", "notebook /x", error=error)])


class FakeSpark:
    def __init__(self, rows=None):
        self.rows, self.queries = rows or [], []

    def sql(self, q):
        self.queries.append(" ".join(q.split()))
        return NS(collect=lambda: self.rows)


class FakeGitHub:
    def __init__(self):
        self.comments = []

    def comment(self, number, body):
        self.comments.append((number, body))
        return True


# ------------------------------------------------------------------ fingerprint

def test_same_failure_on_another_run_or_recreated_job_matches():
    a = fingerprint(_run(), "agentic-ai probe v2")
    assert a == fingerprint(_run(run_id="9999999999", job_id="361322014438099"), "agentic-ai probe v2")
    first = _run(error=ERROR + " at 2026-10-07 15:49:02 (run 7464260318731, "
                       "id 3f2b1c4d-0a9e-4b7c-9d1e-2f3a4b5c6d7e)")
    later = _run(error=ERROR + " at 2026-10-08 09:01:44 (run 1111111111, "
                       "id 00000000-0000-0000-0000-000000000000)")
    assert fingerprint(first, "agentic-ai probe v2") == fingerprint(later, "agentic-ai probe v2")


def test_different_error_task_or_job_does_not_match():
    base = fingerprint(_run(), "probe v2")
    assert base != fingerprint(_run(error="KeyError: 'customer_id'"), "probe v2")
    assert base != fingerprint(_run(task="other_task"), "probe v2")
    assert base != fingerprint(_run(), "nightly orders load")


def test_no_error_text_is_never_collapsed():
    assert fingerprint(_run(error=""), "probe v2") is None
    assert fingerprint(RunContext("1", "2", "probe v2"), "probe v2") is None


# ------------------------------------------------------------------ lookup and linking

def test_find_open_parent_queries_open_incidents_only():
    spark = FakeSpark([{"incident_id": "INC-A", "occurrence_count": 3, "github_issue_number": 41}])
    fp = fingerprint(_run(), "probe v2")
    parent = find_open_parent(Settings(), spark, fp, "INC-B")
    assert parent == {"incident_id": "INC-A", "occurrence_count": 3, "github_issue_number": 41}
    q = spark.queries[0]
    assert f"i.fingerprint = '{fp}'" in q and "i.status IN ('PENDING_APPROVAL')" in q
    assert "i.incident_id <> 'INC-B'" in q


def test_find_open_parent_rejects_unsafe_input():
    spark = FakeSpark([{"incident_id": "INC-A", "occurrence_count": 1, "github_issue_number": None}])
    assert find_open_parent(Settings(), spark, "not-a-fingerprint", "INC-B") is None
    assert find_open_parent(Settings(), spark, "a" * 32, "INC'; DROP") is None
    assert spark.queries == []


def test_record_repeat_links_counts_and_comments():
    spark, gh = FakeSpark(), FakeGitHub()
    fp = fingerprint(_run(), "probe v2")
    incident = {"incident_id": "INC-B", "run_id": "222", "run_end_time": "2026-10-12 09:00:00"}
    count = record_repeat(Settings(), spark, gh, incident, fp,
                          {"incident_id": "INC-A", "occurrence_count": 3, "github_issue_number": 41})

    assert count == 4
    assert "SET status = 'DUPLICATE', duplicate_of = 'INC-A'" in spark.queries[0]
    assert "WHERE incident_id = 'INC-B'" in spark.queries[0]
    assert "occurrence_count = COALESCE(occurrence_count, 1) + 1" in spark.queries[1]
    assert "WHERE incident_id = 'INC-A'" in spark.queries[1]
    assert gh.comments[0][0] == 41 and "happened 4 times" in gh.comments[0][1]


def test_record_repeat_without_issue_sends_no_comment():
    gh = FakeGitHub()
    record_repeat(Settings(), FakeSpark(), gh, {"incident_id": "INC-B"}, "a" * 32,
                  {"incident_id": "INC-A", "occurrence_count": 1, "github_issue_number": None})
    assert gh.comments == []


# ------------------------------------------------------------------ reconciler wiring

INCIDENT = {"incident_id": "INC-B", "job_id": "11", "run_id": "22", "pipeline_name": "probe v2",
            "raw_error": "x"}


@pytest.fixture
def wired(monkeypatch):
    monkeypatch.setattr(reconciler, "build_run_context", lambda *a, **k: _run())
    calls = {}

    def fake_record(s, spark, gh, inc, fp, parent):
        calls["linked"] = parent["incident_id"]

    monkeypatch.setattr(reconciler, "record_repeat", fake_record)
    return calls


def test_repeat_is_linked_and_not_diagnosed(wired, monkeypatch):
    monkeypatch.setattr(reconciler, "find_open_parent",
                        lambda s, spark, fp, ex: {"incident_id": "INC-A", "occurrence_count": 1})
    ctx, fp, is_repeat = reconciler._check_repeat(Settings(), None, FakeGitHub(), None, INCIDENT)
    assert is_repeat and fp and wired["linked"] == "INC-A"


def test_new_failure_is_diagnosed_with_the_context_already_built(wired, monkeypatch):
    monkeypatch.setattr(reconciler, "find_open_parent", lambda *a: None)
    ctx, fp, is_repeat = reconciler._check_repeat(Settings(), None, FakeGitHub(), None, INCIDENT)
    assert not is_repeat and fp and ctx.job_name == "probe v2" and "linked" not in wired


def test_a_failing_lookup_never_blocks_diagnosis(wired, monkeypatch):
    def broken(*a):
        raise RuntimeError("[UNRESOLVED_COLUMN] fingerprint -- migrate not run yet")

    monkeypatch.setattr(reconciler, "find_open_parent", broken)
    ctx, fp, is_repeat = reconciler._check_repeat(Settings(), None, FakeGitHub(), None, INCIDENT)
    assert not is_repeat and ctx is not None


def test_record_fingerprint_is_best_effort():
    class Broken:
        def sql(self, q):
            raise RuntimeError("UNRESOLVED_COLUMN fingerprint")

    reconciler._record_fingerprint(Settings(), Broken(), "INC-B", "a" * 32, 41)    # must not raise
    spark = FakeSpark()
    reconciler._record_fingerprint(Settings(), spark, "INC-B", None, None)
    assert "fingerprint = NULL" in spark.queries[0] and "github_issue_number = NULL" in spark.queries[0]


def test_app_reads_occurrences_best_effort(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("jinja2")
    pytest.importorskip("python_multipart")
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "agentic_ai" / "app"))
    import main

    seen = {}

    def fake_query(q, params=()):
        seen["q"], seen["params"] = q, params
        return [{"incident_id": "INC-A", "occurrence_count": 4, "last_seen_at": None}]

    monkeypatch.setattr(main, "run_query", fake_query)
    assert main.load_occurrences(["INC-A", "INC-B"])["INC-A"]["occurrence_count"] == 4
    assert "IN (?, ?)" in seen["q"] and seen["params"] == ("INC-A", "INC-B")
    assert main.load_occurrences([]) == {}

    def broken(q, params=()):
        raise RuntimeError("PERMISSION_DENIED on incidents")

    monkeypatch.setattr(main, "run_query", broken)
    assert main.load_occurrences(["INC-A"]) == {}