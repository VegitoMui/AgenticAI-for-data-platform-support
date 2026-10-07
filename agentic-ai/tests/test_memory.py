from types import SimpleNamespace as NS

from agentic_ai.config import Settings
from agentic_ai.memory import store
from agentic_ai.memory.store import EMBEDDING_DIM, cosine, incident_text, recall, update_outcome
from agentic_ai.telemetry.run_context import RunContext, TaskError

COLUMN_ERROR = "[UNRESOLVED_COLUMN.WITH_SUGGESTION] name `custmer_id` cannot be resolved. SQLSTATE: 42703"


def _run(error=COLUMN_ERROR, run_id="7464260318731"):
    return RunContext(job_id="516704126536101", run_id=run_id, job_name="probe v5",
                      task_errors=[TaskError("q", "1", "FAILED", "notebook /Users/me/p", error=error)],
                      tables_from_code=["databricks_ws.agentic_ai_dev.incidents"])


def _vec(*head):
    return list(head) + [0.0] * (EMBEDDING_DIM - len(head))


class FakeWorkspace:
    def __init__(self, vec=None, error=None):
        self.vec, self.error, self.inputs = vec, error, []
        self.serving_endpoints = self

    def query(self, name, input):
        self.inputs.append((name, input))
        if self.error:
            raise self.error
        return NS(data=[NS(embedding=self.vec)])


class FakeSpark:
    def __init__(self, rows=None, error=None):
        self.rows, self.error, self.queries = rows or [], error, []

    def sql(self, query):
        self.queries.append(query)
        if self.error:
            raise self.error
        return NS(collect=lambda: self.rows)


def _row(incident_id, vec, status="APPROVED_EXECUTED"):
    return {"incident_id": incident_id, "pipeline_name": "orders", "classified_agent": "analytics",
            "diagnosis": "typo in column name", "fix_suggestion": "fix the column", "actions_json": "[]",
            "resolution_status": status, "resolution_notes": None, "embedding": vec}


def test_incident_text_drops_run_specific_noise():
    text = incident_text(_run(COLUMN_ERROR + " run 7464260318731 at 2026-10-06 06:24:56"))
    assert "7464260318731" not in text and "2026-10-06" not in text
    assert text.startswith("notebook task failed: [UNRESOLVED_COLUMN")
    assert "tables: databricks_ws.agentic_ai_dev.incidents" in text
    assert incident_text(_run(run_id="1")) == incident_text(_run(run_id="2"))


def test_cosine():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([1.0], [1.0, 0.0]) == 0.0


def test_recall_ranks_filters_and_excludes_self():
    rows = [
        _row("INC-SELF", _vec(1.0)),
        _row("INC-CLOSE", _vec(0.95, 0.31)),
        _row("INC-FAR", _vec(0.1, 1.0), status="REJECTED"),
    ]
    w, spark = FakeWorkspace(_vec(1.0)), FakeSpark(rows)
    lines, matches, vec = recall(Settings(), spark, w, _run(), exclude_incident="INC-SELF")

    assert [m["incident_id"] for m in matches] == ["INC-CLOSE"]
    assert lines[0].startswith("[similarity 0.95] orders (agent analytics)")
    assert "outcome: APPROVED_EXECUTED" in lines[0]
    assert len(vec) == EMBEDDING_DIM
    assert f"size(embedding) = {EMBEDDING_DIM}" in spark.queries[0]
    assert w.inputs[0][0] == "databricks-bge-large-en"


def test_recall_never_raises():
    assert recall(Settings(), FakeSpark(), FakeWorkspace(error=RuntimeError("no endpoint")), _run()) \
        == ([], [], None)
    lines, matches, vec = recall(Settings(), FakeSpark(error=RuntimeError("no table")),
                                 FakeWorkspace(_vec(1.0)), _run())
    assert lines == [] and matches == [] and vec is not None
    # wrong dimension is treated as a failed embedding
    assert recall(Settings(), FakeSpark(), FakeWorkspace([1.0, 2.0]), _run())[2] is None


def test_remember_rejects_unsafe_id_and_update_outcome_escapes_notes(monkeypatch):
    d = NS(agent_name="analytics", diagnosis="x", severity=3, fix_complexity="LOW", fix_suggestion="y",
           requires_human=True, actions=[])
    assert store.remember(Settings(), FakeSpark(), "INC'; DROP", "p", _run(), d) is False

    spark = FakeSpark()
    assert update_outcome(Settings(), spark, "INC-1", "REJECTED", "it's wrong") is True
    assert "resolution_status = 'REJECTED'" in spark.queries[0] and "it\\'s wrong" in spark.queries[0]
    assert update_outcome(Settings(), spark, "INC-1", "BAD STATUS") is False



def test_migrations_add_only_missing_columns():
    from agentic_ai.migrations import apply_all

    class MigSpark:
        def __init__(self, has_column):
            self.has_column, self.queries = has_column, []

        def sql(self, q):
            self.queries.append(" ".join(q.split()))

        def table(self, name):
            cols = ["incident_id", "embedding"] + (["embedding_model"] if self.has_column else [])
            return NS(columns=cols)

    fresh = MigSpark(has_column=False)
    assert apply_all(Settings(), fresh)["alters_applied"] == 1
    assert any(q.endswith("incident_memory ADD COLUMNS (embedding_model STRING)") for q in fresh.queries)
    assert not any("IF NOT EXISTS (" in q for q in fresh.queries)

    done = MigSpark(has_column=True)
    assert apply_all(Settings(), done)["alters_applied"] == 0