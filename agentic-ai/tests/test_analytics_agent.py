from types import SimpleNamespace as NS

import pytest

from agentic_ai.agents.analytics import (
    AnalyticsAgent,
    access_check,
    column_check,
    engine_suggestions,
    missing_privilege,
    query_history,
    unresolved_columns,
)
from agentic_ai.agents.base import ToolContext, diagnosis_from_result
from agentic_ai.agents.processing import error_analysis
from agentic_ai.agents.router import route
from agentic_ai.agents.trace import Trace
from agentic_ai.config import Settings
from agentic_ai.telemetry.run_context import RunContext, TaskError

TABLE = "databricks_ws.agentic_ai_dev.incidents"

COLUMN_ERROR = (
    "[UNRESOLVED_COLUMN.WITH_SUGGESTION] A column, variable, or function parameter with name "
    "`custmer_id` cannot be resolved. Did you mean one of the following? [`incident_id`, `job_id`, "
    "`status`]. SQLSTATE: 42703; line 1 pos 7;\n'Project ['custmer_id]\n"
)
COLUMN_TRACE = (
    "AnalysisException                         Traceback (most recent call last)\n"
    "File <command-123>, line 1\n"
    f'----> 1 spark.sql("SELECT custmer_id FROM {TABLE}").show()\n'
    "File /databricks/spark/python/pyspark/sql/connect/session.py:812, in SparkSession.sql\n"
    "File /site-packages/grpc/_channel.py in ssl_credentials\n"
)
ACCESS_ERROR = (
    "[INSUFFICIENT_PERMISSIONS] Insufficient privileges: User does not have SELECT on Table "
    f"'{TABLE}'. SQLSTATE: 42501"
)
TIMEOUT_ERROR = "[QUERY_TIMEOUT] Query has timed out after 3600 seconds. Statement timeout exceeded."


def _tctx(error, trace="", spark=None, w=None, tables=(TABLE,)):
    run = RunContext(job_id="11", run_id="22", job_name="nightly_report",
                     task_errors=[TaskError("query", "33", "FAILED", "notebook /x",
                                            error=error, error_trace=trace)],
                     tables_from_code=list(tables))
    return ToolContext(settings=Settings(), run=run, spark=spark, workspace_client=w)


class FakeSpark:
    """Answers each query with the rows of the first matching prefix; raises if the value is an error."""

    def __init__(self, answers):
        self.answers, self.queries = answers, []

    def sql(self, query):
        self.queries.append(query)
        for prefix, rows in self.answers.items():
            if query.startswith(prefix):
                if isinstance(rows, Exception):
                    raise rows
                return NS(collect=lambda rows=rows: rows)
        return NS(collect=lambda: [])


DESCRIBE = [{"col_name": c} for c in ("incident_id", "job_id", "customer_id", "status")] + [
    {"col_name": ""}, {"col_name": "# Partition Information"}]


# ------------------------------------------------------------------ parsing

def test_parsers():
    assert unresolved_columns(COLUMN_ERROR) == ["custmer_id"]
    assert unresolved_columns("with name `i`.`custmer_id` cannot be resolved") == ["custmer_id"]
    assert unresolved_columns("cannot resolve '`amout`' given input columns") == ["amout"]
    assert engine_suggestions(COLUMN_ERROR) == ["incident_id", "job_id", "status"]
    assert missing_privilege(ACCESS_ERROR) == {"privilege": "SELECT", "object_type": "TABLE",
                                               "object": TABLE}


def test_error_analysis_keeps_error_subclass():
    _, data = error_analysis(_tctx(COLUMN_ERROR, COLUMN_TRACE))
    assert data["query"]["error_class"] == "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    assert data["query"]["sqlstate"] == "42703"


# ------------------------------------------------------------------ column_check

def test_column_check_finds_closest_real_column():
    spark = FakeSpark({"DESCRIBE TABLE": DESCRIBE})
    summary, data = column_check(_tctx(COLUMN_ERROR, spark=spark))
    assert "custmer_id NOT FOUND among 4 columns" in summary and "closest: customer_id" in summary
    assert "engine suggested: incident_id" in summary
    assert data["tables"][TABLE]["matches"]["custmer_id"]["closest"][0] == "customer_id"
    assert spark.queries == ["DESCRIBE TABLE `databricks_ws`.`agentic_ai_dev`.`incidents`"]


def test_column_check_without_tables_or_column():
    summary, _ = column_check(_tctx(COLUMN_ERROR, tables=()))
    assert "no tables identified" in summary and "engine suggested" in summary
    assert "No unresolved column" in column_check(_tctx("ValueError: boom"))[0]


# ------------------------------------------------------------------ query_history

def test_query_history_empty_is_reported_not_failed():
    spark = FakeSpark({"SELECT * FROM system.query.history": []})
    summary, data = query_history(_tctx(COLUMN_ERROR, spark=spark))
    assert "not evidence that the queries were healthy" in summary and data["statements"] == 0
    assert "query_source.job_info.job_id = '11'" in spark.queries[0]


def test_query_history_flags_failures_full_scans_and_spill():
    rows = [
        {"statement_id": "a", "execution_status": "FAILED", "error_message": "[UNRESOLVED_COLUMN] x",
         "statement_text": "SELECT custmer_id FROM t", "total_duration_ms": 120,
         "query_source": {"job_info": {"job_id": "11", "job_run_id": "22"}}},
        {"statement_id": "b", "execution_status": "FINISHED", "statement_text": "SELECT * FROM big",
         "total_duration_ms": 90_000, "read_files": 400, "pruned_files": 0, "read_bytes": 5 << 30,
         "spilled_local_bytes": 2 << 30, "query_source": {"job_info": {"job_id": "11", "job_run_id": "9"}}},
    ]
    summary, data = query_history(_tctx(COLUMN_ERROR, spark=FakeSpark({"SELECT": rows})))
    assert "2 statements" in summary and "(1 from this run), 1 failed" in summary
    assert "FAILED: [UNRESOLVED_COLUMN] x" in summary
    assert "FULL SCAN" in summary and "SPILLED" in summary
    assert data["full_scans"] == 1 and data["spills"] == 1


def test_query_history_rejects_non_numeric_job_id():
    t = _tctx(COLUMN_ERROR, spark=FakeSpark({}))
    t.run.job_id = "1' OR '1'='1"
    assert "No valid job id" in query_history(t)[0] and t.spark.queries == []


# ------------------------------------------------------------------ access_check

class FakeJobs:
    def __init__(self, run):
        self.run = run

    def get_run(self, run_id):
        return self.run


def test_access_check_reports_identity_and_direct_grants():
    w = NS(jobs=FakeJobs(NS(run_as_user_name="etl@corp.com", creator_user_name="me@corp.com")))
    spark = FakeSpark({
        "SHOW GRANTS ON TABLE": [{"Principal": "analysts", "ActionType": "SELECT"}],
        "SHOW GRANTS ON SCHEMA": [{"Principal": "etl@corp.com", "ActionType": "USE SCHEMA"}],
        "SHOW GRANTS ON CATALOG": RuntimeError("PERMISSION_DENIED: cannot show grants"),
    })
    summary, data = access_check(_tctx(ACCESS_ERROR, spark=spark, w=w))
    assert "run executed as etl@corp.com" in summary
    assert "missing SELECT on table " + TABLE in summary
    assert f"table {TABLE}: identity holds directly: nothing" in summary
    assert "schema databricks_ws.agentic_ai_dev: identity holds directly: USE SCHEMA" in summary
    assert "catalog databricks_ws: grants not visible" in summary
    assert data["identity"] == "etl@corp.com"


def test_access_check_falls_back_to_creator():
    w = NS(jobs=FakeJobs(NS(run_as_user_name=None, creator_user_name="me@corp.com")))
    summary, _ = access_check(_tctx(ACCESS_ERROR, spark=FakeSpark({}), w=w))
    assert "run executed as me@corp.com" in summary


# ------------------------------------------------------------------ selection, routing, safety

@pytest.mark.parametrize("error,expected", [
    (COLUMN_ERROR, "QUERY_ERRORS"),
    (TIMEOUT_ERROR, "QUERY_PERFORMANCE"),
    (ACCESS_ERROR, "ACCESS_CONTROL"),
])
def test_subagent_selection(error, expected):
    sub, _ = AnalyticsAgent().select_subagent(_tctx(error).run)
    assert sub.name == expected


def test_unresolved_column_routes_to_analytics_by_rules():
    t = _tctx(COLUMN_ERROR, COLUMN_TRACE)
    r = route(t.run, None, Trace("INC", "p"))
    assert r.agent == "analytics" and r.method == "rules"
    assert r.scores["ingestion"] == 0


def test_grant_action_always_requires_human():
    result = NS(ok=True, provider="fake", parsed={
        "diagnosis": "missing SELECT", "fix_complexity": "LOW", "confidence": 0.95,
        "requires_human": False, "actions": [f"GRANT SELECT ON TABLE {TABLE} TO `etl@corp.com`"]})
    assert diagnosis_from_result(result, "analytics").requires_human is True

    result.parsed["actions"] = [f"ANALYZE TABLE {TABLE} COMPUTE STATISTICS"]
    assert diagnosis_from_result(result, "analytics").requires_human is False