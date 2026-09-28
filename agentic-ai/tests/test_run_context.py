from types import SimpleNamespace as NS

from agentic_ai.config import Settings
from agentic_ai.telemetry.run_context import (
    MAX_TRACE_CHARS,
    build_run_context,
    extract_tables_from_code,
)


class FakeJobs:
    def __init__(self, run, outputs):
        self._run = run
        self._outputs = outputs

    def get_run(self, run_id):
        return self._run

    def get_run_output(self, run_id):
        return self._outputs[run_id]


class FakeSpark:
    """Returns rows for the exact-run lineage query only when told to."""

    def __init__(self, run_rows, recent_rows):
        self.run_rows, self.recent_rows = run_rows, recent_rows

    def sql(self, query):
        rows = self.run_rows if "entity_run_id" in query else self.recent_rows
        return NS(collect=lambda: rows)


def _run():
    failed = NS(
        task_key="load", run_id=11,
        state=NS(result_state=NS(value="FAILED"), life_cycle_state=NS(value="TERMINATED"),
                 state_message="Workload failed"),
        notebook_task=NS(notebook_path="/Repos/etl/load"),
    )
    ok = NS(
        task_key="prep", run_id=10,
        state=NS(result_state=NS(value="SUCCESS"), life_cycle_state=NS(value="TERMINATED"),
                 state_message=""),
        notebook_task=NS(notebook_path="/Repos/etl/prep"),
    )
    return NS(run_name="orders_etl", tasks=[ok, failed])


INCIDENT = {"job_id": "42", "run_id": "7", "pipeline_name": "orders_etl"}


def test_collects_failed_task_error_and_trace_tail():
    trace = "x" * (MAX_TRACE_CHARS + 500) + "\nRuntimeError: boom"
    w = NS(jobs=FakeJobs(_run(), {11: NS(error="RuntimeError: boom", error_trace=trace)}))
    ctx = build_run_context(Settings(), INCIDENT, workspace_client=w, spark=FakeSpark([], []))

    assert len(ctx.task_errors) == 1
    t = ctx.task_errors[0]
    assert t.task_key == "load" and t.error == "RuntimeError: boom"
    assert t.task_type == "notebook /Repos/etl/load"
    assert t.error_trace.endswith("RuntimeError: boom")  # tail kept
    assert ctx.primary_error == "RuntimeError: boom"


def test_lineage_prefers_this_run_and_drops_system_tables():
    rows = [
        {"src": "main.sales.orders", "tgt": "main.sales.orders_clean"},
        {"src": "system.access.audit", "tgt": None},
    ]
    w = NS(jobs=FakeJobs(_run(), {11: NS(error="", error_trace="")}))
    ctx = build_run_context(Settings(), INCIDENT, workspace_client=w, spark=FakeSpark(rows, []))

    assert ctx.lineage_scope == "this_run"
    assert ctx.tables_read == ["main.sales.orders"]
    assert ctx.tables_written == ["main.sales.orders_clean"]


def test_lineage_falls_back_to_recent_runs():
    recent = [{"src": "main.sales.orders", "tgt": None}]
    w = NS(jobs=FakeJobs(_run(), {11: NS(error="", error_trace="")}))
    ctx = build_run_context(Settings(), INCIDENT, workspace_client=w, spark=FakeSpark([], recent))

    assert ctx.lineage_scope == "recent_runs"
    assert "recent runs of this job" in ctx.to_prompt()


def test_retried_task_reported_once_with_latest_attempt_and_ansi_stripped():
    run = _run()
    first = run.tasks[1]
    retry = NS(task_key="load", run_id=12, attempt_number=1, state=first.state,
               notebook_task=first.notebook_task)
    first.attempt_number = 0
    run.tasks.append(retry)
    colored = "\x1b[0;31mValueError\x1b[0m: boom"
    w = NS(jobs=FakeJobs(run, {12: NS(error=colored, error_trace=colored)}))
    ctx = build_run_context(Settings(), INCIDENT, workspace_client=w, spark=FakeSpark([], []))

    assert len(ctx.task_errors) == 1
    t = ctx.task_errors[0]
    assert t.task_run_id == "12" and t.attempts == 2
    assert t.error == "ValueError: boom"
    assert "\x1b" not in t.error_trace
    assert "failed on all 2 attempts" in ctx.to_prompt()


def test_tables_extracted_from_failing_code():
    trace = (
        "1 n = spark.table('databricks_ws.agentic_ai_dev.incidents').count()\n"
        "spark.sql('SELECT * FROM main.sales.orders o JOIN `main`.sales.customers c')\n"
        "File /databricks/python/lib/pyspark.sql.utils.py\n"
        "conf 'spark.sql.shuffle.partitions' is invalid"
    )
    assert extract_tables_from_code(trace) == [
        "databricks_ws.agentic_ai_dev.incidents",
        "main.sales.orders",
    ]


def test_code_tables_reported_when_lineage_empty():
    trace = "spark.table('databricks_ws.agentic_ai_dev.incidents')"
    w = NS(jobs=FakeJobs(_run(), {11: NS(error="ValueError: x", error_trace=trace)}))
    ctx = build_run_context(Settings(), INCIDENT, workspace_client=w, spark=FakeSpark([], []))

    assert ctx.lineage_scope == "none"
    assert ctx.tables == ["databricks_ws.agentic_ai_dev.incidents"]
    assert "Tables referenced in the failing code" in ctx.to_prompt()