from types import SimpleNamespace as NS

import pytest

from agentic_ai.agents import orchestrator
from agentic_ai.agents.base import diagnosis_from_result, needs_person
from agentic_ai.config import Settings, settings_from_args
from agentic_ai.maintenance import scanner
from agentic_ai.maintenance.scanner import SMALL_FILES, small_files
from agentic_ai.telemetry.run_context import TABLE_SCAN_SOURCE, build_run_context

MB = 1024 * 1024
TABLE = "databricks_ws.agentic_probe.orders_small_files"


# ------------------------------------------------------------------ rule

def test_small_files_rule_thresholds():
    hit = small_files(TABLE, {"numFiles": 1000, "sizeInBytes": 400 * MB})
    assert hit and hit.rule == SMALL_FILES and hit.key == f"SMALL_FILES:{TABLE}"
    assert "1,000 files averaging 0.4 MB" in hit.summary and "OPTIMIZE" in hit.summary

    assert small_files(TABLE, {"numFiles": 1000, "sizeInBytes": 100 * MB}) is None   # table too small
    assert small_files(TABLE, {"numFiles": 50, "sizeInBytes": 400 * MB}) is None     # too few files
    assert small_files(TABLE, {"numFiles": 200, "sizeInBytes": 20_000 * MB}) is None  # files big enough


# ------------------------------------------------------------------ scanning

class FakeSpark:
    def __init__(self, tables, details, suppressed=()):
        self.tables, self.details, self.suppressed, self.queries = tables, details, suppressed, []

    def sql(self, q):
        self.queries.append(" ".join(q.split()))
        if q.startswith("SHOW TABLES"):
            rows = [{"tableName": t, "isTemporary": False} for t in self.tables]
        elif q.startswith("DESCRIBE HISTORY"):
            rows = []
        elif q.startswith("DESCRIBE DETAIL"):
            name = q.split("`.`")[-1].strip("` ")
            if name not in self.details:
                raise RuntimeError("[EXPECT_TABLE_NOT_VIEW] is a view")
            rows = [self.details[name]]
        else:
            rows = [{"task_key": k} for k in self.suppressed]
        return NS(collect=lambda: rows)


def _settings():
    return Settings(catalog="databricks_ws", schema="agentic_ai_dev",
                    scan_schemas=("databricks_ws.agentic_probe",))


def test_scan_raises_new_findings_and_suppresses_open_ones(monkeypatch):
    written = []
    monkeypatch.setattr(scanner, "_write_incident", lambda s, spark, f: written.append(f) or "INC-1")
    details = {
        "orders_small_files": {"format": "delta", "numFiles": 1000, "sizeInBytes": 400 * MB},
        "customers": {"format": "delta", "numFiles": 4, "sizeInBytes": 300 * MB},
        "events_csv": {"format": "csv", "numFiles": 5000, "sizeInBytes": 900 * MB},
    }
    spark = FakeSpark(["orders_small_files", "customers", "events_csv", "a_view"], details)

    stats = scanner.run_once(_settings(), spark)
    assert stats == {"tables": 4, "findings": 1, "raised": 1, "suppressed": 0}
    assert written[0].table == TABLE

    already = FakeSpark(["orders_small_files"], details, suppressed=[f"SMALL_FILES:{TABLE}"])
    assert scanner.run_once(_settings(), already)["suppressed"] == 1


def test_suppression_covers_open_and_recently_auto_fixed():
    spark = FakeSpark([], {})
    scanner.suppressed_keys(_settings(), spark)
    q = spark.queries[0]
    assert f"detection_source = '{TABLE_SCAN_SOURCE}'" in q
    assert "status IN ('NEW', 'PENDING_APPROVAL')" in q and "INTERVAL 24 HOURS" in q


def test_no_schemas_configured_scans_nothing():
    assert scanner.run_once(Settings(), FakeSpark([], {}))["tables"] == 0


def test_scan_schemas_from_args():
    s = settings_from_args(["--scan-schemas", "databricks_ws.agentic_probe, databricks_ws.sales"])
    assert s.scan_schemas == ("databricks_ws.agentic_probe", "databricks_ws.sales")


# ------------------------------------------------------------------ run context and routing

INCIDENT = {"incident_id": "INC-1", "pipeline_name": f"table maintenance: {TABLE}",
            "task_key": f"SMALL_FILES:{TABLE}", "raw_error": "SMALL FILES on ...: 1,000 files",
            "detection_source": TABLE_SCAN_SOURCE}


def test_maintenance_context_needs_no_job_run():
    ctx = build_run_context(Settings(), INCIDENT)            # no workspace client, no Spark
    assert ctx.tables_written == [TABLE] and ctx.tables == [TABLE]
    assert ctx.task_errors[0].task_key == "SMALL_FILES" and "1,000 files" in ctx.task_errors[0].error
    assert "table scanner" in ctx.notes[0]


def test_scan_incident_goes_straight_to_storage(monkeypatch):
    seen = {}

    class FakeAgent:
        def run(self, llm, tctx, trace, memory):
            seen["agent_ran"] = True
            return NS(agent_name="storage")

    def no_router(*a, **k):
        raise AssertionError("the router must not be called for scanner findings")

    monkeypatch.setattr(orchestrator, "route", no_router)
    monkeypatch.setitem(orchestrator.AGENTS, "storage", FakeAgent())
    monkeypatch.setattr(orchestrator.store, "recall", lambda *a, **k: ([], [], None))
    monkeypatch.setattr(orchestrator.Trace, "flush", lambda self, s, spark: True)

    out = orchestrator.diagnose_incident(Settings(), None, None, None, INCIDENT)
    assert seen["agent_ran"] and out.route.agent == "storage" and out.route.method == "scan"
    assert out.trace.steps[0].node_name == "classify"


# ------------------------------------------------------------------ safety guards

@pytest.mark.parametrize("action,expected", [
    (f"OPTIMIZE {TABLE}", False),
    (f"OPTIMIZE {TABLE} ZORDER BY (order_date)", False),
    (f"ANALYZE TABLE {TABLE} COMPUTE STATISTICS", False),
    (f"VACUUM {TABLE}", True),                     # retention comes from the table: unknown
    (f"VACUUM {TABLE} DRY RUN", False),                          # deletes nothing
    (f"VACUUM {TABLE} RETAIN 168 HOURS", False),
    (f"VACUUM {TABLE} RETAIN 24 HOURS", True),
    (f"VACUUM {TABLE} RETAIN 0 HOURS", True),
    (f"DROP TABLE {TABLE}", True),
    (f"TRUNCATE TABLE {TABLE}", True),
    (f"DELETE FROM {TABLE} WHERE 1=1", True),
    (f"RESTORE TABLE {TABLE} TO VERSION AS OF 3", True),
    (f"ALTER TABLE {TABLE} DROP COLUMN order_date", True),
    (f"ALTER TABLE {TABLE} SET TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true')", False),
    (f"GRANT SELECT ON TABLE {TABLE} TO `x`", True),
])
def test_needs_person(action, expected):
    assert needs_person(action) is expected


def test_destructive_action_overrides_a_confident_llm():
    result = NS(ok=True, provider="fake", parsed={
        "diagnosis": "x", "fix_complexity": "LOW", "confidence": 0.99, "requires_human": False,
        "actions": [f"OPTIMIZE {TABLE}", f"VACUUM {TABLE} RETAIN 1 HOURS"]})
    assert diagnosis_from_result(result, "storage").requires_human is True

    result.parsed["actions"] = [f"OPTIMIZE {TABLE}"]
    assert diagnosis_from_result(result, "storage").requires_human is False


# ------------------------------------------------------------------ VACUUM_OVERDUE

def _history(*commits):
    """commits: (version, operation, hours_ago, metrics)"""
    from datetime import datetime, timedelta, timezone

    now = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
    rows = [{"version": v, "operation": op, "timestamp": now - timedelta(hours=h), "operationMetrics": m}
            for v, op, h, m in commits]
    return rows, now


def test_vacuum_overdue_counts_only_removals_older_than_168_hours():
    from agentic_ai.maintenance.scanner import VACUUM_OVERDUE, vacuum_overdue

    rows, now = _history(
        (2, "OPTIMIZE", 170, {"numRemovedFiles": "1000", "numRemovedBytes": str(500 * MB)}),
        (1, "WRITE", 171, {"numFiles": "1000"}),
    )
    hit = vacuum_overdue(TABLE, {}, rows, now)
    assert hit and hit.rule == VACUUM_OVERDUE and hit.key == f"VACUUM_OVERDUE:{TABLE}"
    assert "1,000 files (500 MB)" in hit.summary and "RETAIN 168 HOURS" in hit.summary

    recent, now = _history((2, "OPTIMIZE", 24, {"numRemovedFiles": "1000", "numRemovedBytes": str(500 * MB)}))
    assert vacuum_overdue(TABLE, {}, recent, now) is None          # only 1 day old: VACUUM would keep them


def test_vacuum_overdue_stops_at_the_last_vacuum_and_reads_merge_metrics():
    from agentic_ai.maintenance.scanner import vacuum_overdue

    cleaned, now = _history(
        (3, "VACUUM END", 2, {}),
        (2, "OPTIMIZE", 300, {"numRemovedFiles": "1000", "numRemovedBytes": str(500 * MB)}),
    )
    assert vacuum_overdue(TABLE, {}, cleaned, now) is None

    merged, now = _history((5, "MERGE", 200, {"numTargetFilesRemoved": "40",
                                              "numTargetBytesRemoved": str(300 * MB)}))
    assert vacuum_overdue(TABLE, {}, merged, now) is not None      # few files but many bytes

    small, now = _history((5, "DELETE", 200, {"numRemovedFiles": "10", "numRemovedBytes": str(5 * MB)}))
    assert vacuum_overdue(TABLE, {}, small, now) is None           # not worth a VACUUM


def test_scanner_vacuum_finding_and_guard_agree():
    from agentic_ai.maintenance.scanner import vacuum_overdue

    rows, now = _history((2, "OPTIMIZE", 170, {"numRemovedFiles": "1000", "numRemovedBytes": str(500 * MB)}))
    proposed = vacuum_overdue(TABLE, {}, rows, now).summary.split("`")[1]
    assert proposed == f"VACUUM {TABLE} RETAIN 168 HOURS" and needs_person(proposed) is False