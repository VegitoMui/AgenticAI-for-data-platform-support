from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest

from agentic_ai.agents import orchestrator
from agentic_ai.agents.storage import StorageAgent
from agentic_ai.config import Settings
from agentic_ai.maintenance.scanner import (
    DUPLICATE_TABLE,
    base_name,
    compare_duplicate,
    duplicate_candidates,
)
from agentic_ai.telemetry.run_context import TABLE_SCAN_SOURCE, build_run_context

S = "databricks_ws.agentic_probe"
NOW = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)
COLS = [{"col_name": "customer_id", "data_type": "bigint"}, {"col_name": "name", "data_type": "string"},
        {"col_name": ""}, {"col_name": "# Partition Information"}]


@pytest.mark.parametrize("leaf,expected", [
    ("customers_bkp", "customers"), ("customers_backup", "customers"),
    ("customers_copy2", "customers"), ("customers_old", "customers"),
    ("customers_20261013", "customers"), ("customers_2026_10_13", "customers"),
    ("copy_of_customers", "customers"), ("customers_bkp_20261013", "customers"),
    ("customers_v2", "customers"), ("customers", "customers"),
    ("orders_small_files", "orders_small_files"), ("tmp", "tmp"),
])
def test_base_name(leaf, expected):
    assert base_name(leaf) == expected


def test_candidates_need_the_original_in_the_same_schema():
    tables = [f"{S}.customers", f"{S}.customers_bkp", f"{S}.orders_old", f"{S}.orders_small_files",
              "databricks_ws.other.customers_copy"]
    assert duplicate_candidates(tables) == [(f"{S}.customers", f"{S}.customers_bkp")]


class FakeSpark:
    def __init__(self, cols_copy=COLS, rows=(1000, 1000), lineage=None, lineage_error=False):
        self.cols_copy, self.rows, self.lineage, self.lineage_error = cols_copy, rows, lineage, lineage_error

    def sql(self, q):
        q1 = " ".join(q.split())
        if q1.startswith("DESCRIBE TABLE"):
            rows = self.cols_copy if "customers_bkp" in q1 else COLS
        elif q1.startswith("SELECT count(*)"):
            rows = [{"n": self.rows[1] if "customers_bkp" in q1 else self.rows[0]}]
        elif q1.startswith("DESCRIBE DETAIL"):
            age = timedelta(days=40) if "customers_bkp" in q1 else timedelta(hours=2)
            rows = [{"lastModified": NOW - age, "sizeInBytes": 300 * 1024 * 1024}]
        else:
            if self.lineage_error:
                raise RuntimeError("PERMISSION_DENIED system.access")
            rows = [{"last_seen": self.lineage}]
        return NS(collect=lambda: rows)


def test_identical_copy_is_reported_with_its_evidence():
    f = compare_duplicate(FakeSpark(), f"{S}.customers", f"{S}.customers_bkp", NOW)
    assert f.rule == DUPLICATE_TABLE and f.key == f"DUPLICATE_TABLE:{S}.customers_bkp"
    assert f"leftover copy of `{S}.customers`" in f.summary
    assert "identical 2 columns, 1,000 rows against 1,000" in f.summary
    assert "last written 40 days ago against 2 h ago" in f.summary and "300 MB" in f.summary
    assert "no job, notebook or query read or wrote it in the last 90 days" in f.summary


def test_not_a_duplicate_when_columns_or_rows_differ():
    other_cols = [{"col_name": "customer_id", "data_type": "bigint"}, {"col_name": ""}]
    pair = (f"{S}.customers", f"{S}.customers_bkp")
    assert compare_duplicate(FakeSpark(cols_copy=other_cols), *pair, NOW) is None   # different columns
    assert compare_duplicate(FakeSpark(rows=(1000, 500)), *pair, NOW) is None       # half the rows
    assert compare_duplicate(FakeSpark(rows=(1000, 950)), *pair, NOW)               # within 10%


def test_unreadable_lineage_is_said_plainly():
    f = compare_duplicate(FakeSpark(lineage_error=True), f"{S}.customers", f"{S}.customers_bkp", NOW)
    assert "lineage was not readable" in f.summary


# ------------------------------------------------------------------ diagnosis path

INCIDENT = {"incident_id": "INC-1", "detection_source": TABLE_SCAN_SOURCE,
            "pipeline_name": f"table maintenance: {S}.customers_bkp",
            "task_key": f"DUPLICATE_TABLE:{S}.customers_bkp",
            "raw_error": f"DUPLICATE TABLE: {S}.customers_bkp looks like a leftover copy "
                         f"of `{S}.customers`: ..."}


def test_both_tables_become_evidence_and_the_right_sub_agent_is_chosen():
    ctx = build_run_context(Settings(), INCIDENT)
    assert ctx.tables_written == [f"{S}.customers_bkp"] and ctx.tables_from_code == [f"{S}.customers"]
    sub, hits = StorageAgent().select_subagent(ctx)
    assert sub.name == "DUPLICATE_TABLES" and "leftover copy" in hits


def test_duplicate_findings_always_need_a_person(monkeypatch):
    class ConfidentAgent:
        def run(self, llm, tctx, trace, memory):
            return NS(agent_name="storage", requires_human=False)

    monkeypatch.setitem(orchestrator.AGENTS, "storage", ConfidentAgent())
    monkeypatch.setattr(orchestrator.store, "recall", lambda *a, **k: ([], [], None))
    monkeypatch.setattr(orchestrator.Trace, "flush", lambda self, s, spark: True)

    out = orchestrator.diagnose_incident(Settings(), None, None, None, INCIDENT)
    assert out.diagnosis.requires_human is True

    small_files = dict(INCIDENT, task_key=f"SMALL_FILES:{S}.orders")
    out = orchestrator.diagnose_incident(Settings(), None, None, None, small_files)
    assert out.diagnosis.requires_human is False                  # only duplicates are forced