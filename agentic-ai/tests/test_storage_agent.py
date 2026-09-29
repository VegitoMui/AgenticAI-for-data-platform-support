from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

from agentic_ai.agents.base import ToolContext
from agentic_ai.agents.storage import (
    StorageAgent,
    data_profile,
    delta_history,
    quote,
    table_detail,
    table_exists,
    table_properties,
    target_tables,
)
from agentic_ai.agents.trace import Trace
from agentic_ai.config import Settings
from agentic_ai.telemetry.run_context import RunContext, TaskError

NOW = datetime.now(timezone.utc)
MB = 1024 * 1024


class FakeSpark:
    """Answers by query prefix; a value that is an Exception is raised."""

    def __init__(self, answers):
        self.answers = answers
        self.queries = []

    def sql(self, query):
        self.queries.append(query)
        for prefix, rows in self.answers.items():
            if query.strip().startswith(prefix):
                if isinstance(rows, Exception):
                    raise rows
                return NS(collect=lambda rows=rows: rows)
        raise RuntimeError(f"unexpected query: {query[:60]}")


def _tctx(spark, written=(), read=(), code=(), error="boom"):
    run = RunContext(job_id="1", run_id="2", job_name="orders_etl",
                     task_errors=[TaskError("t", "3", "FAILED", "notebook /x", error=error)],
                     tables_written=list(written), tables_read=list(read), tables_from_code=list(code))
    return ToolContext(settings=Settings(), run=run, spark=spark)


def test_quote_rejects_unsafe_names():
    assert quote("main.sales.orders") == "`main`.`sales`.`orders`"
    assert quote("main.sales") is None
    assert quote("main.sales.orders; DROP TABLE x") is None


def test_target_tables_prefers_written_then_read_then_code_and_dedupes():
    tctx = _tctx(None, written=["a.b.w"], read=["a.b.r", "a.b.w"], code=["a.b.c", "bad name"])
    assert target_tables(tctx) == ["a.b.w", "a.b.r", "a.b.c"]


def test_missing_table_suggests_similar_names():
    spark = FakeSpark({
        "DESCRIBE TABLE": RuntimeError("[TABLE_OR_VIEW_NOT_FOUND] orderz"),
        "SHOW TABLES": [{"tableName": "orders"}, {"tableName": "customers"}],
    })
    summary, data = table_exists(_tctx(spark, code=["main.sales.orderz"]))
    assert "NOT READABLE" in summary and "main.sales.orders" in summary
    assert data["main.sales.orderz"]["exists"] is False


def test_detail_flags_small_files_only_for_large_tables():
    many_small = [{"format": "delta", "numFiles": 20000, "sizeInBytes": 20000 * 2 * MB,
                   "partitionColumns": [], "clusteringColumns": [], "lastModified": NOW,
                   "minReaderVersion": 1, "minWriterVersion": 2}]
    summary, data = table_detail(_tctx(FakeSpark({"DESCRIBE DETAIL": many_small}), written=["m.s.big"]))
    assert "SMALL-FILE PROBLEM LIKELY" in summary and data["m.s.big"]["small_files"]

    tiny = [dict(many_small[0], numFiles=300, sizeInBytes=300 * 10_000)]
    summary, data = table_detail(_tctx(FakeSpark({"DESCRIBE DETAIL": tiny}), written=["m.s.tiny"]))
    assert "unlikely to matter" in summary and not data["m.s.tiny"]["small_files"]


def test_history_counts_writers_and_upkeep():
    hist = [
        {"operation": "WRITE", "timestamp": NOW - timedelta(hours=1), "job": {"jobId": "11"}},
        {"operation": "MERGE", "timestamp": NOW - timedelta(hours=2), "job": {"jobId": "22"}},
        {"operation": "OPTIMIZE", "timestamp": NOW - timedelta(days=10), "job": None},
    ]
    summary, data = delta_history(_tctx(FakeSpark({"DESCRIBE HISTORY": hist}), written=["m.s.t"]))
    assert "MULTIPLE CONCURRENT WRITERS" in summary
    assert data["m.s.t"]["writes_24h"] == 2
    assert "not in recent history" in summary  # no VACUUM


def test_properties_note_missing_auto_optimize():
    props = [{"key": "delta.logRetentionDuration", "value": "interval 30 days"}]
    summary, _ = table_properties(_tctx(FakeSpark({"SHOW TBLPROPERTIES": props}), written=["m.s.t"]))
    assert "optimized writes not enabled" in summary


def test_profile_reports_duplicates_and_heavy_nulls():
    spark = FakeSpark({
        "DESCRIBE TABLE": [{"col_name": "id"}, {"col_name": "email"}, {"col_name": ""}],
        "SELECT": [{"rows_sampled": 100, "distinct_rows": 90, "id": 0, "email": 40}],
    })
    summary, data = data_profile(_tctx(spark, written=["m.s.t"]))
    assert "10 fully duplicate rows" in summary and "email 40%" in summary
    assert "TABLESAMPLE" in spark.queries[-1]


def test_no_tables_is_a_finding_not_an_error():
    summary, data = table_detail(_tctx(FakeSpark({})))
    assert "No tables were identified" in summary and data == {}


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def chat_json(self, prompt, system_message="", max_tokens=0):
        self.prompts.append(prompt)
        return NS(ok=True, provider="fake", content="", parsed={
            "diagnosis": "Many small files.", "evidence_used": ["table_detail"], "severity": 5,
            "fix_complexity": "LOW", "fix_suggestion": "Compact", "actions": ["OPTIMIZE m.s.big"],
            "requires_human": False, "confidence": 0.8,
        })


def test_agent_routes_small_files_to_optimization_and_grounds_prompt():
    spark = FakeSpark({
        "DESCRIBE DETAIL": [{"format": "delta", "numFiles": 20000, "sizeInBytes": 20000 * 2 * MB,
                             "lastModified": NOW, "minReaderVersion": 1, "minWriterVersion": 2}],
        "DESCRIBE HISTORY": [{"operation": "WRITE", "timestamp": NOW, "job": {"jobId": "1"}}],
        "SHOW TBLPROPERTIES": [],
    })
    llm, trace = FakeLLM(), Trace("INC", "orders_etl")
    tctx = _tctx(spark, written=["m.s.big"], error="query slow: too many small files")
    d = StorageAgent().run(llm, tctx, trace)

    assert d.toolkit_used == "storage/OPTIMIZATION"
    assert [s.node_name for s in trace.steps] == [
        "select_subagent", "tool:table_detail", "tool:delta_history", "tool:table_properties", "reason",
    ]
    assert "SMALL-FILE PROBLEM LIKELY" in llm.prompts[0]