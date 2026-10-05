import time
from types import SimpleNamespace as NS

import pytest

from agentic_ai.agents import ingestion
from agentic_ai.agents.base import ToolContext
from agentic_ai.agents.ingestion import (
    IngestionAgent,
    find_endpoints,
    landing_path_check,
    reachability,
    source_endpoints,
)
from agentic_ai.config import Settings
from agentic_ai.telemetry.run_context import RunContext, TaskError

JDBC_ERROR = (
    "com.microsoft.sqlserver.jdbc.SQLServerException: The TCP/IP connection to the host "
    "sql-prod.example.net, port 1433 has failed. url=jdbc:sqlserver://sql-prod.example.net:1433;"
    "database=sales\nFile /databricks/spark/python/pyspark/sql/connect/session.py:512"
)
RATE_ERROR = (
    "requests.exceptions.HTTPError: 429 Client Error: Too Many Requests, "
    "status code: 429, Retry-After: 30"
)
LANDING_ERROR = (
    "[CF_EMPTY_DIR_FOR_SCHEMA_INFERENCE] Cannot infer schema when the input path "
    "`abfss://landing@acct.dfs.core.windows.net/orders/` is empty."
)


def _tctx(error, spark=None):
    run = RunContext(job_id="1", run_id="2", job_name="orders_ingest",
                     task_errors=[TaskError("load", "3", "FAILED", "notebook /x", error=error)])
    return ToolContext(settings=Settings(), run=run, spark=spark)


def test_find_endpoints_parses_jdbc_host_and_ignores_source_files():
    found = find_endpoints(JDBC_ERROR)
    assert found["hosts"] == {"sql-prod.example.net": 1433}
    assert not any(h.endswith(".py") for h in found["hosts"])


def test_find_endpoints_status_codes_retry_after_and_paths():
    rate = find_endpoints(RATE_ERROR)
    assert rate["status_codes"] == ["429"] and rate["retry_after_sec"] == [30]

    land = find_endpoints(LANDING_ERROR)
    assert land["paths"] == ["abfss://landing@acct.dfs.core.windows.net/orders"]


def test_source_endpoints_summary():
    summary, _ = source_endpoints(_tctx(RATE_ERROR))
    assert "HTTP status codes: 429" in summary and "retry after 30s" in summary


def test_reachability_reports_down_and_up(monkeypatch):
    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_connect(addr, timeout):
        if addr[0] == "sql-prod.example.net":
            raise TimeoutError("timed out")
        return _Conn()

    monkeypatch.setattr(ingestion.socket, "create_connection", fake_connect)
    summary, data = reachability(_tctx(JDBC_ERROR))
    assert "NOT reachable now" in summary and data["sql-prod.example.net"]["reachable"] is False

    summary, data = reachability(_tctx("connection to api.example.com:443 reset"))
    assert data["api.example.com"]["reachable"] is True


class FakeSpark:
    def __init__(self, rows=None, error=None):
        self.rows, self.error, self.queries = rows or [], error, []

    def sql(self, query):
        self.queries.append(query)
        if self.error:
            raise self.error
        return NS(collect=lambda: self.rows)


def test_landing_path_stale_empty_and_missing():
    now_ms = int(time.time() * 1000)
    stale = FakeSpark([{"name": "a.json", "modification_time": now_ms - 3 * 86_400_000}])
    summary, _ = landing_path_check(_tctx(LANDING_ERROR, stale))
    assert "NO NEW DATA ARRIVING" in summary and stale.queries[0].startswith("LIST 'abfss://")

    empty = FakeSpark([])
    assert "NO FILES" in landing_path_check(_tctx(LANDING_ERROR, empty))[0]

    missing = FakeSpark(error=RuntimeError("PATH_NOT_FOUND"))
    summary, data = landing_path_check(_tctx(LANDING_ERROR, missing))
    assert "NOT LISTABLE" in summary and not list(data.values())[0]["exists"]


@pytest.mark.parametrize("error,expected", [
    (JDBC_ERROR, "CONNECTIVITY"),
    (RATE_ERROR, "RATE_LIMIT_RECOVERY"),
    (LANDING_ERROR, "LANDING_FILES"),
])
def test_subagent_selection(error, expected):
    sub, _ = IngestionAgent().select_subagent(_tctx(error).run)
    assert sub.name == expected