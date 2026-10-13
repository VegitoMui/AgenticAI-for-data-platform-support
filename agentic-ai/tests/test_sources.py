from types import SimpleNamespace as NS

from agentic_ai.config import Settings
from agentic_ai.telemetry import sources
from agentic_ai.telemetry.sources import SystemTablesSource


class FakeSpark:
    def __init__(self):
        self.queries = []
        self.conf = NS(get=lambda key, default="": default)

    def sql(self, q):
        self.queries.append(" ".join(q.split()))
        return NS(collect=lambda: [])


def test_failed_runs_and_job_names_are_filtered_to_this_workspace():
    spark = FakeSpark()
    src = SystemTablesSource(Settings(), spark=spark, workspace_id="7405617514546966")
    src.fetch_failed_runs(sources.default_since(Settings()), 25)
    src._job_names(["123"])

    assert "AND workspace_id = '7405617514546966'" in spark.queries[0]
    assert "system.lakeflow.job_run_timeline" in spark.queries[0]
    assert "AND workspace_id = '7405617514546966'" in spark.queries[1]
    assert "system.lakeflow.jobs" in spark.queries[1]


def test_unknown_workspace_id_degrades_to_no_filter(monkeypatch):
    monkeypatch.setattr(sources, "_current_workspace_id", lambda spark=None: "")
    spark = FakeSpark()
    SystemTablesSource(Settings(), spark=spark).fetch_failed_runs(sources.default_since(Settings()), 25)
    assert "workspace_id" not in spark.queries[0]


def test_current_workspace_id_prefers_api_then_spark_conf(monkeypatch):
    client = NS(get_workspace_id=lambda: 7405617514546966)
    monkeypatch.setattr(sources, "_get_workspace_client", lambda: client)
    assert sources._current_workspace_id(FakeSpark()) == "7405617514546966"

    def broken():
        raise RuntimeError("no auth")

    monkeypatch.setattr(sources, "_get_workspace_client", broken)
    spark = FakeSpark()
    spark.conf = NS(get=lambda key, default="": "42" if key.endswith("clusterOwnerOrgId") else default)
    assert sources._current_workspace_id(spark) == "42"
    assert sources._current_workspace_id(FakeSpark()) == ""


def test_job_names_missing_from_system_table_come_from_the_jobs_api():
    class Jobs:
        def get(self, job_id):
            if job_id == 404:
                raise RuntimeError("RESOURCE_DOES_NOT_EXIST")
            return NS(settings=NS(name=f"orders load {job_id}"))

    src = SystemTablesSource(Settings(), spark=FakeSpark(), workspace_id="1",
                             workspace_client=NS(jobs=Jobs()))
    names = src._names_from_api(["9766604919893", "404", "abc"])
    assert names == {"9766604919893": "orders load 9766604919893"}