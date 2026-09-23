"""
Failed-run detection sources.

Two implementations of the same interface:

  SystemTablesSource -- queries system.lakeflow.job_run_timeline. Preferred:
                        SQL-joinable against query history and lineage, which
                        Phase 4 and 5 need. Requires USE SCHEMA + SELECT on the
                        system schemas, granted at metastore level.

  JobsApiSource      -- calls jobs.list_runs() and filters to terminal failed
                        states. Works with ordinary workspace permissions, but
                        sees only jobs the caller can access, is rate limited,
                        and cannot be joined to other telemetry.

Config selects which. 'auto' probes system tables once and falls back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from agentic_ai.config import Settings

log = logging.getLogger(__name__)


@dataclass
class FailedRun:
    """One failed pipeline run, normalised across both sources."""

    run_id: str
    job_id: str
    job_name: str
    task_key: str
    result_state: str
    state_message: str
    start_time: datetime
    end_time: datetime
    run_page_url: str = ""
    source: str = ""
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def dedupe_key(self) -> str:
        """Stable identity for a run. Used by the watcher to avoid re-raising."""
        return f"{self.job_id}:{self.run_id}:{self.task_key or '_job'}"

    def as_error_text(self) -> str:
        """The text handed to the orchestrator as raw_error."""
        parts = [
            f"Job '{self.job_name}' run {self.run_id} finished with "
            f"result_state {self.result_state}."
        ]
        if self.task_key:
            parts.append(f"Failing task: {self.task_key}.")
        if self.state_message:
            parts.append(f"State message: {self.state_message}")
        return " ".join(parts)


def _epoch_ms_to_dt(value: int | None) -> datetime:
    if not value:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc)


def _ensure_utc(dt) -> datetime:
    """Spark returns TIMESTAMP columns as naive datetimes. Everything in this
    module treats datetimes as UTC-aware, so naive values from Spark are
    assumed to already be UTC (which is what Databricks stores) and get
    tzinfo attached rather than converted."""
    if dt is None:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

class DetectionSource:
    """Interface. Both implementations return runs that ENDED after `since`."""

    name = "base"

    def fetch_failed_runs(self, since: datetime, limit: int) -> list[FailedRun]:
        raise NotImplementedError

    def available(self) -> bool:
        raise NotImplementedError


class SystemTablesSource(DetectionSource):
    name = "system_tables"

    def __init__(self, settings: Settings, spark=None):
        self.settings = settings
        self._spark = spark or _get_spark()

    def available(self) -> bool:
        try:
            self._spark.sql(
                "SELECT 1 FROM system.lakeflow.job_run_timeline LIMIT 1"
            ).collect()
            return True
        except Exception as exc:
            log.info("system tables unavailable: %s", str(exc)[:160])
            return False

    def _job_names(self, job_ids: list[str]) -> dict[str, str]:
        """Look up current job names. Returns {} if system.lakeflow.jobs is absent.

        Kept separate from the main query so a missing or ungranted `jobs`
        table degrades to synthesised names instead of failing detection.
        """
        if not job_ids:
            return {}
        id_list = ", ".join(f"'{j}'" for j in job_ids)
        try:
            rows = self._spark.sql(f"""
                SELECT job_id, name
                FROM (
                    SELECT
                        job_id,
                        name,
                        ROW_NUMBER() OVER (PARTITION BY job_id ORDER BY change_time DESC) AS rn
                    FROM system.lakeflow.jobs
                    WHERE CAST(job_id AS STRING) IN ({id_list})
                )
                WHERE rn = 1
            """).collect()
            return {str(r.asDict().get("job_id")): r.asDict().get("name") or "" for r in rows}
        except Exception as exc:
            log.info("job name lookup unavailable: %s", str(exc)[:160])
            return {}

    def fetch_failed_runs(self, since: datetime, limit: int) -> list[FailedRun]:
        states = ", ".join(f"'{s}'" for s in self.settings.failed_states)
        since_literal = since.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        rows = self._spark.sql(f"""
            SELECT
                run_id,
                job_id,
                result_state,
                COALESCE(termination_code, '') AS state_message,
                period_start_time,
                period_end_time
            FROM system.lakeflow.job_run_timeline
            WHERE period_end_time > TIMESTAMP '{since_literal}'
              AND result_state IN ({states})
            ORDER BY period_end_time DESC
            LIMIT {int(limit)}
        """).collect()

        dicts = [r.asDict() for r in rows]
        names = self._job_names(sorted({str(d.get("job_id")) for d in dicts}))

        out = []
        for d in dicts:
            job_id = str(d.get("job_id", ""))
            out.append(FailedRun(
                run_id=str(d.get("run_id", "")),
                job_id=job_id,
                job_name=names.get(job_id) or f"job_{job_id}",
                task_key="",
                result_state=d.get("result_state", "") or "",
                state_message=d.get("state_message", "") or "",
                start_time=_ensure_utc(d.get("period_start_time")),
                end_time=_ensure_utc(d.get("period_end_time")),
                run_page_url="",
                source=self.name,
                raw=d,
            ))
        log.info("system_tables: %s failed run(s) since %s", len(out), since_literal)
        return out


class JobsApiSource(DetectionSource):
    name = "jobs_api"

    def __init__(self, settings: Settings, workspace_client=None):
        self.settings = settings
        self._w = workspace_client or _get_workspace_client()

    def available(self) -> bool:
        try:
            next(iter(self._w.jobs.list(limit=1)), None)
            return True
        except Exception as exc:
            log.warning("jobs api unavailable: %s", str(exc)[:160])
            return False

    def fetch_failed_runs(self, since: datetime, limit: int) -> list[FailedRun]:
        start_time_from = int(since.astimezone(timezone.utc).timestamp() * 1000)
        wanted = set(self.settings.failed_states)

        out = []
        for run in self._w.jobs.list_runs(
            completed_only=True,
            start_time_from=start_time_from,
            expand_tasks=True,
            limit=int(limit),
        ):
            state = run.state
            result_state = getattr(getattr(state, "result_state", None), "value", "") or ""
            if result_state not in wanted:
                continue

            failing_task = ""
            for task in (run.tasks or []):
                task_state = getattr(task, "state", None)
                task_result = getattr(getattr(task_state, "result_state", None), "value", "")
                if task_result in wanted:
                    failing_task = task.task_key or ""
                    break

            out.append(FailedRun(
                run_id=str(run.run_id or ""),
                job_id=str(run.job_id or ""),
                job_name=run.run_name or f"job_{run.job_id}",
                task_key=failing_task,
                result_state=result_state,
                state_message=(getattr(state, "state_message", "") or "")[:1000],
                start_time=_epoch_ms_to_dt(run.start_time),
                end_time=_epoch_ms_to_dt(run.end_time),
                run_page_url=run.run_page_url or "",
                source=self.name,
            ))

        log.info("jobs_api: %s failed run(s) since %s", len(out), since.isoformat())
        return out


def get_source(settings: Settings, spark=None) -> DetectionSource:
    """Build the configured detection source.

    settings.detection_source is 'system_tables', 'jobs_api', or 'auto'.
    'auto' prefers system tables and silently falls back so the watcher keeps
    running while metastore grants are outstanding.
    """
    choice = (settings.detection_source or "auto").lower()

    if choice == "system_tables":
        return SystemTablesSource(settings, spark=spark)
    if choice == "jobs_api":
        return JobsApiSource(settings)

    system_source = SystemTablesSource(settings, spark=spark)
    if system_source.available():
        log.info("auto-selected detection source: system_tables")
        return system_source
    log.warning("system tables not reachable, falling back to jobs_api")
    return JobsApiSource(settings)


def default_since(settings: Settings) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=settings.watcher_lookback_hours)


def _get_spark():
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


def _get_workspace_client():
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient()