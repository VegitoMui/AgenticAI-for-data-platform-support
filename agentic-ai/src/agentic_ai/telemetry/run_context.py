"""
Run context: the evidence every agent starts from.

A detected incident only says "job X run Y failed". Before any agent reasons,
this module gathers what actually happened:

  task errors   For each failed task, the real error message and the tail of
                its stack trace, from the Jobs API (get_run + get_run_output).
  task types    What each task ran (notebook path, wheel entry point, SQL...).
  tables        Which tables the run read and wrote, from
                system.access.table_lineage. Lineage lags by minutes to hours,
                so it falls back to tables this job touched in recent runs.
  code tables   Fully-qualified table names found in the error text and stack
                trace (e.g. spark.table('cat.sch.tbl')). Always available the
                moment the run fails, labelled separately because it is weaker
                evidence than lineage.

Every lookup is best-effort: a failure in one source is recorded in `notes`
and never stops the others, so an agent always gets whatever is available.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from agentic_ai.config import Settings

log = logging.getLogger(__name__)

MAX_ERROR_CHARS = 1500
MAX_TRACE_CHARS = 3000
LINEAGE_FALLBACK_DAYS = 30
# Notebook tracebacks carry ANSI colour codes; they waste LLM tokens and
# clutter the App and email, so they are stripped.
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# catalog.schema.table inside quotes/backticks, or after a SQL keyword.
_QUOTED_TABLE = re.compile(r"[`'\"]([A-Za-z_]\w*\.[A-Za-z_]\w*\.[A-Za-z_]\w*)[`'\"]")
_SQL_TABLE = re.compile(
    r"(?i)\b(?:from|join|into|table|update)\s+`?([A-Za-z_]\w*\.[A-Za-z_]\w*\.[A-Za-z_]\w*)`?"
)
# Spark error messages quote each part separately: `cat`.`sch`.`tbl`
_BACKTICK_PARTS = re.compile(r"`([A-Za-z_]\w*)`\.`([A-Za-z_]\w*)`\.`([A-Za-z_]\w*)`")
# Dotted names that look like tables but are Python modules or Spark configs.
_NOT_TABLE_PREFIXES = ("spark.", "pyspark.", "databricks.", "system.", "java.", "org.", "com.", "py4j.")
FAILED_TASK_STATES = {"FAILED", "TIMEDOUT", "TIMED_OUT", "CANCELED", "CANCELLED", "ERROR", "INTERNAL_ERROR"}


@dataclass
class TaskError:
    task_key: str
    task_run_id: str
    result_state: str
    task_type: str
    error: str = ""
    error_trace: str = ""
    attempts: int = 1


@dataclass
class RunContext:
    job_id: str
    run_id: str
    job_name: str
    task_errors: list[TaskError] = field(default_factory=list)
    tables_read: list[str] = field(default_factory=list)
    tables_written: list[str] = field(default_factory=list)
    lineage_scope: str = "none"  # this_run | recent_runs | none
    tables_from_code: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def tables(self) -> list[str]:
        return sorted(set(self.tables_read) | set(self.tables_written) | set(self.tables_from_code))

    @property
    def primary_error(self) -> str:
        for t in self.task_errors:
            if t.error:
                return t.error
        return ""

    def to_prompt(self) -> str:
        """Compact, bounded text block for LLM prompts."""
        lines = [f"Job: {self.job_name} (job_id {self.job_id}, run {self.run_id})"]
        if self.task_errors:
            for t in self.task_errors:
                tries = f", failed on all {t.attempts} attempts" if t.attempts > 1 else ""
                lines.append(f"Failed task '{t.task_key}' [{t.task_type}] -> {t.result_state}{tries}")
                if t.error:
                    lines.append(f"  Error: {t.error}")
                if t.error_trace:
                    lines.append("  Stack trace (tail):")
                    lines.extend(f"    {ln}" for ln in t.error_trace.splitlines())
        else:
            lines.append("No task-level error details were available.")

        scope = {
            "this_run": "this run",
            "recent_runs": f"recent runs of this job (last {LINEAGE_FALLBACK_DAYS} days)",
            "none": "unknown",
        }[self.lineage_scope]
        lines.append(f"Tables read ({scope}): {', '.join(self.tables_read) or 'none found'}")
        lines.append(f"Tables written ({scope}): {', '.join(self.tables_written) or 'none found'}")
        if self.tables_from_code:
            lines.append(f"Tables referenced in the failing code: {', '.join(self.tables_from_code)}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "job_name": self.job_name,
            "task_errors": [t.__dict__ for t in self.task_errors],
            "tables_read": self.tables_read,
            "tables_written": self.tables_written,
            "lineage_scope": self.lineage_scope,
            "tables_from_code": self.tables_from_code,
            "notes": self.notes,
        }


def _clean(text: str) -> str:
    return _ANSI.sub("", text or "").strip()


def _clip_head(text: str, limit: int) -> str:
    text = _clean(text)
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"


def _clip_tail(text: str, limit: int) -> str:
    """Keep the end of a stack trace, where the actual exception is."""
    text = _clean(text)
    return text if len(text) <= limit else "...[truncated]\n" + text[-limit:]


def _task_type(task) -> str:
    if getattr(task, "notebook_task", None):
        return f"notebook {task.notebook_task.notebook_path}"
    if getattr(task, "python_wheel_task", None):
        w = task.python_wheel_task
        return f"wheel {w.package_name}:{w.entry_point}"
    if getattr(task, "spark_python_task", None):
        return f"python {task.spark_python_task.python_file}"
    if getattr(task, "sql_task", None):
        return "sql"
    if getattr(task, "pipeline_task", None):
        return "pipeline"
    return "other"


def _enum_value(obj) -> str:
    return getattr(obj, "value", None) or (str(obj) if obj is not None else "")


def fetch_task_errors(w, run_id: str, ctx: RunContext) -> None:
    try:
        run = w.jobs.get_run(run_id=int(run_id))
    except Exception as exc:
        ctx.notes.append(f"get_run failed: {str(exc)[:200]}")
        return

    if not ctx.job_name and getattr(run, "run_name", None):
        ctx.job_name = run.run_name

    # A task can appear more than once when it was retried (each attempt is a
    # separate task run with its own run_id). Report each task once, using its
    # latest attempt, and record how many attempts there were.
    latest: dict[str, object] = {}
    attempts: dict[str, int] = {}
    for task in run.tasks or []:
        key = task.task_key or str(task.run_id)
        attempts[key] = attempts.get(key, 0) + 1
        prev = latest.get(key)
        if prev is None or (getattr(task, "attempt_number", 0) or 0) >= (
            getattr(prev, "attempt_number", 0) or 0
        ):
            latest[key] = task

    for key, task in latest.items():
        task_run_id = str(task.run_id or "")
        state = getattr(task, "state", None)
        result_state = _enum_value(getattr(state, "result_state", None))
        life_state = _enum_value(getattr(state, "life_cycle_state", None))
        effective = result_state or life_state
        if effective not in FAILED_TASK_STATES:
            continue

        err = TaskError(
            task_key=task.task_key or "",
            task_run_id=task_run_id,
            result_state=effective,
            attempts=attempts[key],
            task_type=_task_type(task),
            error=_clip_head(getattr(state, "state_message", "") or "", MAX_ERROR_CHARS),
        )
        try:
            output = w.jobs.get_run_output(run_id=int(task.run_id))
            if output.error:
                err.error = _clip_head(output.error, MAX_ERROR_CHARS)
            err.error_trace = _clip_tail(output.error_trace or "", MAX_TRACE_CHARS)
        except Exception as exc:
            ctx.notes.append(f"get_run_output failed for {task.task_key}: {str(exc)[:200]}")
        ctx.task_errors.append(err)


def extract_tables_from_code(text: str) -> list[str]:
    """Fully-qualified table names quoted or used after a SQL keyword."""
    text = text or ""
    found = set(_QUOTED_TABLE.findall(text)) | set(_SQL_TABLE.findall(text))
    found |= {".".join(parts) for parts in _BACKTICK_PARTS.findall(text)}
    return sorted(n for n in found if not n.lower().startswith(_NOT_TABLE_PREFIXES))


def _lineage_rows(spark, where: str) -> list:
    return spark.sql(f"""
        SELECT DISTINCT source_table_full_name AS src, target_table_full_name AS tgt
        FROM system.access.table_lineage
        WHERE entity_type = 'JOB' AND {where}
    """).collect()


def fetch_lineage(spark, job_id: str, run_id: str, ctx: RunContext) -> None:
    try:
        rows = _lineage_rows(spark, f"entity_id = '{job_id}' AND entity_run_id = '{run_id}'")
        scope = "this_run"
        if not rows:
            rows = _lineage_rows(
                spark,
                f"entity_id = '{job_id}' "
                f"AND event_time > current_timestamp() - INTERVAL {LINEAGE_FALLBACK_DAYS} DAYS",
            )
            scope = "recent_runs" if rows else "none"
    except Exception as exc:
        ctx.notes.append(f"lineage lookup failed: {str(exc)[:200]}")
        return

    read, written = set(), set()
    for r in rows:
        src, tgt = r["src"], r["tgt"]
        if src and not src.startswith("system."):
            read.add(src)
        if tgt and not tgt.startswith("system."):
            written.add(tgt)
    ctx.tables_read = sorted(read)
    ctx.tables_written = sorted(written)
    ctx.lineage_scope = scope


def build_run_context(settings: Settings, incident: dict, workspace_client=None, spark=None) -> RunContext:
    """Gather evidence for one incident row (needs job_id, run_id, pipeline_name)."""
    job_id = str(incident.get("job_id") or "")
    run_id = str(incident.get("run_id") or "")
    ctx = RunContext(job_id=job_id, run_id=run_id, job_name=incident.get("pipeline_name") or "")

    if not run_id:
        ctx.notes.append("incident has no run_id; nothing to look up")
        return ctx

    if workspace_client is None:
        from databricks.sdk import WorkspaceClient

        workspace_client = WorkspaceClient()
    if spark is None:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()

    fetch_task_errors(workspace_client, run_id, ctx)
    ctx.tables_from_code = extract_tables_from_code(
        "\n".join(f"{t.error}\n{t.error_trace}" for t in ctx.task_errors)
    )
    if job_id:
        fetch_lineage(spark, job_id, run_id, ctx)

    log.info(
        "run context for %s: %s failed task(s), %s table(s) [%s]",
        run_id, len(ctx.task_errors), len(ctx.tables), ctx.lineage_scope,
    )
    return ctx