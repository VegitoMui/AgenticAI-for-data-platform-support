"""
Processing agent: the code and compute that ran the job.

Sub-agents
  CODE_AND_CONFIG     application code errors (bugs, bad input, explicit
                      raises) and bad Spark configuration. The default, since
                      most job failures start here.
  MEMORY              out-of-memory, GC overhead, lost executors, driver crashes.
  DATA_DISTRIBUTION   skew, shuffle, spill, fetch failures.
  STREAMING           streaming queries, checkpoints, offsets.

Tools
  error_analysis    Parses the real error and stack trace: exception type,
                    Spark error class, SQLSTATE, the user code line that
                    failed, and whether the failure was raised by application
                    code or by the engine. No external calls.
  run_history       Recent runs of the same job from the Jobs API: is this the
                    first failure after a run of successes (something changed),
                    consistent failure, or flaky? Also compares duration with
                    recent successful runs.
  compute_profile   What the failed task ran on: serverless environment or
                    cluster (node types, workers, autoscale, Spark version,
                    Spark conf overrides).
  table_layout      Reuses the Storage agent's DESCRIBE DETAIL tool on the
                    tables involved, for partitioning and file counts when
                    investigating skew.

All tools are read-only.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from agentic_ai.agents.base import BaseAgent, SubAgent, Tool, ToolContext
from agentic_ai.agents.storage import table_detail

RUN_HISTORY_LIMIT = 20

# Exceptions that are raised by application code rather than the engine.
_APP_EXCEPTIONS = {
    "ValueError", "KeyError", "TypeError", "IndexError", "AttributeError", "NameError",
    "ZeroDivisionError", "AssertionError", "RuntimeError", "Exception", "NotImplementedError",
    "FileNotFoundError", "ImportError", "ModuleNotFoundError", "UnboundLocalError",
}
_EXCEPTION = re.compile(r"\b((?:[a-z_][\w]*\.)*[A-Z]\w*(?:Exception|Error))\b")
_ERROR_CLASS = re.compile(r"\[([A-Z][A-Z0-9_]{3,})\]")
_SQLSTATE = re.compile(r"SQLSTATE:\s*([0-9A-Z]{5})")
_USER_LINE = re.compile(r"-{2,}>\s*\d+\s+(.+)")
_USER_FRAME = re.compile(r"File <command-\d+>, line (\d+)|File (/Workspace/[^,]+), line (\d+)")

def _user_failing_line(text: str) -> str:
    """The '---->' line inside the job's own code (a notebook cell or a
    /Workspace file), not one inside a library it called. Falls back to the
    last '---->' line if no user frame is found."""
    in_user_frame, user_line, any_line = False, "", ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("File "):
            in_user_frame = bool(_USER_FRAME.match(stripped))
            continue
        m = _USER_LINE.search(line)
        if m:
            any_line = m.group(1).strip()
            if in_user_frame:
                user_line = any_line
    return user_line or any_line

def _workspace(tctx: ToolContext):
    if tctx.workspace_client is None:
        from databricks.sdk import WorkspaceClient

        tctx.workspace_client = WorkspaceClient()
    return tctx.workspace_client


def _enum(v) -> str:
    return getattr(v, "value", None) or (str(v) if v is not None else "")


def _hours_since_ms(ms) -> float | None:
    if not ms:
        return None
    return (datetime.now(timezone.utc).timestamp() * 1000 - ms) / 3_600_000


def _ago(hours: float | None) -> str:
    if hours is None:
        return "unknown"
    return f"{hours:.1f} h ago" if hours < 48 else f"{hours / 24:.0f} days ago"


# ------------------------------------------------------------------ tools

def error_analysis(tctx: ToolContext) -> tuple[str, dict]:
    if not tctx.run.task_errors:
        return "No error text was available for this run.", {}

    findings, data = [], {}
    for t in tctx.run.task_errors:
        text = f"{t.error}\n{t.error_trace}"
        exceptions = _EXCEPTION.findall(text)
        final = exceptions[-1] if exceptions else "unknown"
        short = final.split(".")[-1]
        error_class = (_ERROR_CLASS.findall(text) or [""])[0]
        sqlstate = (_SQLSTATE.findall(text) or [""])[0]
        
        user_line = _user_failing_line(text)
        in_user_code = bool(_USER_FRAME.search(text))

        if short in _APP_EXCEPTIONS and in_user_code:
            origin = "RAISED BY APPLICATION CODE"
        elif error_class or "java." in final or "spark" in final.lower() or "py4j" in text.lower():
            origin = "raised by the Spark engine"
        elif in_user_code:
            origin = "raised by a library called from the job's code"
        else:
            origin = "origin unclear"

        parts = [f"task '{t.task_key}': final exception {final}", origin]
        if error_class:
            parts.append(f"error class {error_class}")
        if sqlstate:
            parts.append(f"SQLSTATE {sqlstate}")
        if user_line:
            parts.append(f"failing line: {user_line[:200]}")
        if t.attempts > 1:
            parts.append(f"failed on all {t.attempts} attempts, so not transient")
        findings.append(", ".join(parts))
        data[t.task_key] = {"exception": final, "origin": origin, "error_class": error_class,
                            "sqlstate": sqlstate, "user_line": user_line, "attempts": t.attempts}
    return "; ".join(findings), data


def run_history(tctx: ToolContext) -> tuple[str, dict]:
    job_id, run_id = tctx.run.job_id, tctx.run.run_id
    if not job_id:
        return "No job id; run history unavailable.", {}
    runs = list(_workspace(tctx).jobs.list_runs(job_id=int(job_id), completed_only=True,
                                                limit=RUN_HISTORY_LIMIT))
    if not runs:
        return "No completed runs found for this job.", {}

    # newest first
    runs.sort(key=lambda r: r.start_time or 0, reverse=True)
    states = [_enum(getattr(r.state, "result_state", None)) for r in runs]
    ok = [s == "SUCCESS" for s in states]

    streak = 0
    for s in ok:
        if s:
            break
        streak += 1
    successes = sum(ok)
    last_success = next((r for r, good in zip(runs, ok, strict=True) if good), None)

    if successes == 0:
        pattern = f"NEVER SUCCEEDED in the last {len(runs)} runs (likely a new or always-broken job)"
    elif streak == 1:
        pattern = f"FIRST FAILURE after {successes} recent successes (something changed recently)"
    elif streak > 1:
        pattern = f"FAILING CONSISTENTLY: last {streak} runs failed"
    else:
        pattern = "latest run succeeded"
    failures = len(runs) - successes
    if 0 < failures and streak <= 1 and failures >= 3:
        pattern += f"; FLAKY: {failures} of {len(runs)} recent runs failed"

    ok_durations = [
        (r.end_time - r.start_time) / 1000
        for r, good in zip(runs, ok, strict=True) if good and r.end_time and r.start_time
    ]
    this = next((r for r in runs if str(r.run_id) == str(run_id)), None)
    duration_note = ""
    if this and this.end_time and this.start_time and ok_durations:
        this_s = (this.end_time - this.start_time) / 1000
        typical = sorted(ok_durations)[len(ok_durations) // 2]
        duration_note = f"this run lasted {this_s:.0f}s vs typical successful {typical:.0f}s"

    parts = [f"{len(runs)} recent runs: {successes} succeeded, {failures} failed", pattern]
    if last_success:
        parts.append(f"last success {_ago(_hours_since_ms(last_success.start_time))}")
    if duration_note:
        parts.append(duration_note)
    return ", ".join(parts), {"states": states, "failure_streak": streak, "successes": successes}


def compute_profile(tctx: ToolContext) -> tuple[str, dict]:
    run_id = tctx.run.run_id
    if not run_id:
        return "No run id; compute profile unavailable.", {}
    run = _workspace(tctx).jobs.get_run(run_id=int(run_id))
    failed_keys = {t.task_key for t in tctx.run.task_errors}
    job_clusters = {jc.job_cluster_key: jc.new_cluster for jc in (getattr(run, "job_clusters", None) or [])}

    findings, data = [], {}
    for task in run.tasks or []:
        # Retried tasks appear once per attempt; they share the same compute.
        if task.task_key in data or (failed_keys and task.task_key not in failed_keys):
            continue
        spec = getattr(task, "new_cluster", None) or job_clusters.get(getattr(task, "job_cluster_key", None))
        if spec is not None:
            autoscale = getattr(spec, "autoscale", None)
            workers = (f"autoscale {autoscale.min_workers}-{autoscale.max_workers}" if autoscale
                       else f"{getattr(spec, 'num_workers', 0)} workers")
            conf = getattr(spec, "spark_conf", None) or {}
            desc = (f"job cluster: node {getattr(spec, 'node_type_id', '?')}, "
                    f"driver {getattr(spec, 'driver_node_type_id', None) or 'same as workers'}, "
                    f"{workers}, runtime {getattr(spec, 'spark_version', '?')}")
            if conf:
                desc += f", spark_conf overrides: {', '.join(f'{k}={v}' for k, v in conf.items())}"
            kind = "job_cluster"
        elif getattr(task, "existing_cluster_id", None):
            desc = f"existing all-purpose cluster {task.existing_cluster_id}"
            kind = "existing_cluster"
        else:
            desc = ("serverless compute: memory and node sizing are managed by Databricks, "
                    "so cluster-level Spark settings cannot be changed")
            kind = "serverless"
        findings.append(f"task '{task.task_key}': {desc}")
        data[task.task_key] = {"kind": kind}
    return "; ".join(findings) or "No task compute information found.", data


# ------------------------------------------------------------------ agent

_COMMON = ("error_analysis", "run_history", "compute_profile")


class ProcessingAgent(BaseAgent):
    name = "processing"
    description = "Application code errors, Spark configuration, memory, skew and streaming."

    sub_agents = [
        SubAgent(
            "CODE_AND_CONFIG",
            "Did the job's own code or configuration cause this? Separate bugs and bad input "
            "(fix the code) from engine problems. Use run history to tell a new regression "
            "from a long-standing or flaky failure.",
            ("valueerror", "keyerror", "typeerror", "attributeerror", "assertionerror",
             "nameerror", "zerodivisionerror", "raise ", "spark.conf", "invalid value",
             "config"),
            _COMMON,
        ),
        SubAgent(
            "MEMORY",
            "Did the job run out of memory? Relate the failure to the compute it ran on; "
            "on serverless, cluster memory settings cannot be changed.",
            ("outofmemory", "out of memory", "heap space", "gc overhead", "container killed",
             "executor lost", "driver crashed", "oomkilled"),
            _COMMON,
        ),
        SubAgent(
            "DATA_DISTRIBUTION",
            "Is work unevenly spread? Look for skew, shuffle spill and fetch failures, "
            "and at how the input tables are partitioned.",
            ("skew", "shuffle", "spill", "fetchfailed", "straggler", "partition"),
            _COMMON + ("table_layout",),
        ),
        SubAgent(
            "STREAMING",
            "Is a streaming query failing? Look at checkpoint, offset and source problems.",
            ("streaming", "streamingqueryexception", "checkpoint", "offset", "watermark",
             "trigger", "microbatch"),
            _COMMON,
        ),
    ]

    tools = {
        "error_analysis": Tool("error_analysis", "Exception, error class, failing line, origin",
                               error_analysis),
        "run_history": Tool("run_history", "Recent runs: first failure, consistent or flaky",
                            run_history),
        "compute_profile": Tool("compute_profile", "Serverless or cluster, sizing, spark_conf",
                                compute_profile),
        "table_layout": Tool("table_layout", "DESCRIBE DETAIL of the tables involved", table_detail),
    }