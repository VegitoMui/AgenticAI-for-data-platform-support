"""
Reconciler: Phase 2 diagnosis, approval routing, and execution.

Runs on a schedule (default every 2 minutes). One pass does four things,
in order:

  1. Expire stale approval_requests (pending longer than approval_expiry_hours).
     Expiry takes NO action -- it does not approve or reject, it just marks
     the request EXPIRED so it stops waiting and the incident is visibly
     stuck rather than silently ignored.

  2. Diagnose NEW incidents. Each gets a Diagnosis (diagnosis.py). Based on
     needs_approval(), it is either:
       a) auto-executed immediately, remediation_log written, a GitHub issue
          opened and immediately closed with the outcome, or
       b) blocked: an approval_requests row is written, a GitHub issue is
          opened and left open, incident status becomes PENDING_APPROVAL.

  3. Execute approval_requests with status='approved' (set by the App UI,
     not by this job). Writes remediation_log, closes the GitHub issue,
     updates incident status.

  4. Close out approval_requests with status='rejected_pending_close' --
     close the issue with the reviewer's note, update incident status,
     no execution.

GitHub issue creation/closure failures are logged and swallowed -- they must
never block remediation or approval processing.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

from agentic_ai.config import Settings
from agentic_ai.llm.client import LLMClient
from agentic_ai.remediation import executor
from agentic_ai.remediation.diagnosis import diagnose, needs_approval
from agentic_ai.vcs.github import GitHubClient

log = logging.getLogger(__name__)


def _get_spark():
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


# --------------------------------------------------------------------- step 1

def expire_stale_requests(settings: Settings, spark) -> int:
    table = settings.table("approval_requests")
    incidents = settings.table("incidents")
    cutoff = datetime.now(timezone.utc) - timedelta(hours=settings.approval_expiry_hours)
    cutoff_literal = cutoff.strftime("%Y-%m-%d %H:%M:%S")

    stale = spark.sql(f"""
        SELECT request_id, incident_id FROM {table}
        WHERE status = 'pending' AND requested_at < TIMESTAMP '{cutoff_literal}'
    """).collect()

    if not stale:
        return 0

    for row in stale:
        spark.sql(f"""
            UPDATE {table}
            SET status = 'EXPIRED', reviewed_at = current_timestamp(),
                reviewer_notes = 'Auto-expired after {settings.approval_expiry_hours}h with no response'
            WHERE request_id = '{row['request_id']}'
        """)
        spark.sql(f"""
            UPDATE {incidents}
            SET status = 'EXPIRED', updated_at = current_timestamp()
            WHERE incident_id = '{row['incident_id']}'
        """)
    log.info("expired %s stale approval request(s)", len(stale))
    return len(stale)


# --------------------------------------------------------------------- step 2

def _write_remediation_log(settings: Settings, spark, incident_id: str, request_id: str,
                           pipeline_name: str, agent_name: str, complexity: str,
                           results: list[dict]) -> tuple[int, int]:
    table = settings.table("remediation_log")
    if not results:
        return 0, 0

    from pyspark.sql.types import (
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    schema = StructType([
        StructField("remediation_id", StringType()),
        StructField("incident_id", StringType()),
        StructField("request_id", StringType()),
        StructField("pipeline_name", StringType()),
        StructField("agent_name", StringType()),
        StructField("action_index", IntegerType()),
        StructField("action_text", StringType()),
        StructField("action_type", StringType()),
        StructField("execution_status", StringType()),
        StructField("execution_output", StringType()),
        StructField("error_message", StringType()),
        StructField("complexity", StringType()),
        StructField("duration_sec", DoubleType()),
    ])

    rows = [(
        f"REM-{uuid.uuid4().hex[:8].upper()}", incident_id, request_id, pipeline_name,
        agent_name, r["action_index"], r["action_text"], r["action_type"],
        r["execution_status"], r["execution_output"], r["error_message"],
        complexity, r["duration_sec"],
    ) for r in results]

    df = spark.createDataFrame(rows, schema=schema)
    df = df.selectExpr(*[f.name for f in schema.fields], "current_timestamp() AS executed_at")
    df.write.format("delta").mode("append").saveAsTable(table)

    succeeded = sum(1 for r in results if r["execution_status"] == "SUCCESS")
    failed = sum(1 for r in results if r["execution_status"] == "FAILED")
    return succeeded, failed


def _write_approval_request(settings: Settings, spark, incident: dict, d) -> str:
    table = settings.table("approval_requests")
    request_id = f"APR-{uuid.uuid4().hex[:8].upper()}"

    from pyspark.sql.types import (
        BooleanType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    schema = StructType([
        StructField("request_id", StringType()),
        StructField("incident_id", StringType()),
        StructField("pipeline_name", StringType()),
        StructField("agent_name", StringType()),
        StructField("priority", StringType()),
        StructField("severity", IntegerType()),
        StructField("diagnosis", StringType()),
        StructField("proposed_actions", StringType()),
        StructField("fix_complexity", StringType()),
        StructField("fix_suggestion", StringType()),
        StructField("requires_human", BooleanType()),
        StructField("status", StringType()),
    ])

    row = [(
        request_id, incident["incident_id"], incident["pipeline_name"], d.agent_name,
        d.fix_complexity, d.severity, d.diagnosis, json.dumps(d.actions)[:2000],
        d.fix_complexity, d.fix_suggestion, d.requires_human, "pending",
    )]
    df = spark.createDataFrame(row, schema=schema)
    df = df.selectExpr(
        *[f.name for f in schema.fields],
        "current_timestamp() AS requested_at",
        f"current_timestamp() + INTERVAL {settings.approval_expiry_hours} HOURS AS expires_at",
        "CAST(NULL AS TIMESTAMP) AS reviewed_at",
        "CAST(NULL AS STRING) AS reviewed_by",
        "CAST(NULL AS STRING) AS reviewer_notes",
        "CAST(NULL AS INT) AS github_issue_number",
    )
    df.write.format("delta").mode("append").saveAsTable(table)
    return request_id


def _issue_body(incident: dict, d) -> str:
    actions = "\n".join(f"- `{a}`" for a in d.actions) or "(no concrete actions proposed)"
    return (
        f"**Pipeline:** {incident['pipeline_name']}\n"
        f"**Detected via:** {incident.get('detection_source', 'unknown')}\n"
        f"**Result state:** {incident.get('result_state', 'unknown')}\n\n"
        f"**Diagnosis** (confidence {d.confidence:.2f}, {d.provider}):\n{d.diagnosis}\n\n"
        f"**Proposed actions:**\n{actions}\n\n"
        f"**Fix complexity:** {d.fix_complexity}"
    )


def process_new_incidents(settings: Settings, spark, llm: LLMClient, gh: GitHubClient) -> dict:
    table = settings.table("incidents")
    rows = spark.sql(f"""
        SELECT * FROM {table} WHERE status = 'NEW'
        ORDER BY created_at ASC LIMIT {settings.watcher_batch_limit}
    """).collect()

    diagnosed, auto_executed, blocked = 0, 0, 0

    for row in rows:
        incident = row.asDict()
        incident_id = incident["incident_id"]

        try:
            d = diagnose(llm, incident["raw_error"], incident["pipeline_name"])
        except Exception:
            log.exception("diagnosis failed for %s -- leaving as NEW for retry", incident_id)
            continue

        diagnosed += 1
        spark.sql(f"""
            UPDATE {table}
            SET classified_agent = '{d.agent_name}', severity = {d.severity},
                updated_at = current_timestamp()
            WHERE incident_id = '{incident_id}'
        """)

        issue = gh.create_issue(
            title=f"[{d.agent_name}] {incident['pipeline_name']}: {d.fix_suggestion[:80]}",
            body=_issue_body(incident, d),
            incident_id=incident_id,
            severity=d.severity,
        )
        issue_number = issue.get("number") if issue else None

        if needs_approval(d):
            request_id = _write_approval_request(settings, spark, incident, d)
            if issue_number:
                spark.sql(f"""
                    UPDATE {settings.table('approval_requests')}
                    SET github_issue_number = {issue_number} WHERE request_id = '{request_id}'
                """)
                if settings.app_base_url:
                    # The App lists every pending request on its home page.
                    link = settings.app_base_url.rstrip("/")
                    gh.comment(issue_number, f"Request {request_id}: [review and approve]({link})")
            spark.sql(f"""
                UPDATE {table}
                SET status = 'PENDING_APPROVAL', updated_at = current_timestamp()
                WHERE incident_id = '{incident_id}'
            """)
            blocked += 1
            log.info("%s -> PENDING_APPROVAL (%s)", incident_id, request_id)
            continue

        results = executor.execute_all(spark, d.actions)
        succeeded, failed = _write_remediation_log(
            settings, spark, incident_id, "", incident["pipeline_name"],
            d.agent_name, d.fix_complexity, results,
        )
        status = "AUTO_EXECUTED" if failed == 0 else "AUTO_EXECUTED_WITH_FAILURES"
        spark.sql(f"""
            UPDATE {table}
            SET status = '{status}', updated_at = current_timestamp()
            WHERE incident_id = '{incident_id}'
        """)
        if issue_number:
            gh.close_issue(
                issue_number,
                comment=f"Auto-executed: {succeeded} action(s) succeeded, {failed} failed.",
                reason="completed",
            )
        auto_executed += 1
        log.info("%s -> %s (%s ok, %s failed)", incident_id, status, succeeded, failed)

    return {"diagnosed": diagnosed, "auto_executed": auto_executed, "blocked": blocked}


# --------------------------------------------------------------------- step 3

def execute_approved(settings: Settings, spark, gh: GitHubClient) -> int:
    approvals = settings.table("approval_requests")
    incidents = settings.table("incidents")

    rows = spark.sql(f"""
        SELECT * FROM {approvals} WHERE status = 'approved'
        LIMIT {settings.watcher_batch_limit}
    """).collect()

    executed = 0
    for row in rows:
        req = row.asDict()
        request_id = req["request_id"]
        try:
            actions = json.loads(req.get("proposed_actions") or "[]")
        except Exception:
            actions = []

        results = executor.execute_all(spark, actions)
        succeeded, failed = _write_remediation_log(
            settings, spark, req["incident_id"], request_id, req["pipeline_name"],
            req["agent_name"], req["fix_complexity"], results,
        )
        status = "APPROVED_EXECUTED" if failed == 0 else "APPROVED_EXECUTED_WITH_FAILURES"

        spark.sql(f"""
            UPDATE {approvals}
            SET status = '{status}' WHERE request_id = '{request_id}'
        """)
        spark.sql(f"""
            UPDATE {incidents}
            SET status = '{status}', updated_at = current_timestamp()
            WHERE incident_id = '{req['incident_id']}'
        """)
        if req.get("github_issue_number"):
            gh.close_issue(
                int(req["github_issue_number"]),
                comment=f"Approved and executed: {succeeded} action(s) succeeded, {failed} failed.",
                reason="completed",
            )
        executed += 1
        log.info("%s -> %s", request_id, status)

    return executed


# --------------------------------------------------------------------- step 4

def close_rejected(settings: Settings, spark, gh: GitHubClient) -> int:
    approvals = settings.table("approval_requests")
    incidents = settings.table("incidents")

    rows = spark.sql(f"""
        SELECT * FROM {approvals} WHERE status = 'rejected_pending_close'
        LIMIT {settings.watcher_batch_limit}
    """).collect()

    closed = 0
    for row in rows:
        req = row.asDict()
        spark.sql(f"""
            UPDATE {approvals} SET status = 'REJECTED' WHERE request_id = '{req['request_id']}'
        """)
        spark.sql(f"""
            UPDATE {incidents}
            SET status = 'REJECTED', updated_at = current_timestamp()
            WHERE incident_id = '{req['incident_id']}'
        """)
        if req.get("github_issue_number"):
            gh.close_issue(
                int(req["github_issue_number"]),
                comment=f"Rejected: {req.get('reviewer_notes') or 'no notes provided'}",
                reason="not_planned",
            )
        closed += 1
    return closed


# ------------------------------------------------------------------- run_once

def run_once(settings: Settings, spark=None) -> dict:
    spark = spark or _get_spark()
    llm = LLMClient.from_settings(settings)
    gh = GitHubClient.from_settings(settings)

    expired = expire_stale_requests(settings, spark)
    diag_stats = process_new_incidents(settings, spark, llm, gh)
    approved = execute_approved(settings, spark, gh)
    rejected = close_rejected(settings, spark, gh)

    summary = {
        "expired": expired,
        **diag_stats,
        "approved_executed": approved,
        "rejected_closed": rejected,
    }
    log.info("reconciler pass complete: %s", summary)
    return summary