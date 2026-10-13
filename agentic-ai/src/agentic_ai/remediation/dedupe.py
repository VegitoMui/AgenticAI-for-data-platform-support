"""
Repeat-failure collapsing.

A job that keeps failing the same way should produce one incident, not one
per run. Before an incident is diagnosed, the reconciler fingerprints it and
looks for an OPEN incident with the same fingerprint:

  found      the new incident becomes a DUPLICATE linked to the open one; the
             open incident's occurrence count and last-seen time go up, and a
             comment is added to its GitHub issue. No diagnosis, no LLM call,
             no email, no memory row.
  not found  diagnosed as usual, and its fingerprint stored so later repeats
             can find it.

Fingerprint = job name + each failed task's key and error message, with
run-specific noise (run ids, UUIDs, timestamps) removed. The job NAME is used
rather than the job id, because a redeployed or recreated job keeps its name
but gets a new id. The stack trace is left out: its tail varies with JVM
frames while the error message stays the same.

An incident is OPEN while it waits for a decision (PENDING_APPROVAL). Once it
is approved, rejected, expired or auto-fixed, a repeat is a new incident --
a failure that returns after a fix is news, and memory will recall the fix.

Incidents with no task error text are never collapsed: without the message
there is no evidence two failures are the same.
"""

from __future__ import annotations

import hashlib
import logging
import re

from agentic_ai.config import Settings
from agentic_ai.telemetry.run_context import RunContext

log = logging.getLogger(__name__)

OPEN_STATUSES = ("PENDING_APPROVAL",)
MAX_ERROR_CHARS = 400

_NOISE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"   # uuid
    r"|\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\S*"                          # timestamp
    r"|\b\d{6,}\b"                                                         # run / job ids
    r"|\b[0-9a-f]{16,}\b",                                                 # long hex ids
    re.I,
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]+$")


def _normalise(text: str) -> str:
    text = _NOISE.sub("#", (text or "").lower())
    return " ".join(text.split())[:MAX_ERROR_CHARS]


def fingerprint(run: RunContext, pipeline_name: str) -> str | None:
    """Stable id for 'this job failing this way', or None if there is no error text."""
    parts = sorted(
        f"{t.task_key}|{_normalise(t.error)}" for t in run.task_errors if (t.error or "").strip()
    )
    if not parts:
        return None
    basis = "\n".join([_normalise(pipeline_name or run.job_name)] + parts)
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]


def find_open_parent(settings: Settings, spark, fp: str, exclude_incident: str) -> dict | None:
    """The oldest open incident with this fingerprint, with its GitHub issue number."""
    if not fp or not re.fullmatch(r"[0-9a-f]{32}", fp) or not _SAFE_ID.match(exclude_incident or ""):
        return None
    statuses = ", ".join(f"'{s}'" for s in OPEN_STATUSES)
    rows = spark.sql(f"""
        SELECT i.incident_id, COALESCE(i.occurrence_count, 1) AS occurrence_count,
               COALESCE(i.github_issue_number, a.github_issue_number) AS github_issue_number
        FROM {settings.table('incidents')} i
        LEFT JOIN {settings.table('approval_requests')} a
          ON a.incident_id = i.incident_id AND a.github_issue_number IS NOT NULL
        WHERE i.fingerprint = '{fp}' AND i.status IN ({statuses})
          AND i.incident_id <> '{exclude_incident}'
        ORDER BY i.created_at ASC
        LIMIT 1
    """).collect()
    if not rows:
        return None
    r = rows[0]
    get = r.asDict() if hasattr(r, "asDict") else dict(r)
    return {"incident_id": get["incident_id"], "occurrence_count": int(get["occurrence_count"] or 1),
            "github_issue_number": get.get("github_issue_number")}


def record_repeat(settings: Settings, spark, gh, incident: dict, fp: str, parent: dict) -> int:
    """Link `incident` to `parent` as a duplicate. Returns the parent's new occurrence count."""
    incidents = settings.table("incidents")
    child, parent_id = incident["incident_id"], parent["incident_id"]
    if not (_SAFE_ID.match(child) and _SAFE_ID.match(parent_id)):
        raise ValueError(f"unsafe incident id: {child!r} / {parent_id!r}")

    spark.sql(f"""
        UPDATE {incidents}
        SET status = 'DUPLICATE', duplicate_of = '{parent_id}', fingerprint = '{fp}',
            updated_at = current_timestamp()
        WHERE incident_id = '{child}'
    """)
    spark.sql(f"""
        UPDATE {incidents}
        SET occurrence_count = COALESCE(occurrence_count, 1) + 1,
            last_seen_at = current_timestamp(), updated_at = current_timestamp()
        WHERE incident_id = '{parent_id}'
    """)
    count = parent["occurrence_count"] + 1

    if parent.get("github_issue_number"):
        when = incident.get("run_end_time") or "just now"
        gh.comment(
            int(parent["github_issue_number"]),
            f"Failed again: run {incident.get('run_id') or 'unknown'} ({when}), recorded as "
            f"{child}. This failure has now happened {count} times while waiting for a decision. "
            "No new diagnosis was made; the proposal above still applies.",
        )
    log.info("%s is a repeat of %s (%s occurrences)", child, parent_id, count)
    return count