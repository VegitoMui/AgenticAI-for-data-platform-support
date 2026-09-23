"""
Approvals App.

Runs under its own service principal: the Databricks Apps runtime provisions
one because a SQL warehouse resource is declared for this app, and injects
its OAuth credentials into the environment. databricks.sdk Config() picks
those up automatically -- no PAT, no secret scope, no shared credential.

Talks to the agentic_ai tables over Databricks SQL, since Apps have no
Spark session.

Layout: a queue of pending requests on the left, the selected request in
detail on the right (?id=APR-...), and a short list of recent decisions.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from urllib.parse import urlencode

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

CATALOG = os.environ.get("AGENTIC_CATALOG", "databricks_ws")
SCHEMA = os.environ.get("AGENTIC_SCHEMA", "agentic_ai_dev")
ENVIRONMENT = "dev" if SCHEMA.endswith("_dev") else "prod"

_cfg = None


def table(name: str) -> str:
    return f"{CATALOG}.{SCHEMA}.{name}"


def get_connection():
    # Imported lazily so the module can be loaded without Databricks
    # credentials (e.g. for a local template preview).
    global _cfg
    from databricks import sql
    from databricks.sdk.core import Config

    if _cfg is None:
        _cfg = Config()
    cfg = _cfg
    return sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{os.environ['DATABRICKS_WAREHOUSE_ID']}",
        credentials_provider=lambda: cfg.authenticate,
    )


def run_query(query: str, params: tuple = ()) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def run_execute(query: str, params: tuple = ()) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)


# ------------------------------------------------------------------ shaping

# Mirrors remediation/executor.py so the reviewer sees exactly which proposed
# actions the reconciler will run and which it will skip.
_SQL_PREFIXES = (
    "OPTIMIZE", "VACUUM", "ANALYZE", "ALTER TABLE", "GRANT",
    "CREATE TABLE IF NOT EXISTS", "MSCK", "REPAIR", "REFRESH",
)
_CLUSTER_CONFIGS = {
    "spark.executor.memory", "spark.driver.memory", "spark.executor.cores",
    "spark.executor.instances", "spark.driver.cores",
    "spark.yarn.executor.memoryOverhead", "spark.memory.fraction",
    "spark.memory.storageFraction", "spark.sql.adaptive.enabled",
    "spark.sql.adaptive.coalescePartitions.enabled",
    "spark.sql.adaptive.skewJoin.enabled",
    "spark.sql.adaptive.skewJoin.skewedPartitionFactor",
    "spark.sql.autoBroadcastJoinThreshold",
    "spark.databricks.optimizer.dynamicFilePruning",
    "spark.sql.streaming.checkpointLocation",
    "spark.sql.streaming.stopActiveRunOnRestart",
}
_CONF_PATTERN = re.compile(r"spark\.conf\.set\(['\"](.+?)['\"]")


def describe_action(text: str) -> dict:
    """Return how the executor will treat one proposed action."""
    stripped = text.strip()
    upper = stripped.upper()
    if any(upper.startswith(p) for p in _SQL_PREFIXES):
        return {"text": stripped, "kind": "SQL", "runs": True, "note": "Runs as SQL"}
    if stripped.startswith("spark.conf.set"):
        match = _CONF_PATTERN.search(stripped)
        key = match.group(1) if match else ""
        if key in _CLUSTER_CONFIGS:
            return {"text": stripped, "kind": "Config", "runs": False,
                    "note": "Skipped: cluster-level setting, change it on the cluster"}
        return {"text": stripped, "kind": "Config", "runs": True, "note": "Sets a Spark config"}
    return {"text": stripped, "kind": "Advice", "runs": False,
            "note": "Not run: guidance for a person"}


def severity_label(sev) -> tuple[str, str]:
    try:
        n = int(sev)
    except (TypeError, ValueError):
        return "Unknown", "unknown"
    if n >= 9:
        return "Critical", "critical"
    if n >= 7:
        return "High", "high"
    if n >= 4:
        return "Medium", "medium"
    return "Low", "low"


def _as_utc(dt):
    if dt is None:
        return None
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def relative(dt, future: bool) -> str:
    dt = _as_utc(dt)
    if dt is None:
        return ""
    seconds = (dt - datetime.now(timezone.utc)).total_seconds()
    if not future:
        seconds = -seconds
    if seconds <= 0:
        return "now" if future else "just now"
    minutes = int(seconds // 60)
    if minutes < 60:
        span = f"{minutes} min"
    elif minutes < 48 * 60:
        span = f"{minutes // 60} h"
    else:
        span = f"{minutes // (60 * 24)} days"
    return f"in {span}" if future else f"{span} ago"


def hours_left(dt) -> float:
    dt = _as_utc(dt)
    if dt is None:
        return 999.0
    return (dt - datetime.now(timezone.utc)).total_seconds() / 3600


def shape_request(r: dict) -> dict:
    try:
        raw_actions = json.loads(r.get("proposed_actions") or "[]")
    except (TypeError, ValueError):
        raw_actions = []
    actions = [describe_action(str(a)) for a in raw_actions if str(a).strip()]
    label, tone = severity_label(r.get("severity"))
    left = hours_left(r.get("expires_at"))
    return {
        **r,
        "actions": actions,
        "runnable": sum(1 for a in actions if a["runs"]),
        "severity_label": label,
        "severity_tone": tone,
        "requested_rel": relative(r.get("requested_at"), future=False),
        "expires_rel": relative(r.get("expires_at"), future=True),
        "expiring_soon": left < 12,
        "agent_title": (r.get("agent_name") or "unknown").capitalize(),
    }


DECISION_TEXT = {
    "approved": "Approved, waiting to run",
    "APPROVED_EXECUTED": "Approved and run",
    "APPROVED_EXECUTED_WITH_FAILURES": "Approved, some actions failed",
    "rejected_pending_close": "Rejected, closing",
    "REJECTED": "Rejected",
    "EXPIRED": "Expired without a decision",
}


# ------------------------------------------------------------------ routes

@app.get("/", response_class=HTMLResponse)
def index(request: Request, id: str | None = None, done: str | None = None, ref: str | None = None):
    pending = [shape_request(r) for r in run_query(f"""
        SELECT request_id, incident_id, pipeline_name, agent_name, severity,
               diagnosis, proposed_actions, fix_complexity, fix_suggestion,
               requires_human, requested_at, expires_at, github_issue_number
        FROM {table('approval_requests')}
        WHERE status = 'pending'
        ORDER BY severity DESC, requested_at ASC
    """)]

    recent = run_query(f"""
        SELECT request_id, pipeline_name, status, reviewed_by, reviewed_at
        FROM {table('approval_requests')}
        WHERE status <> 'pending' AND reviewed_at IS NOT NULL
        ORDER BY reviewed_at DESC
        LIMIT 8
    """)
    for r in recent:
        r["decision_text"] = DECISION_TEXT.get(r.get("status"), r.get("status"))
        r["decided_rel"] = relative(r.get("reviewed_at"), future=False)
        r["positive"] = str(r.get("status", "")).lower().startswith("approved")

    selected = next((p for p in pending if p["request_id"] == id), None)
    if selected is None and pending:
        selected = pending[0]

    notice = None
    if done in ("approved", "rejected") and ref:
        notice = {
            "approved": f"Approved {ref}. The reconciler runs it within 2 minutes.",
            "rejected": f"Rejected {ref}. Its ticket will be closed with your note.",
        }[done]

    return templates.TemplateResponse(request=request, name="list.html", context={
        "pending": pending,
        "selected": selected,
        "recent": recent,
        "notice": notice,
        "environment": ENVIRONMENT,
        "reviewer": request.headers.get("X-Forwarded-Email", ""),
        "expiring_count": sum(1 for p in pending if p["expiring_soon"]),
    })


def _reviewer(request: Request) -> str:
    """The signed-in user's email, forwarded by the Apps proxy."""
    return request.headers.get("X-Forwarded-Email", "app-user")


@app.post("/approve")
def approve(request: Request, request_id: str = Form(...)):
    run_execute(
        f"UPDATE {table('approval_requests')} "
        f"SET status = 'approved', reviewed_at = current_timestamp(), reviewed_by = ? "
        f"WHERE request_id = ? AND status = 'pending'",
        (_reviewer(request), request_id),
    )
    return RedirectResponse("/?" + urlencode({"done": "approved", "ref": request_id}), status_code=303)


@app.post("/reject")
def reject(request: Request, request_id: str = Form(...), notes: str = Form("")):
    run_execute(
        f"UPDATE {table('approval_requests')} "
        f"SET status = 'rejected_pending_close', reviewed_at = current_timestamp(), "
        f"reviewed_by = ?, reviewer_notes = ? "
        f"WHERE request_id = ? AND status = 'pending'",
        (_reviewer(request), notes.strip() or "No reason given", request_id),
    )
    return RedirectResponse("/?" + urlencode({"done": "rejected", "ref": request_id}), status_code=303)