"""
Approvals App.

Runs under its own service principal: the Databricks Apps runtime provisions
one because a SQL warehouse resource is declared for this app, and injects
its OAuth credentials into the environment. databricks.sdk Config() picks
those up automatically -- no PAT, no secret scope, no shared credential.

Talks to the agentic_ai tables over Databricks SQL, since Apps have no
Spark session.
"""

from __future__ import annotations

import os

from databricks import sql
from databricks.sdk.core import Config
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

CATALOG = os.environ.get("AGENTIC_CATALOG", "databricks_ws")
SCHEMA = os.environ.get("AGENTIC_SCHEMA", "agentic_ai_dev")
WAREHOUSE_ID = os.environ["DATABRICKS_WAREHOUSE_ID"]

# Reads the app's own service-principal credentials from the environment
# the Apps runtime sets up.
_cfg = Config()


def table(name: str) -> str:
    return f"{CATALOG}.{SCHEMA}.{name}"


def get_connection():
    return sql.connect(
        server_hostname=_cfg.host,
        http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
        credentials_provider=lambda: _cfg.authenticate,
    )


def run_query(query: str, params: tuple = ()) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def run_execute(query: str, params: tuple = ()) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)


def _reviewer(request: Request) -> str:
    """The signed-in user's email, forwarded by the Apps proxy. Falls back
    to a fixed label if the header is absent (e.g. local testing)."""
    return request.headers.get("X-Forwarded-Email", "app-user")


@app.get("/", response_class=HTMLResponse)
def list_pending(request: Request):
    rows = run_query(f"""
        SELECT request_id, incident_id, pipeline_name, agent_name, severity,
               diagnosis, proposed_actions, fix_complexity, fix_suggestion,
               requested_at, expires_at
        FROM {table('approval_requests')}
        WHERE status = 'pending'
        ORDER BY requested_at DESC
    """)
    return templates.TemplateResponse("list.html", {"request": request, "requests": rows})


@app.post("/approve")
def approve(request: Request, request_id: str = Form(...)):
    run_execute(
        f"UPDATE {table('approval_requests')} "
        f"SET status = 'approved', reviewed_at = current_timestamp(), reviewed_by = ? "
        f"WHERE request_id = ? AND status = 'pending'",
        (_reviewer(request), request_id),
    )
    return RedirectResponse("/", status_code=303)


@app.post("/reject")
def reject(request: Request, request_id: str = Form(...), notes: str = Form("")):
    run_execute(
        f"UPDATE {table('approval_requests')} "
        f"SET status = 'rejected_pending_close', reviewed_at = current_timestamp(), "
        f"reviewed_by = ?, reviewer_notes = ? "
        f"WHERE request_id = ? AND status = 'pending'",
        (_reviewer(request), notes, request_id),
    )
    return RedirectResponse("/", status_code=303)