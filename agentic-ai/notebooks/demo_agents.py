# Databricks notebook source
# MAGIC %md
# MAGIC # Demo: Agentic AI for Data Platform Support
# MAGIC
# MAGIC Run top to bottom. Nothing has to be typed during the demo.
# MAGIC
# MAGIC | Part | When | What it does |
# MAGIC |---|---|---|
# MAGIC | 0. Setup | Always first | Names, helpers |
# MAGIC | 1. Pre-flight | 30 min before | LLM, App, jobs and tables are ready |
# MAGIC | 2. Reset | 30 min before | Closes open requests so the demo starts clean |
# MAGIC | 3. Pre-bake | 15 min before | Fails probes, builds maintenance table, runs watcher and scanner |
# MAGIC | Act 1 | Live | Reconciler diagnoses the failures; show board, traces, App, email, GitHub |
# MAGIC | Act 2 | Live | Same failure again: collapsed into one incident, no new diagnosis |
# MAGIC | Act 3 | Live | Maintenance fixed on its own: small files compacted |
# MAGIC | Act 4 | Live | Approvals: 3 requests with steps run after approval, 2 advice-only rejected |
# MAGIC | Act 5 | Live | Memory: what the agents remember about past incidents |
# MAGIC | 9. Cleanup | After | Drops demo tables, stops the App |
# MAGIC
# MAGIC The watcher and reconciler jobs stay paused; this notebook starts single runs of them.
# MAGIC
# MAGIC Requests in the App: 2 advice-only (v4 unreachable API, v5 column typo: the fix is outside
# MAGIC Databricks or in code) and 3 with steps the agent runs once approved (v6 schema mismatch:
# MAGIC ADD COLUMNS, v7 table feature not enabled: SET TBLPROPERTIES, duplicate table: DROP TABLE).

# COMMAND ----------

# MAGIC %pip install openai requests --quiet
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Setup

# COMMAND ----------

import os
import sys
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
from databricks.sdk.service.workspace import ImportFormat, Language

CATALOG, SCHEMA, SCOPE = "databricks_ws", "agentic_ai_dev", "agentic-ai"
S = f"{CATALOG}.{SCHEMA}"
PROBE_SCHEMA = f"{CATALOG}.agentic_probe"
SMALL_FILES_TABLE = f"{PROBE_SCHEMA}.sales_events"         # fresh table, compacted in Act 3
VACUUM_TABLE = f"{PROBE_SCHEMA}.orders_small_files"        # OPTIMIZEd on 13 Oct, VACUUM due
DUP_ORIGINAL, DUP_COPY = f"{PROBE_SCHEMA}.customers", f"{PROBE_SCHEMA}.customers_bkp"
SCHEMA_TABLE = f"{PROBE_SCHEMA}.orders_daily"              # v6 appends a column it does not have
FEATURE_TABLE = f"{PROBE_SCHEMA}.order_status"             # v7 needs a table feature it does not have
APP = "agentic-ai-approvals-dev"

WATCHER = "[dev] agentic-ai watcher"
RECONCILER = "[dev] agentic-ai reconciler"
SCANNER = "[dev] agentic-ai table scanner"
BASE_PARAMS = [f"--catalog={CATALOG}", f"--schema={SCHEMA}", f"--secret-scope={SCOPE}"]

# Only these jobs (and table scans) are shown on the demo board; teammates' jobs are left out.
PROBE_PREFIX = "agentic-ai probe"
SINCE_HOURS = 6     # demo queries look back this far; pre-run on the day of the demo

w = WorkspaceClient()
me = w.current_user.me().user_name

# probe key -> (job name, task key, notebook source). Jobs are created only if missing.
#   v4, v5: advice only (fix is outside Databricks / in code)
#   v6, v7: the agent proposes SQL that runs once a person approves it
PROBES = {
    "v4": ("agentic-ai probe v4 (source unreachable)", "call_source_api",
           "import urllib.request\n"
           "urllib.request.urlopen('https://orders-api.agentic-probe.invalid/v1/orders', timeout=10)\n"),
    "v5": ("agentic-ai probe v5 (unresolved column)", "query_bad_column",
           f"spark.sql('SELECT custmer_id FROM {S}.incidents').show()\n"),
    "v6": ("agentic-ai probe v6 (schema mismatch on write)", "append_daily_orders",
           "df = spark.range(10).selectExpr('id AS order_id', 'rand() * 100 AS amount', "
           "'rand() * 5 AS discount')\n"
           f"df.write.mode('append').saveAsTable('{SCHEMA_TABLE}')\n"),
    "v7": ("agentic-ai probe v7 (table feature not enabled)", "set_status_default",
           f"spark.sql(\"ALTER TABLE {FEATURE_TABLE} ALTER COLUMN status SET DEFAULT 'new'\")\n"),
}


def job_id(name: str) -> int | None:
    """Exact name first; otherwise a job whose name ends with it (bundle dev mode adds a prefix)."""
    found = [j for j in w.jobs.list(name=name) if j.settings and j.settings.name == name]
    if not found:
        found = [j for j in w.jobs.list() if j.settings and (j.settings.name or "").endswith(name)]
    if len(found) > 1:
        print(f"note: {len(found)} jobs match {name!r}, using {found[0].settings.name!r}")
    return found[0].job_id if found else None


def make_probe_tables() -> None:
    """Fresh tables for v6 and v7, so they fail the same way every time."""
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {PROBE_SCHEMA}")
    spark.sql(f"CREATE OR REPLACE TABLE {SCHEMA_TABLE} (order_id BIGINT, amount DOUBLE)")
    spark.sql(f"INSERT INTO {SCHEMA_TABLE} VALUES (1, 10.0), (2, 20.0)")
    # Pinned to the oldest protocol, so the column-defaults feature is not enabled.
    spark.sql(f"DROP TABLE IF EXISTS {FEATURE_TABLE}")
    spark.sql(f"""
        CREATE TABLE {FEATURE_TABLE} (order_id BIGINT, status STRING)
        TBLPROPERTIES ('delta.minReaderVersion' = '1', 'delta.minWriterVersion' = '2',
                       'delta.enableDeletionVectors' = 'false')
    """)
    d = spark.sql(f"DESCRIBE DETAIL {FEATURE_TABLE}").first().asDict()
    feats = d.get("tableFeatures") or []
    print(f"{FEATURE_TABLE}: protocol {d.get('minReaderVersion')}/{d.get('minWriterVersion')}, "
          f"features {feats}")
    if any("columnDefaults" in f for f in feats):
        print("WARNING: column defaults already enabled, so v7 will not fail")


def ensure_probe(key: str) -> int:
    """Reuse the probe job by name, or create it with its notebook."""
    name, task_key, source = PROBES[key]
    existing = job_id(name)
    if existing:
        return existing
    path = f"/Users/{me}/agentic_ai_probe_{key}"
    w.workspace.upload(path=path, content=("# Databricks notebook source\n" + source).encode(),
                       format=ImportFormat.SOURCE, language=Language.PYTHON, overwrite=True)
    job = w.jobs.create(name=name, tasks=[jobs.Task(
        task_key=task_key, notebook_task=jobs.NotebookTask(notebook_path=path))])
    print(f"created {name} -> job {job.job_id}")
    return job.job_id


def start(name: str, python_params: list[str] | None = None) -> int:
    jid = job_id(name)
    if jid is None:
        raise RuntimeError(f"job not found: {name}")
    run = w.jobs.run_now(job_id=jid, python_params=python_params) if python_params \
        else w.jobs.run_now(job_id=jid)
    print(f"started {name} -> run {run.run_id}")
    return run.run_id


def wait(run_ids: list[int], timeout_min: int = 25) -> None:
    """Poll until every run has finished, then print how each ended."""
    pending, deadline, started = set(run_ids), time.time() + timeout_min * 60, time.time()
    while pending and time.time() < deadline:
        for rid in list(pending):
            state = w.jobs.get_run(rid).state
            if state.life_cycle_state.value in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
                pending.discard(rid)
        if pending:
            print(f"  {len(pending)} run(s) still going ({time.time() - started:,.0f}s)", end="\r")
            time.sleep(10)
    print()
    for rid in run_ids:
        run = w.jobs.get_run(rid)
        result = run.state.result_state.value if run.state.result_state else run.state.life_cycle_state.value
        print(f"  run {rid}: {result}  {run.run_page_url}")


def run_job(name: str, python_params: list[str] | None = None) -> None:
    wait([start(name, python_params)])


def run_watcher() -> None:
    # jobs_api reads the Jobs API directly: no wait for system tables to catch up.
    run_job(WATCHER, BASE_PARAMS + ["--detection-source=jobs_api"])


def board_filter() -> str:
    return (f"(pipeline_name LIKE '{PROBE_PREFIX}%' OR detection_source = 'table_scan') "
            f"AND created_at >= current_timestamp() - INTERVAL {SINCE_HOURS} HOURS")


def table_files(table: str) -> str:
    d = spark.sql(f"DESCRIBE DETAIL {table}").first()
    avg = d.sizeInBytes / max(d.numFiles, 1) / 1024**2
    return f"{d.numFiles:,} files, {d.sizeInBytes / 1024**2:,.0f} MB, average {avg:,.2f} MB per file"


print("ready:", S, "| user:", me)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Pre-flight (30 minutes before)

# COMMAND ----------

# 1a. LLM: Azure AI Foundry should PASS. Fallbacks failing is fine as long as one passes.
sys.path.append(os.path.abspath("../src"))
from agentic_ai.config import Settings  # noqa: E402
from agentic_ai.llm.client import LLMClient  # noqa: E402

settings = Settings(catalog=CATALOG, schema=SCHEMA, secret_scope=SCOPE)
for r in LLMClient.from_settings(settings).health_check():
    print(f"  {r['status'].upper():5} {r['name']}  {r.get('latency_sec', '')}{r.get('error', '')}")

# COMMAND ----------

# 1b. App: starts it if stopped (takes 2-4 minutes). Open the URL in a browser tab.
app = w.apps.get(APP)
state = app.compute_status.state.value if app.compute_status and app.compute_status.state else "UNKNOWN"
print("App state:", state, "|", app.url)
if state != "ACTIVE":
    w.apps.start(APP)
    print("starting the App; re-run this cell in a few minutes until it says ACTIVE")

# COMMAND ----------

# 1c. Jobs exist and are paused; probe jobs exist (created if missing).
for name in (WATCHER, RECONCILER, SCANNER):
    jid = job_id(name)
    sched = w.jobs.get(jid).settings.schedule if jid else None
    paused = sched.pause_status.value if sched and sched.pause_status else "no schedule"
    print(f"  {name}: job {jid} ({paused})")
for key in PROBES:
    print(f"  probe {key}: job {ensure_probe(key)}")

# COMMAND ----------

# 1d. What is open right now. 'pending' requests and NEW incidents should be 0 after the reset.
display(spark.sql(f"""
    SELECT 'approval_requests' AS tbl, status, count(*) AS n FROM {S}.approval_requests
    WHERE status IN ('pending', 'approved', 'rejected_pending_close') GROUP BY status
    UNION ALL
    SELECT 'incidents', status, count(*) FROM {S}.incidents
    WHERE status IN ('NEW', 'PENDING_APPROVAL') GROUP BY status
"""))

# COMMAND ----------

# 1e. Is VACUUM due on the table compacted on 13 Oct? Due once the OPTIMIZE is over 168 hours old
#     and no VACUUM has run since. If due, the scanner raises VACUUM OVERDUE and it is fixed in Act 3.
hist = spark.sql(f"DESCRIBE HISTORY {VACUUM_TABLE}").select("timestamp", "operation").collect()
last_vacuum = max((h.timestamp for h in hist if h.operation == "VACUUM END"), default=None)
removals = [h.timestamp for h in hist if h.operation in ("OPTIMIZE", "DELETE", "MERGE", "UPDATE")
            and (last_vacuum is None or h.timestamp > last_vacuum)]
age_h = (spark.sql("SELECT current_timestamp() AS t").first().t - min(removals)).total_seconds() / 3600 \
    if removals else 0
print(f"{VACUUM_TABLE}: {table_files(VACUUM_TABLE)}")
print(f"last VACUUM: {last_vacuum} | oldest removal since: {min(removals) if removals else None}"
      f" ({age_h:,.0f} h ago) -> VACUUM {'DUE' if age_h > 168 else 'not due yet'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Reset (30 minutes before)
# MAGIC
# MAGIC A new failure that matches an incident still waiting for a decision is collapsed into it,
# MAGIC so the demo failures would not get their own diagnosis. This closes every open request,
# MAGIC parks backlog incidents, and moves the watcher's start point to now.

# COMMAND ----------

# 2a. Mark every pending request as rejected; the reconciler closes them (incident, GitHub issue, memory).
spark.sql(f"""
    UPDATE {S}.approval_requests
    SET status = 'rejected_pending_close', reviewed_by = '{me}', reviewed_at = current_timestamp(),
        reviewer_notes = 'Closed before the demo'
    WHERE status = 'pending'
""")
# Backlog NEW incidents (teammates' jobs, old probes) would be diagnosed in the middle of the demo.
# They are parked, not deleted; 9c puts them back.
spark.sql(f"""
    UPDATE {S}.incidents SET status = 'PARKED_FOR_DEMO', updated_at = current_timestamp()
    WHERE status = 'NEW'
""")
# Only failures from now on are picked up by the watcher.
spark.sql(f"UPDATE {S}.watcher_state SET last_run_end_time = current_timestamp()")
run_job(RECONCILER)

# COMMAND ----------

# 2b. Open incidents left without a request (rare) are closed directly. Then re-run 1d: all zero.
spark.sql(f"""
    UPDATE {S}.incidents SET status = 'REJECTED', updated_at = current_timestamp()
    WHERE status = 'PENDING_APPROVAL'
      AND incident_id NOT IN (SELECT incident_id FROM {S}.approval_requests WHERE status = 'pending')
""")
display(spark.sql(f"""
    SELECT status, count(*) AS n FROM {S}.incidents
    WHERE status IN ('NEW', 'PENDING_APPROVAL') GROUP BY status
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Pre-bake (15 minutes before)
# MAGIC
# MAGIC Fails the probe jobs, builds a table with 1,000 small files, makes sure the duplicate pair
# MAGIC exists, then runs the watcher and the scanner. Afterwards the incidents are NEW and waiting
# MAGIC for the reconciler, which runs live in Act 1. Takes about 8 minutes.

# COMMAND ----------

# 3a. Fresh tables for v6/v7, then fail the probes: v5, v6, v7 once and v4 three times (for Act 2).
make_probe_tables()
runs = [start(PROBES[k][0]) for k in ("v5", "v6", "v7")]
for _ in range(3):
    runs.append(start(PROBES["v4"][0]))
    time.sleep(5)

# 3b. While they run: a fresh table with 1,000 small files (Act 3).
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {PROBE_SCHEMA}")
spark.sql(f"DROP TABLE IF EXISTS {SMALL_FILES_TABLE}")
spark.sql(f"""
    CREATE TABLE {SMALL_FILES_TABLE} (event_id BIGINT, event_ref STRING, amount DOUBLE)
    TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'false', 'delta.autoOptimize.autoCompact' = 'false')
""")
(spark.range(24_000_000)
      .selectExpr("id AS event_id", "uuid() AS event_ref", "rand() * 1000 AS amount")
      .repartition(1000)
      .write.mode("append").saveAsTable(SMALL_FILES_TABLE))
BEFORE = table_files(SMALL_FILES_TABLE)
print("small-files table before:", BEFORE)

# 3c. The duplicate pair (Act 4): an original and a copy with the same columns and rows.
if not spark.catalog.tableExists(DUP_ORIGINAL):
    (spark.range(50_000)
          .selectExpr("id AS customer_id", "concat('customer_', id) AS name",
                      "concat('city_', id % 40) AS city")
          .write.saveAsTable(DUP_ORIGINAL))
if not spark.catalog.tableExists(DUP_COPY):
    spark.sql(f"CREATE TABLE {DUP_COPY} AS SELECT * FROM {DUP_ORIGINAL}")
print("duplicate pair:", DUP_ORIGINAL, spark.table(DUP_ORIGINAL).count(), "rows |",
      DUP_COPY, spark.table(DUP_COPY).count(), "rows")

wait(runs)

# COMMAND ----------

# 3d. Watcher (raises an incident per failed run) and scanner (raises maintenance findings).
wait([start(WATCHER, BASE_PARAMS + ["--detection-source=jobs_api"]), start(SCANNER)])

# COMMAND ----------

# 3e. Expect: v4 x3, v5, v6, v7 from the watcher; SMALL_FILES, DUPLICATE_TABLE (and VACUUM_OVERDUE
#     if 1e said DUE) from the scanner. All NEW. If a probe is missing, run 3d again.
display(spark.sql(f"""
    SELECT incident_id, pipeline_name, task_key, detection_source, status, created_at
    FROM {S}.incidents WHERE {board_filter()} ORDER BY created_at
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Act 1. Failure to diagnosis (live)
# MAGIC
# MAGIC Start the reconciler, then talk through the architecture while it runs (3 to 5 minutes).
# MAGIC For each incident it reads the run, routes it to a specialist agent, the agent picks a
# MAGIC sub-agent and runs tools, recalls similar past incidents, and proposes a fix. Safe fixes run
# MAGIC on their own; anything else waits for approval in the App, with an email and a GitHub issue.

# COMMAND ----------

run_job(RECONCILER)

# COMMAND ----------

# Incident board: one row per demo incident.
display(spark.sql(f"""
    SELECT incident_id, pipeline_name, task_key, classified_agent AS agent, severity, status,
           COALESCE(occurrence_count, 1) AS times_failed, duplicate_of, github_issue_number AS issue,
           created_at
    FROM {S}.incidents WHERE {board_filter()}
    ORDER BY created_at
"""))

# COMMAND ----------

# How each agent reached its answer: router -> specialist -> sub-agent -> tools -> memory -> proposal.
def show_trace(incident_id: str, width: int = 240) -> None:
    rows = spark.sql(f"""
        SELECT step_number, node_name, sub_agent_selected, tools_run, duration_sec, detail
        FROM {S}.subagent_execution_log WHERE incident_id = '{incident_id}' ORDER BY step_number
    """).collect()
    head = spark.sql(f"SELECT pipeline_name FROM {S}.incidents WHERE incident_id = '{incident_id}'").first()
    print(f"=== {incident_id}  {head.pipeline_name if head else ''}")
    for r in rows:
        extra = " | ".join(x for x in (r.sub_agent_selected, r.tools_run) if x)
        print(f"  {r.step_number:>2}. {r.node_name:<22} {r.duration_sec or 0:6.1f}s  {extra}")
        print(f"      {(r.detail or '').replace(chr(10), ' ')[:width]}")
    print()


demo = spark.sql(f"""
    SELECT incident_id FROM {S}.incidents
    WHERE {board_filter()} AND status <> 'DUPLICATE' ORDER BY created_at
""").collect()
for row in demo:
    show_trace(row.incident_id)

# COMMAND ----------

# What is waiting for a person (same as the App home page).
display(spark.sql(f"""
    SELECT request_id, incident_id, agent_name, priority, severity, fix_complexity,
           fix_suggestion, proposed_actions, github_issue_number AS issue, expires_at
    FROM {S}.approval_requests
    WHERE status = 'pending' AND incident_id IN (SELECT incident_id FROM {S}.incidents WHERE {board_filter()})
    ORDER BY severity DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Act 2. Noise control (live)
# MAGIC
# MAGIC v4 already failed three times and shows as one incident. Fail it once more: the reconciler
# MAGIC recognises the same failure, adds to the count and comments on the existing GitHub issue.
# MAGIC No new diagnosis, no LLM call, no new email. About 4 minutes.

# COMMAND ----------

wait([start(PROBES["v4"][0])])
run_watcher()
run_job(RECONCILER)
display(spark.sql(f"""
    SELECT incident_id, status, COALESCE(occurrence_count, 1) AS times_failed, duplicate_of,
           github_issue_number AS issue, last_seen_at
    FROM {S}.incidents
    WHERE pipeline_name LIKE '{PROBE_PREFIX} v4%'
      AND created_at >= current_timestamp() - INTERVAL {SINCE_HOURS} HOURS
    ORDER BY created_at
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Act 3. Self-healing maintenance (already done in Act 1)
# MAGIC
# MAGIC No job failed here: the scanner found the problem by looking at the table. OPTIMIZE (and
# MAGIC VACUUM with the 168-hour retention) are safe, so the reconciler ran them without asking.

# COMMAND ----------

print("small-files table before:", globals().get("BEFORE", "1,000 files (see 3b)"))
print("small-files table now:   ", table_files(SMALL_FILES_TABLE))
print("VACUUM table now:        ", table_files(VACUUM_TABLE))
display(spark.sql(f"""
    SELECT r.incident_id, i.task_key AS finding, r.action_index, r.action_type, r.execution_status,
           r.action_text, left(r.execution_output, 200) AS output, r.duration_sec, r.executed_at
    FROM {S}.remediation_log r JOIN {S}.incidents i USING (incident_id)
    WHERE i.detection_source = 'table_scan'
      AND i.created_at >= current_timestamp() - INTERVAL {SINCE_HOURS} HOURS
    ORDER BY r.executed_at, r.action_index
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Act 4. Decisions only a person can make (live)
# MAGIC
# MAGIC Five requests wait in the App. Three have steps the agent runs once approved; they always
# MAGIC wait because they change a schema, upgrade a table's protocol, or drop a table:
# MAGIC - v6: `ALTER TABLE orders_daily ADD COLUMNS (discount ...)`
# MAGIC - v7: `ALTER TABLE order_status SET TBLPROPERTIES ('delta.feature.allowColumnDefaults' ...)`
# MAGIC - duplicate table: `DROP TABLE customers_bkp`
# MAGIC
# MAGIC Two are advice only (v4 unreachable API, v5 column typo): the fix is outside Databricks or
# MAGIC in code, so the agent explains it and a person makes the change.
# MAGIC
# MAGIC In the App: **approve** the three with steps, **reject** v4 and v5 with a short note
# MAGIC ("fixing the hostname with the API team", "fixing the typo in the notebook").
# MAGIC Then run the next cells.

# COMMAND ----------

run_job(RECONCILER)
display(spark.sql(f"""
    SELECT a.request_id, a.incident_id, a.agent_name, a.status, a.reviewed_by, a.reviewer_notes,
           r.action_text, r.execution_status, left(r.error_message, 160) AS error
    FROM {S}.approval_requests a
    LEFT JOIN {S}.remediation_log r ON r.request_id = a.request_id
    WHERE a.incident_id IN (SELECT incident_id FROM {S}.incidents WHERE {board_filter()})
    ORDER BY a.requested_at
"""))
print("copy still exists:", spark.catalog.tableExists(DUP_COPY))
print("orders_daily columns:", [c.name for c in spark.table(SCHEMA_TABLE).schema])

# COMMAND ----------

# Proof the fixes worked: re-run v6 and v7. Both should now end SUCCESS.
wait([start(PROBES["v6"][0]), start(PROBES["v7"][0])])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Act 5. Memory: it learns from every incident
# MAGIC
# MAGIC Every diagnosis is stored with an embedding (`databricks-bge-large-en`). The next similar
# MAGIC failure recalls it, including how it ended (approved, rejected and why, fixed).

# COMMAND ----------

display(spark.sql(f"""
    SELECT incident_id, pipeline_name, classified_agent, resolution_status,
           left(resolution_notes, 120) AS resolution_notes, similar_incidents_json, created_at
    FROM {S}.incident_memory
    WHERE incident_id IN (SELECT incident_id FROM {S}.incidents WHERE {board_filter()})
    ORDER BY created_at
"""))
display(spark.sql(f"""
    SELECT pipeline_name, resolution_status, count(*) AS incidents_remembered
    FROM {S}.incident_memory GROUP BY ALL ORDER BY incidents_remembered DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Cleanup (after the demo)

# COMMAND ----------

# 9a. Drop the demo tables and put the duplicate pair back for next time.
for t in (SMALL_FILES_TABLE, SCHEMA_TABLE, FEATURE_TABLE):
    spark.sql(f"DROP TABLE IF EXISTS {t}")
if not spark.catalog.tableExists(DUP_COPY):
    spark.sql(f"CREATE TABLE {DUP_COPY} AS SELECT * FROM {DUP_ORIGINAL}")
print("demo table dropped; duplicate pair restored")

# COMMAND ----------

# 9b. Stop the App (it bills while running) and confirm the jobs are still paused.
w.apps.stop(APP)
for name in (WATCHER, RECONCILER, SCANNER):
    sched = w.jobs.get(job_id(name)).settings.schedule
    print(f"  {name}: {sched.pause_status.value if sched and sched.pause_status else 'no schedule'}")

# COMMAND ----------

# 9c. Optional: put the parked backlog back. The next reconciler run will diagnose it.
# spark.sql(f"UPDATE {S}.incidents SET status = 'NEW', updated_at = current_timestamp() "
#           f"WHERE status = 'PARKED_FOR_DEMO'")
