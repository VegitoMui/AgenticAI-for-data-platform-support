# Databricks notebook source
# MAGIC %md
# MAGIC # Setup: Connectivity Check
# MAGIC
# MAGIC Runs entirely in the workspace. Verifies that config resolution, the LLM
# MAGIC failover chain, GitHub access, and system table access all work with the
# MAGIC secrets stored by `setup_secrets`.
# MAGIC
# MAGIC Run this after every deploy to a new environment.

# COMMAND ----------

# MAGIC %pip install openai requests --quiet
# MAGIC %restart_python

# COMMAND ----------

import sys
sys.path.append("/Workspace/Users/yashovardhan.rawat@airodigitallabs.com/AgenticAI-for-data-platform-support/agentic-ai/src")

import logging
logging.basicConfig(level=logging.INFO)

from agentic_ai.config import Settings

settings = Settings(catalog="databricks_ws", schema="agentic_ai", secret_scope="agentic-ai")
print("catalog.schema:", settings.fq_schema)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Table name normalisation
# MAGIC Both forms must resolve identically. If they differ, the double-prefix
# MAGIC guard is broken and every downstream SQL statement inherits the bug.

# COMMAND ----------

a = settings.table("agentic_ai.remediation_log")
b = settings.table("remediation_log")
print(a)
print(b)
assert a == b == f"{settings.fq_schema}.remediation_log", "table() normalisation failed"
print("PASS")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. LLM failover chain

# COMMAND ----------

from agentic_ai.llm.client import LLMClient

llm = LLMClient.from_settings(settings)
results = llm.health_check()
for r in results:
    print(f"  {r['status'].upper():5} {r['name']}  {r.get('latency_sec', '')}{r.get('error', '')}")

passing = sum(1 for r in results if r["status"] == "pass")
assert passing > 0, "no LLM provider reachable"
print(f"\n{passing}/{len(results)} providers reachable")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. GitHub access
# MAGIC Reads recent commits. Confirms the token and Contents:read permission.

# COMMAND ----------

from datetime import datetime, timedelta, timezone

from agentic_ai.vcs.github import GitHubClient

gh = GitHubClient.from_settings(settings)
commits = gh.commits_since(datetime.now(timezone.utc) - timedelta(days=14))
print(f"{len(commits)} commit(s) in the last 14 days")
for c in commits[:5]:
    print("  " + c.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. System table access
# MAGIC The watcher depends on this. If it errors, the system schemas are not
# MAGIC enabled or the running identity lacks SELECT.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Failed-run detection source
# MAGIC
# MAGIC Probes both detection sources and reports which are usable.
# MAGIC SystemTablesSource is preferred; JobsApiSource is the fallback that
# MAGIC works without metastore-level system schema grants.

# COMMAND ----------

from datetime import datetime, timedelta, timezone

FAILED_STATES = ("FAILED", "TIMED_OUT", "CANCELED")
LOOKBACK_HOURS = 24

# --- Source A: system tables -------------------------------------------------

system_tables_ok = False
try:
    df = spark.sql(f"""
        SELECT result_state, count(*) AS runs
        FROM system.lakeflow.job_run_timeline
        WHERE period_end_time > current_timestamp() - INTERVAL {LOOKBACK_HOURS} HOURS
        GROUP BY result_state
        ORDER BY runs DESC
    """)
    rows = df.collect()
    system_tables_ok = True
    print(f"SystemTablesSource: AVAILABLE ({len(rows)} distinct result_state values)")
    for r in rows:
        print(f"    {r['result_state'] or 'NULL':12} {r['runs']}")
except Exception as exc:
    print(f"SystemTablesSource: UNAVAILABLE -- {str(exc)[:160]}")

# --- Source B: Jobs API ------------------------------------------------------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
since = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
start_time_from = int(since.timestamp() * 1000)

runs, failed = [], []
for run in w.jobs.list_runs(
    completed_only=True,
    start_time_from=start_time_from,
    expand_tasks=False,
    limit=25,
):
    runs.append(run)
    state = run.state
    result_state = getattr(state.result_state, "value", None) if state else None
    if result_state in FAILED_STATES:
        failed.append((run.run_id, run.job_id, run.run_name, result_state,
                       (state.state_message or "")[:120]))

print(f"\nJobsApiSource: AVAILABLE -- {len(runs)} completed run(s) in the "
      f"last {LOOKBACK_HOURS}h, {len(failed)} in a failed state")
for run_id, job_id, name, result_state, msg in failed[:10]:
    print(f"    run={run_id} job={job_id} {result_state:10} {str(name)[:40]}")
    if msg:
        print(f"        {msg}")

assert runs or failed is not None, "Jobs API returned nothing and did not error -- investigate"
print(f"\nDetection source to use for Phase 1: "
      f"{'system_tables' if system_tables_ok else 'jobs_api'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Target schema is writable

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {settings.fq_schema}")
probe = settings.table("_connectivity_probe")
spark.sql(f"CREATE TABLE IF NOT EXISTS {probe} (checked_at TIMESTAMP)")
spark.sql(f"INSERT INTO {probe} VALUES (current_timestamp())")
spark.sql(f"DROP TABLE {probe}")
print("PASS: schema is writable")

# COMMAND ----------

print("All connectivity checks complete.")

# COMMAND ----------

