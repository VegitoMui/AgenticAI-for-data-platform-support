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

# MAGIC %pip install /Workspace/agentic-ai/prod/artifacts/.internal/agentic_ai-0.1.0-py3-none-any.whl --quiet
# MAGIC %restart_python

# COMMAND ----------

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

display(spark.sql("""
    SELECT result_state, count(*) AS runs
    FROM system.lakeflow.job_run_timeline
    WHERE period_end_time > current_timestamp() - INTERVAL 7 DAYS
    GROUP BY result_state
    ORDER BY runs DESC
"""))

# COMMAND ----------

display(spark.sql("""
    SELECT count(*) AS queries
    FROM system.query.history
    WHERE start_time > current_timestamp() - INTERVAL 1 DAYS
"""))

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
