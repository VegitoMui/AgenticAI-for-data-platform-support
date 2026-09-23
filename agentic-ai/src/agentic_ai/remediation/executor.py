"""
Action execution. Ported from the notebook RemediationEngine, trimmed to
just classification + execution -- the reconciler owns approval routing
and logging, this module just runs one action and reports what happened.
"""

from __future__ import annotations

import logging
import re
import time

log = logging.getLogger(__name__)

EXECUTABLE_TYPES = {"SQL_DDL", "SPARK_CONF"}

CLUSTER_LEVEL_CONFIGS = {
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

SQL_DDL_PREFIXES = (
    "OPTIMIZE", "VACUUM", "ANALYZE", "ALTER TABLE", "GRANT",
    "CREATE TABLE IF NOT EXISTS", "MSCK", "REPAIR", "REFRESH",
)

PLACEHOLDER_TABLES = (
    "target_table", "your_table", "table_name", "schema.table", "agentic_ai.table",
)


def classify_action(action_text: str) -> str:
    stripped = action_text.strip()
    if not stripped:
        return "EMPTY"
    if stripped.startswith("#"):
        return "COMMENT"
    if stripped.startswith("dbutils."):
        return "DBUTILS"
    if stripped.startswith("spark.conf.set"):
        return "SPARK_CONF"
    upper = stripped.upper()
    for prefix in SQL_DDL_PREFIXES:
        if upper.startswith(prefix):
            return "SQL_DDL"
    if any(w in stripped.lower() for w in
           ("run ", "add ", "apply ", "check ", "verify ", "ensure ", "review ", "investigate ", "enable ")):
        return "INSTRUCTION"
    return "UNKNOWN"


def execute_action(spark, action_text: str, action_type: str) -> tuple[str, str, str, float]:
    """Returns (status, output, error_message, duration_sec). status is
    SUCCESS, SKIPPED, or FAILED."""
    start = time.time()
    stripped = action_text.strip().rstrip(";")

    try:
        if action_type == "SQL_DDL":
            if any(p in stripped.lower() for p in PLACEHOLDER_TABLES):
                return "SKIPPED", "", "Placeholder table name detected -- skipped", round(time.time() - start, 2)
            result = spark.sql(stripped)
            try:
                rows = result.collect()
                output = f"OK -- {len(rows)} row(s) returned"
            except Exception:
                output = "OK -- statement executed (no rows)"
            return "SUCCESS", output, "", round(time.time() - start, 2)

        if action_type == "SPARK_CONF":
            match = re.search(r"spark\.conf\.set\(['\"](.+?)['\"]\s*,\s*['\"](.+?)['\"]\)", stripped)
            if not match:
                return "SKIPPED", "", "Could not parse spark.conf.set arguments", round(time.time() - start, 2)
            key, value = match.group(1), match.group(2)
            if key in CLUSTER_LEVEL_CONFIGS:
                return "SKIPPED", "", f"{key} is cluster-level -- cannot apply at runtime", round(time.time() - start, 2)
            spark.conf.set(key, value)
            return "SUCCESS", f"Set {key} = {value}", "", round(time.time() - start, 2)

        if action_type == "DBUTILS":
            return "SKIPPED", "", "DBUTILS actions require manual execution", round(time.time() - start, 2)
        if action_type == "COMMENT":
            return "SKIPPED", action_text, "Documentation only", round(time.time() - start, 2)

        return "SKIPPED", "", f"Action type {action_type} not executable", round(time.time() - start, 2)

    except Exception as e:
        return "FAILED", "", str(e)[:300], round(time.time() - start, 2)


def execute_all(spark, actions: list[str]) -> list[dict]:
    """Runs every action, returns one result dict per action regardless of
    outcome. Never raises -- a failed action is a result, not an exception."""
    out = []
    for i, action_text in enumerate(actions):
        action_type = classify_action(action_text)
        status, output, error, duration = execute_action(spark, action_text, action_type)
        out.append({
            "action_index": i,
            "action_text": action_text,
            "action_type": action_type,
            "execution_status": status,
            "execution_output": output,
            "error_message": error,
            "duration_sec": duration,
        })
        log.info("action %s [%s] -> %s", i, action_type, status)
    return out