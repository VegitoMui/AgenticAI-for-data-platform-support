"""
Table scanner: maintenance problems that never make a job fail.

Small files, overdue VACUUM and duplicate tables slow jobs down and cost
money, but no run ever FAILS because of them, so the watcher never sees them.
This job looks at the tables themselves instead.

One pass, for every Delta table in the configured schemas (settings.scan_schemas):
  1. DESCRIBE DETAIL.
  2. Apply the rules below. Each hit is a finding.
  3. Raise one incident per finding (detection_source 'table_scan', status NEW),
     unless the same finding on the same table is already open, or was fixed
     automatically within the cool-down -- so a fix that did not help never loops.

The reconciler then diagnoses it with the Storage agent like any incident.

Rules (thresholds shared with the Storage agent, so the two always agree):
  SMALL_FILES   at least SMALL_FILE_MIN_FILES files, average file below
                SMALL_FILE_BYTES, table at least LAYOUT_IRRELEVANT_BYTES.

Read-only. Nothing here changes a table.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from agentic_ai.agents.storage import (
    LAYOUT_IRRELEVANT_BYTES,
    SMALL_FILE_BYTES,
    SMALL_FILE_MIN_FILES,
    _rows,
    quote,
)
from agentic_ai.config import Settings
from agentic_ai.telemetry.run_context import TABLE_SCAN_SOURCE

log = logging.getLogger(__name__)

SMALL_FILES = "SMALL_FILES"
MB = 1024 * 1024


@dataclass
class Finding:
    rule: str
    table: str
    summary: str

    @property
    def key(self) -> str:
        """Stored as the incident's task_key: one open incident per rule per table."""
        return f"{self.rule}:{self.table}"


# ------------------------------------------------------------------ rules

def small_files(table: str, detail: dict) -> Finding | None:
    files = int(detail.get("numFiles") or 0)
    size = int(detail.get("sizeInBytes") or 0)
    if files < SMALL_FILE_MIN_FILES or size < LAYOUT_IRRELEVANT_BYTES:
        return None
    avg = size / files
    if avg >= SMALL_FILE_BYTES:
        return None
    return Finding(
        SMALL_FILES, table,
        f"SMALL FILES on {table}: {files:,} files averaging {avg / MB:,.1f} MB "
        f"({size / MB:,.0f} MB in total; files under {SMALL_FILE_BYTES // MB} MB count as small). "
        "Reads open far more files than needed. Compacting with OPTIMIZE is the usual fix.",
    )


RULES = (small_files,)


# ------------------------------------------------------------------ scanning

def schema_tables(spark, schema: str) -> list[str]:
    parts = schema.split(".")
    if len(parts) != 2 or not quote(f"{schema}.x"):
        log.warning("skipping invalid schema name %r (expected catalog.schema)", schema)
        return []
    rows = _rows(spark, f"SHOW TABLES IN `{parts[0]}`.`{parts[1]}`")
    out = []
    for r in rows:
        if r.get("isTemporary"):
            continue
        name = f"{schema}.{r.get('tableName')}"
        if quote(name):
            out.append(name)
    return out


def scan_table(spark, table: str) -> list[Finding]:
    try:
        detail = _rows(spark, f"DESCRIBE DETAIL {quote(table)}")[0]
    except Exception as exc:          # views, permissions, dropped mid-scan
        log.info("skipping %s: %s", table, str(exc).splitlines()[0][:120])
        return []
    if str(detail.get("format") or "").lower() != "delta":
        return []
    return [f for rule in RULES if (f := rule(table, detail))]


def suppressed_keys(settings: Settings, spark) -> set[str]:
    """Findings not to raise again: still open, or fixed automatically within the cool-down."""
    cooldown = int(settings.scan_cooldown_hours)
    rows = spark.sql(f"""
        SELECT DISTINCT task_key FROM {settings.table('incidents')}
        WHERE detection_source = '{TABLE_SCAN_SOURCE}' AND task_key IS NOT NULL
          AND (status IN ('NEW', 'PENDING_APPROVAL')
               OR (status LIKE 'AUTO_EXECUTED%'
                   AND updated_at >= current_timestamp() - INTERVAL {cooldown} HOURS))
    """).collect()
    return {r["task_key"] for r in rows}


def _write_incident(settings: Settings, spark, finding: Finding) -> str:
    from pyspark.sql.types import StringType, StructField, StructType, TimestampType

    incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.now(timezone.utc)
    schema = StructType([
        StructField("incident_id", StringType()),
        StructField("dedupe_key", StringType()),
        StructField("pipeline_name", StringType()),
        StructField("task_key", StringType()),
        StructField("result_state", StringType()),
        StructField("raw_error", StringType()),
        StructField("run_start_time", TimestampType()),
        StructField("run_end_time", TimestampType()),
        StructField("detection_source", StringType()),
        StructField("status", StringType()),
    ])
    row = [(incident_id, f"scan:{finding.key}:{now:%Y%m%d%H%M%S}", f"table maintenance: {finding.table}",
            finding.key, "MAINTENANCE", finding.summary, now, now, TABLE_SCAN_SOURCE, "NEW")]
    df = spark.createDataFrame(row, schema=schema).selectExpr(
        "incident_id", "dedupe_key", "pipeline_name",
        "CAST(NULL AS STRING) AS job_id", "CAST(NULL AS STRING) AS run_id",
        "task_key", "result_state", "raw_error", "CAST(NULL AS STRING) AS run_page_url",
        "run_start_time", "run_end_time", "detection_source", "status",
        "CAST(NULL AS INT) AS severity", "CAST(NULL AS STRING) AS classified_agent",
        "CAST(NULL AS INT) AS github_issue_number",
        "current_timestamp() AS created_at", "current_timestamp() AS updated_at",
    )
    df.write.format("delta").mode("append").saveAsTable(settings.table("incidents"))
    return incident_id


def run_once(settings: Settings, spark=None) -> dict:
    if spark is None:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
    if not settings.scan_schemas:
        log.warning("no scan schemas configured (--scan-schemas); nothing to scan")
        return {"tables": 0, "findings": 0, "raised": 0, "suppressed": 0}

    tables: list[str] = []
    for schema in settings.scan_schemas:
        try:
            tables += schema_tables(spark, schema)
        except Exception as exc:
            log.warning("cannot list %s: %s", schema, str(exc).splitlines()[0][:160])
    tables = tables[: settings.scan_table_limit]

    skip = suppressed_keys(settings, spark)
    findings, raised, suppressed = 0, 0, 0
    for table in tables:
        for f in scan_table(spark, table):
            findings += 1
            if f.key in skip:
                suppressed += 1
                continue
            incident_id = _write_incident(settings, spark, f)
            skip.add(f.key)
            raised += 1
            log.info("raised %s: %s", incident_id, f.summary[:160])

    summary = {"tables": len(tables), "findings": findings, "raised": raised, "suppressed": suppressed}
    log.info("table scan complete: %s", summary)
    return summary