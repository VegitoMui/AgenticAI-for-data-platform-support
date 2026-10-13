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
  SMALL_FILES     at least SMALL_FILE_MIN_FILES files, average file below
                  SMALL_FILE_BYTES, table at least LAYOUT_IRRELEVANT_BYTES.
  VACUUM_OVERDUE  files removed from the table (by OPTIMIZE, DELETE, MERGE, UPDATE,
                  overwrite) MORE than 168 hours ago, since the last VACUUM, add up
                  to VACUUM_MIN_REMOVED_FILES files or LAYOUT_IRRELEVANT_BYTES bytes.
                  Only removals older than 168 hours count: the system never vacuums
                  with a shorter retention, so newer ones could not be deleted anyway.
  DUPLICATE_TABLE a table whose name is another table's name plus a copy marker
                  (_copy, _bkp, _backup, _old, _tmp, _clone, _v2, a date, copy_of_ ...),
                  in the same schema, with identical columns and a row count within
                  DUPLICATE_ROW_TOLERANCE of it. Always goes to approval: whether a copy
                  can go is its owner's decision (APPROVAL_ONLY_RULES).

Read-only. Nothing here changes a table.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from agentic_ai.agents.base import MIN_VACUUM_RETAIN_HOURS
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
VACUUM_OVERDUE = "VACUUM_OVERDUE"
DUPLICATE_TABLE = "DUPLICATE_TABLE"
# Findings the system may only recommend: a person always decides.
APPROVAL_ONLY_RULES = frozenset({DUPLICATE_TABLE})
DUPLICATE_ROW_TOLERANCE = 0.10
LINEAGE_LOOKBACK_DAYS = 90
_COPY_MARKER = re.compile(
    r"^(?:copy_of_|copy_|backup_|bkp_|old_|tmp_|temp_)"
    r"|(?:_copy\d*|_bkp|_backup|_bak|_old|_tmp|_temp|_clone|_test|_v\d+|_\d{8}|_\d{4}_\d{2}_\d{2})$",
    re.I,
)
VACUUM_MIN_REMOVED_FILES = 100
HISTORY_SCAN_LIMIT = 200
MB = 1024 * 1024
# operationMetrics keys for files and bytes a commit removed, across operations.
_REMOVED_FILES = ("numRemovedFiles", "numTargetFilesRemoved")
_REMOVED_BYTES = ("numRemovedBytes", "numTargetBytesRemoved")


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

def small_files(table: str, detail: dict, history: list[dict] | None = None) -> Finding | None:
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


def _int(v) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def _utc(ts) -> datetime | None:
    if ts is None:
        return None
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def vacuum_overdue(table: str, detail: dict, history: list[dict] | None = None,
                   now: datetime | None = None) -> Finding | None:
    """Files removed more than 168 hours ago, not yet vacuumed, worth deleting."""
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(hours=MIN_VACUUM_RETAIN_HOURS)
    files = size = 0
    for h in sorted(history or [], key=lambda h: _int(h.get("version")), reverse=True):
        op = str(h.get("operation") or "").upper()
        if op.startswith("VACUUM END"):
            break                              # older removals were already eligible then
        ts = _utc(h.get("timestamp"))
        if ts is None or ts > cutoff:
            continue                           # too recent: a 168-hour VACUUM keeps these
        metrics = h.get("operationMetrics") or {}
        files += sum(_int(metrics.get(k)) for k in _REMOVED_FILES)
        size += sum(_int(metrics.get(k)) for k in _REMOVED_BYTES)
    if files < VACUUM_MIN_REMOVED_FILES and size < LAYOUT_IRRELEVANT_BYTES:
        return None
    return Finding(
        VACUUM_OVERDUE, table,
        f"VACUUM OVERDUE on {table}: {files:,} files ({size / MB:,.0f} MB) were removed from the "
        f"table more than {MIN_VACUUM_RETAIN_HOURS} hours ago and are still on storage, because the "
        f"table has not been vacuumed since. `VACUUM {table} RETAIN {MIN_VACUUM_RETAIN_HOURS} HOURS` "
        "deletes them and keeps everything newer, so time travel within the last 7 days still works.",
    )


RULES = (small_files, vacuum_overdue)


# ------------------------------------------------------------------ duplicate tables

def base_name(leaf: str) -> str:
    """'orders_bkp_20261013' -> 'orders'. Unchanged if the name has no copy marker."""
    name = leaf
    for _ in range(3):
        stripped = _COPY_MARKER.sub("", name)
        if stripped == name or not stripped:
            break
        name = stripped
    return name


def duplicate_candidates(tables: list[str]) -> list[tuple[str, str]]:
    """(original, copy) pairs: same schema, copy's name is the original's plus a marker."""
    present = {t.lower(): t for t in tables}
    pairs = []
    for t in tables:
        catalog, schema, leaf = t.split(".")
        original = present.get(f"{catalog}.{schema}.{base_name(leaf)}".lower())
        if original and original.lower() != t.lower():
            pairs.append((original, t))
    return pairs


def _columns(spark, table: str) -> list[tuple[str, str]]:
    cols = []
    for r in _rows(spark, f"DESCRIBE TABLE {quote(table)}"):
        name = str(r.get("col_name") or "")
        if not name or name.startswith("#"):
            break
        cols.append((name.lower(), str(r.get("data_type") or "").lower()))
    return cols


def _age(ts, now: datetime) -> str:
    t = _utc(ts)
    if t is None:
        return "at an unknown time"
    hours = (now - t).total_seconds() / 3600
    return f"{hours:.0f} h ago" if hours < 48 else f"{hours / 24:.0f} days ago"


def _last_lineage(spark, table: str) -> str:
    """When any job, notebook or query last read or wrote the table. Best-effort: lineage
    lags and may not be readable, which is reported rather than guessed."""
    try:
        rows = _rows(spark, f"""
            SELECT max(event_time) AS last_seen FROM system.access.table_lineage
            WHERE (source_table_full_name = '{table}' OR target_table_full_name = '{table}')
              AND event_time >= current_timestamp() - INTERVAL {LINEAGE_LOOKBACK_DAYS} DAYS
        """)
    except Exception:
        return "lineage was not readable, so recent use is unknown"
    last = rows[0].get("last_seen") if rows else None
    if last is None:
        return f"no job, notebook or query read or wrote it in the last {LINEAGE_LOOKBACK_DAYS} days"
    return f"last read or written {_age(last, datetime.now(timezone.utc))} according to lineage"


def compare_duplicate(spark, original: str, copy: str, now: datetime | None = None) -> Finding | None:
    now = now or datetime.now(timezone.utc)
    try:
        cols_o, cols_c = _columns(spark, original), _columns(spark, copy)
        if not cols_o or cols_o != cols_c:
            return None
        rows_o = _int(_rows(spark, f"SELECT count(*) AS n FROM {quote(original)}")[0].get("n"))
        rows_c = _int(_rows(spark, f"SELECT count(*) AS n FROM {quote(copy)}")[0].get("n"))
        if not rows_o or abs(rows_o - rows_c) / rows_o > DUPLICATE_ROW_TOLERANCE:
            return None
        det_o = _rows(spark, f"DESCRIBE DETAIL {quote(original)}")[0]
        det_c = _rows(spark, f"DESCRIBE DETAIL {quote(copy)}")[0]
    except Exception as exc:
        log.info("cannot compare %s with %s: %s", copy, original, str(exc).splitlines()[0][:120])
        return None
    return Finding(
        DUPLICATE_TABLE, copy,
        f"DUPLICATE TABLE: {copy} looks like a leftover copy of `{original}`: identical "
        f"{len(cols_c)} columns, {rows_c:,} rows against {rows_o:,}, last written "
        f"{_age(det_c.get('lastModified'), now)} against {_age(det_o.get('lastModified'), now)}, "
        f"{_int(det_c.get('sizeInBytes')) / MB:,.0f} MB on storage. "
        f"For the copy, {_last_lineage(spark, copy)}. "
        "Whether it can be archived or removed is its owner's decision; nothing is changed automatically.",
    )


def duplicate_tables(spark, tables: list[str]) -> list[Finding]:
    return [f for o, c in duplicate_candidates(tables) if (f := compare_duplicate(spark, o, c))]


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
    try:
        history = _rows(spark, f"DESCRIBE HISTORY {quote(table)} LIMIT {HISTORY_SCAN_LIMIT}")
    except Exception as exc:
        log.info("no history for %s: %s", table, str(exc).splitlines()[0][:120])
        history = []
    return [f for rule in RULES if (f := rule(table, detail, history))]


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
    all_findings = [f for table in tables for f in scan_table(spark, table)]
    all_findings += duplicate_tables(spark, tables)
    for f in all_findings:
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