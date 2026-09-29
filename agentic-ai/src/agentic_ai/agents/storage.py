"""
Storage agent: problems with Delta tables themselves.

Sub-agents
  TABLE_HEALTH   missing tables, concurrent-write conflicts, protocol and log
                 problems. The default.
  OPTIMIZATION   small files, file layout, OPTIMIZE / VACUUM upkeep.
  DATA_QUALITY   nulls, duplicates, schema mismatches on write.

Every tool is read-only: DESCRIBE, SHOW, DESCRIBE HISTORY, and a bounded
sampled aggregate. Nothing here changes a table -- changes are proposed as
actions and go through the approval gate like any other fix.

Tables come from the run context: tables the job wrote first (most likely to
be the broken ones), then tables it read, then names found in the failing
code. Names are validated before they are put into SQL.
"""

from __future__ import annotations

import difflib
import re
from datetime import datetime, timezone

from agentic_ai.agents.base import BaseAgent, SubAgent, Tool, ToolContext

MAX_TABLES = 3
SMALL_FILE_BYTES = 32 * 1024 * 1024          # average file below this is "small"
SMALL_FILE_MIN_FILES = 100                   # ...once there are at least this many files
LAYOUT_IRRELEVANT_BYTES = 256 * 1024 * 1024  # below this total size, layout rarely matters
HISTORY_LIMIT = 30
PROFILE_SAMPLE_ROWS = 10000
PROFILE_MAX_COLUMNS = 8

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
RELEVANT_PROPERTIES = (
    "delta.autoOptimize.optimizeWrite",
    "delta.autoOptimize.autoCompact",
    "delta.targetFileSize",
    "delta.tuneFileSizesForRewrites",
    "delta.logRetentionDuration",
    "delta.deletedFileRetentionDuration",
    "delta.enableChangeDataFeed",
    "delta.appendOnly",
    "delta.enableDeletionVectors",
    "delta.isolationLevel",
)


# ------------------------------------------------------------------ helpers

def quote(table: str) -> str | None:
    """`cat`.`sch`.`tbl` for a valid three-part name, else None."""
    parts = table.split(".")
    if len(parts) != 3 or not all(_IDENT.match(p) for p in parts):
        return None
    return ".".join(f"`{p}`" for p in parts)


def target_tables(tctx: ToolContext) -> list[str]:
    run = tctx.run
    ordered = list(run.tables_written) + list(run.tables_read) + list(run.tables_from_code)
    out: list[str] = []
    for t in ordered:
        if quote(t) and t not in out:
            out.append(t)
    return out[:MAX_TABLES]


def _rows(spark, query: str) -> list[dict]:
    return [r.asDict() if hasattr(r, "asDict") else dict(r) for r in spark.sql(query).collect()]


def _mb(n) -> str:
    return f"{(n or 0) / (1024 * 1024):,.1f} MB"


def _age_hours(ts) -> float | None:
    if ts is None:
        return None
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - ts).total_seconds() / 3600


def _ago(hours: float | None) -> str:
    if hours is None:
        return "unknown"
    return f"{hours:.1f} h ago" if hours < 48 else f"{hours / 24:.0f} days ago"


def _no_tables() -> tuple[str, dict]:
    return "No tables were identified for this run (no lineage and none named in the error).", {}


# ------------------------------------------------------------------ tools

def table_exists(tctx: ToolContext) -> tuple[str, dict]:
    tables = target_tables(tctx)
    if not tables:
        return _no_tables()
    findings, data = [], {}
    for t in tables:
        try:
            _rows(tctx.spark, f"DESCRIBE TABLE {quote(t)}")
            findings.append(f"{t}: exists")
            data[t] = {"exists": True}
            continue
        except Exception as exc:
            reason = str(exc).splitlines()[0][:160]

        catalog, schema, leaf = t.split(".")
        similar: list[str] = []
        try:
            listing = _rows(tctx.spark, f"SHOW TABLES IN `{catalog}`.`{schema}`")
            names = [r.get("tableName", "") for r in listing]
            similar = [f"{catalog}.{schema}.{n}" for n in difflib.get_close_matches(leaf, names, n=3)]
        except Exception:
            pass
        hint = f"; similar names in the schema: {', '.join(similar)}" if similar else ""
        findings.append(f"{t}: NOT READABLE ({reason}){hint}")
        data[t] = {"exists": False, "reason": reason, "similar": similar}
    return "; ".join(findings), data


def table_detail(tctx: ToolContext) -> tuple[str, dict]:
    tables = target_tables(tctx)
    if not tables:
        return _no_tables()
    findings, data = [], {}
    for t in tables:
        try:
            d = _rows(tctx.spark, f"DESCRIBE DETAIL {quote(t)}")[0]
        except Exception as exc:
            findings.append(f"{t}: DESCRIBE DETAIL failed ({str(exc).splitlines()[0][:120]})")
            continue
        files = int(d.get("numFiles") or 0)
        size = int(d.get("sizeInBytes") or 0)
        avg = size / files if files else 0
        small = files >= SMALL_FILE_MIN_FILES and avg < SMALL_FILE_BYTES and size >= LAYOUT_IRRELEVANT_BYTES

        parts = [
            f"{t}: format {d.get('format')}, {files:,} files, {_mb(size)} total, {_mb(avg)} average file",
            f"partitioned by {d.get('partitionColumns') or 'nothing'}",
            f"clustered by {d.get('clusteringColumns') or 'nothing'}",
            f"reader/writer protocol {d.get('minReaderVersion')}/{d.get('minWriterVersion')}",
            f"last modified {_ago(_age_hours(d.get('lastModified')))}",
        ]
        if small:
            parts.append("SMALL-FILE PROBLEM LIKELY")
        elif size < LAYOUT_IRRELEVANT_BYTES:
            parts.append("table is small, so file layout is unlikely to matter")
        findings.append(", ".join(parts))
        data[t] = {"num_files": files, "size_bytes": size, "avg_file_bytes": avg, "small_files": small}
    return "; ".join(findings), data


def delta_history(tctx: ToolContext) -> tuple[str, dict]:
    tables = target_tables(tctx)
    if not tables:
        return _no_tables()
    findings, data = [], {}
    for t in tables:
        try:
            hist = _rows(tctx.spark, f"DESCRIBE HISTORY {quote(t)} LIMIT {HISTORY_LIMIT}")
        except Exception as exc:
            findings.append(f"{t}: history unavailable ({str(exc).splitlines()[0][:120]})")
            continue
        if not hist:
            findings.append(f"{t}: no history")
            continue

        ops: dict[str, int] = {}
        last: dict[str, float | None] = {}
        writers_24h: set[str] = set()
        writes_24h = 0
        for h in hist:
            op = str(h.get("operation") or "UNKNOWN")
            ops[op] = ops.get(op, 0) + 1
            age = _age_hours(h.get("timestamp"))
            if op not in last:
                last[op] = age
            if age is not None and age <= 24 and op not in ("OPTIMIZE", "VACUUM START", "VACUUM END"):
                writes_24h += 1
                job = h.get("job") or {}
                job_id = job.get("jobId") if isinstance(job, dict) else getattr(job, "jobId", None)
                writers_24h.add(str(job_id or h.get("userName") or "unknown"))

        latest = hist[0]
        op_summary = ", ".join(f"{k} x{v}" for k, v in sorted(ops.items(), key=lambda kv: -kv[1]))
        opt_age = last.get("OPTIMIZE")
        vac_age = last.get("VACUUM END", last.get("VACUUM START"))
        parts = [
            f"{t}: last {len(hist)} versions: {op_summary}",
            f"latest {latest.get('operation')} {_ago(_age_hours(latest.get('timestamp')))}",
            f"last OPTIMIZE {_ago(opt_age) if opt_age is not None else 'not in recent history'}",
            f"last VACUUM {_ago(vac_age) if vac_age is not None else 'not in recent history'}",
            f"{writes_24h} writes by {len(writers_24h)} distinct writer(s) in the last 24 h",
        ]
        if len(writers_24h) > 1:
            parts.append("MULTIPLE CONCURRENT WRITERS")
        findings.append(", ".join(parts))
        data[t] = {"operations": ops, "writes_24h": writes_24h, "writers_24h": sorted(writers_24h),
                   "hours_since_optimize": opt_age, "hours_since_vacuum": vac_age}
    return "; ".join(findings), data


def table_properties(tctx: ToolContext) -> tuple[str, dict]:
    tables = target_tables(tctx)
    if not tables:
        return _no_tables()
    findings, data = [], {}
    for t in tables:
        try:
            listing = _rows(tctx.spark, f"SHOW TBLPROPERTIES {quote(t)}")
            props = {r.get("key"): r.get("value") for r in listing}
        except Exception as exc:
            findings.append(f"{t}: properties unavailable ({str(exc).splitlines()[0][:120]})")
            continue
        relevant = {k: props[k] for k in RELEVANT_PROPERTIES if k in props}
        shown = ", ".join(f"{k}={v}" for k, v in relevant.items()) or "no relevant Delta properties set"
        notes = []
        if str(props.get("delta.autoOptimize.optimizeWrite", "")).lower() != "true":
            notes.append("optimized writes not enabled")
        if str(props.get("delta.autoOptimize.autoCompact", "")).lower() != "true":
            notes.append("auto compaction not enabled")
        findings.append(f"{t}: {shown}" + (f" ({'; '.join(notes)})" if notes else ""))
        data[t] = relevant
    return "; ".join(findings), data


def data_profile(tctx: ToolContext) -> tuple[str, dict]:
    tables = target_tables(tctx)
    if not tables:
        return _no_tables()
    findings, data = [], {}
    for t in tables:
        q = quote(t)
        try:
            cols = []
            for r in _rows(tctx.spark, f"DESCRIBE TABLE {q}"):
                name = str(r.get("col_name") or "")
                if not name or name.startswith("#"):
                    break
                if _IDENT.match(name):
                    cols.append(name)
            cols = cols[:PROFILE_MAX_COLUMNS]
            null_exprs = ", ".join(f"sum(CASE WHEN `{c}` IS NULL THEN 1 ELSE 0 END) AS `{c}`" for c in cols)
            select = "count(*) AS rows_sampled, count(DISTINCT hash(*)) AS distinct_rows"
            if null_exprs:
                select += ", " + null_exprs
            p = _rows(tctx.spark, f"SELECT {select} FROM {q} TABLESAMPLE ({PROFILE_SAMPLE_ROWS} ROWS)")[0]
        except Exception as exc:
            findings.append(f"{t}: profile failed ({str(exc).splitlines()[0][:120]})")
            continue

        sampled = int(p.get("rows_sampled") or 0)
        dupes = sampled - int(p.get("distinct_rows") or 0)
        nulls = {c: int(p.get(c) or 0) for c in cols}
        heavy = [f"{c} {n / sampled:.0%}" for c, n in nulls.items() if sampled and n / sampled >= 0.2]
        parts = [f"{t}: {sampled:,} rows sampled", f"{dupes:,} fully duplicate rows in sample"]
        parts.append(f"columns with 20%+ nulls: {', '.join(heavy)}" if heavy else "no column is 20%+ null")
        findings.append(", ".join(parts))
        data[t] = {"rows_sampled": sampled, "duplicate_rows": dupes, "nulls": nulls}
    return "; ".join(findings), data


# ------------------------------------------------------------------ agent

class StorageAgent(BaseAgent):
    name = "storage"
    description = "Delta table health, file layout and upkeep, and data quality on write."

    sub_agents = [
        SubAgent(
            "TABLE_HEALTH",
            "Is the table present, readable and consistent? Look for missing tables, "
            "concurrent writers, protocol or log problems.",
            ("not found", "does not exist", "table_or_view", "concurrent", "protocol",
             "corrupt", "checksum", "_delta_log", "deltaanalysisexception"),
            ("table_exists", "table_detail", "delta_history", "table_properties"),
        ),
        SubAgent(
            "OPTIMIZATION",
            "Is the file layout hurting reads or writes? Look at file counts and sizes, "
            "OPTIMIZE / VACUUM recency and auto-optimize settings.",
            ("small files", "too many files", "optimize", "vacuum", "zorder", "file size",
             "slow", "listing"),
            ("table_detail", "delta_history", "table_properties"),
        ),
        SubAgent(
            "DATA_QUALITY",
            "Is the data itself wrong? Look at nulls, duplicates and schema mismatches.",
            ("duplicate", "null", "schema mismatch", "cannot write", "constraint",
             "merge", "cast", "overflow", "invalid value"),
            ("table_exists", "table_detail", "data_profile", "delta_history"),
        ),
    ]

    tools = {
        "table_exists": Tool("table_exists", "Is each table readable; similar names if not", table_exists),
        "table_detail": Tool("table_detail", "DESCRIBE DETAIL: files, size, layout, protocol", table_detail),
        "delta_history": Tool("delta_history", "DESCRIBE HISTORY: writers, OPTIMIZE/VACUUM", delta_history),
        "table_properties": Tool("table_properties", "Relevant Delta table properties", table_properties),
        "data_profile": Tool("data_profile", "Sampled null and duplicate profile", data_profile),
    }