"""
Watcher: Phase 1 detection.

Runs on a schedule (default every 5 minutes). One pass:
  1. Read the watermark (last run_end_time already processed).
  2. Pull failed runs since the watermark from the configured detection source.
  3. Skip anything already in processed_runs (belt-and-braces dedupe -- the
     watermark alone is not trusted, since a source can return overlapping
     windows or run out of order).
  4. Write one row per new failure to `incidents` with status='NEW'.
  5. Record each in processed_runs.
  6. Advance the watermark to the latest run_end_time seen.

Deliberately does NOT run diagnosis. This job must stay fast and cheap on a
5-minute schedule; classification and remediation are slow, LLM-bound, and
belong to a later stage that reads status='NEW' incidents on its own cadence.
That split is also what the Phase 2 producer/reconciler design assumes.

Every pass touches watcher_state, even when nothing new is found -- otherwise
last_run_at never reflects reality and get_watermark keeps recomputing the
full lookback window forever instead of narrowing after the first run.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone

from agentic_ai.config import Settings
from agentic_ai.telemetry.sources import FailedRun, get_source

log = logging.getLogger(__name__)

WATCHER_NAME = "live_incident_watcher"


def _get_spark():
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


def get_watermark(settings: Settings, spark=None) -> datetime:
    """Last run_end_time already processed. Falls back to the configured
    lookback window if this watcher has never advanced past one."""
    spark = spark or _get_spark()
    table = settings.table("watcher_state")

    rows = spark.sql(f"""
        SELECT last_run_end_time FROM {table}
        WHERE watcher_name = '{WATCHER_NAME}'
    """).collect()

    if rows and rows[0]["last_run_end_time"] is not None:
        wm = rows[0]["last_run_end_time"]
        if wm.tzinfo is None:
            wm = wm.replace(tzinfo=timezone.utc)
        return wm

    fallback = datetime.now(timezone.utc) - timedelta(hours=settings.watcher_lookback_hours)
    log.info("no watermark found, starting from lookback window: %s", fallback.isoformat())
    return fallback


def set_watermark(settings: Settings, ts: datetime, detection_source: str, spark=None) -> None:
    """Advance the watermark to ts. Called only when a run newer than the
    current watermark was actually processed."""
    spark = spark or _get_spark()
    table = settings.table("watcher_state")
    ts_literal = ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    spark.sql(f"""
        MERGE INTO {table} AS t
        USING (
            SELECT
                '{WATCHER_NAME}' AS watcher_name,
                TIMESTAMP '{ts_literal}' AS last_run_end_time,
                current_timestamp() AS last_run_at,
                '{detection_source}' AS detection_source
        ) AS s
        ON t.watcher_name = s.watcher_name
        WHEN MATCHED THEN UPDATE SET
            t.last_run_end_time = s.last_run_end_time,
            t.last_run_at = s.last_run_at,
            t.detection_source = s.detection_source
        WHEN NOT MATCHED THEN INSERT (
            watcher_name, last_run_end_time, last_run_at, detection_source
        ) VALUES (
            s.watcher_name, s.last_run_end_time, s.last_run_at, s.detection_source
        )
    """)


def _touch_watcher_state(settings: Settings, detection_source: str, spark=None) -> None:
    """Record that a pass happened, without moving the watermark.

    Used when a pass finds nothing new, so last_run_at reflects reality even
    though there is nothing to advance last_run_end_time to. get_watermark
    still falls back to the lookback window as long as last_run_end_time
    is NULL, so this is safe to call repeatedly.
    """
    spark = spark or _get_spark()
    table = settings.table("watcher_state")

    spark.sql(f"""
        MERGE INTO {table} AS t
        USING (
            SELECT '{WATCHER_NAME}' AS watcher_name,
                   current_timestamp() AS last_run_at,
                   '{detection_source}' AS detection_source
        ) AS s
        ON t.watcher_name = s.watcher_name
        WHEN MATCHED THEN UPDATE SET
            t.last_run_at = s.last_run_at,
            t.detection_source = s.detection_source
        WHEN NOT MATCHED THEN INSERT (
            watcher_name, last_run_end_time, last_run_at, detection_source
        ) VALUES (
            s.watcher_name, NULL, s.last_run_at, s.detection_source
        )
    """)


def already_processed(settings: Settings, dedupe_keys: list[str], spark=None) -> set[str]:
    """Which of these dedupe_keys are already recorded in processed_runs."""
    if not dedupe_keys:
        return set()
    spark = spark or _get_spark()
    table = settings.table("processed_runs")
    key_list = ", ".join(f"'{k}'" for k in dedupe_keys)

    rows = spark.sql(f"""
        SELECT dedupe_key FROM {table} WHERE dedupe_key IN ({key_list})
    """).collect()
    return {r["dedupe_key"] for r in rows}


def _write_incident(settings: Settings, run: FailedRun, spark) -> str:
    incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
    table = settings.table("incidents")

    from pyspark.sql.types import StringType, StructField, StructType, TimestampType

    schema = StructType([
        StructField("incident_id", StringType()),
        StructField("dedupe_key", StringType()),
        StructField("pipeline_name", StringType()),
        StructField("job_id", StringType()),
        StructField("run_id", StringType()),
        StructField("task_key", StringType()),
        StructField("result_state", StringType()),
        StructField("raw_error", StringType()),
        StructField("run_page_url", StringType()),
        StructField("run_start_time", TimestampType()),
        StructField("run_end_time", TimestampType()),
        StructField("detection_source", StringType()),
        StructField("status", StringType()),
    ])

    row = [(
        incident_id,
        run.dedupe_key,
        run.job_name,
        run.job_id,
        run.run_id,
        run.task_key,
        run.result_state,
        run.as_error_text(),
        run.run_page_url,
        run.start_time,
        run.end_time,
        run.source,
        "NEW",
    )]

    df = spark.createDataFrame(row, schema=schema)
    df = df.selectExpr(
        "incident_id", "dedupe_key", "pipeline_name", "job_id", "run_id", "task_key",
        "result_state", "raw_error", "run_page_url", "run_start_time", "run_end_time",
        "detection_source", "status",
        "CAST(NULL AS INT) AS severity",
        "CAST(NULL AS STRING) AS classified_agent",
        "CAST(NULL AS INT) AS github_issue_number",
        "current_timestamp() AS created_at",
        "current_timestamp() AS updated_at",
    )
    df.write.format("delta").mode("append").saveAsTable(table)
    return incident_id


def _write_processed(settings: Settings, run: FailedRun, incident_id: str, spark) -> None:
    table = settings.table("processed_runs")

    from pyspark.sql.types import StringType, StructField, StructType, TimestampType

    schema = StructType([
        StructField("dedupe_key", StringType()),
        StructField("run_id", StringType()),
        StructField("job_id", StringType()),
        StructField("task_key", StringType()),
        StructField("incident_id", StringType()),
        StructField("result_state", StringType()),
        StructField("run_end_time", TimestampType()),
        StructField("detection_source", StringType()),
    ])

    row = [(
        run.dedupe_key, run.run_id, run.job_id, run.task_key,
        incident_id, run.result_state, run.end_time, run.source,
    )]

    df = spark.createDataFrame(row, schema=schema)
    df = df.selectExpr(
        "dedupe_key", "run_id", "job_id", "task_key", "incident_id",
        "result_state", "run_end_time", "detection_source",
        "current_timestamp() AS processed_at",
    )
    df.write.format("delta").mode("append").saveAsTable(table)


def run_once(settings: Settings, spark=None) -> int:
    """One watcher pass. Returns the number of incidents raised."""
    spark = spark or _get_spark()

    source = get_source(settings, spark=spark)
    since = get_watermark(settings, spark=spark)
    log.info("watcher pass starting | source=%s since=%s", source.name, since.isoformat())

    runs = source.fetch_failed_runs(since, settings.watcher_batch_limit)
    if not runs:
        log.info("no failed runs found since %s", since.isoformat())
        _touch_watcher_state(settings, source.name, spark=spark)
        return 0

    seen = already_processed(settings, [r.dedupe_key for r in runs], spark=spark)
    new_runs = [r for r in runs if r.dedupe_key not in seen]

    skipped = len(runs) - len(new_runs)
    if skipped:
        log.info("skipped %s already-processed run(s)", skipped)

    raised = 0
    latest_end = since

    for run in new_runs:
        try:
            incident_id = _write_incident(settings, run, spark)
            _write_processed(settings, run, incident_id, spark)
            raised += 1
            log.info("raised %s for %s (%s)", incident_id, run.dedupe_key, run.result_state)
        except Exception:
            log.exception("failed to raise incident for %s -- will retry next pass", run.dedupe_key)
            continue

        run_end = run.end_time
        if run_end and run_end.tzinfo is None:
            run_end = run_end.replace(tzinfo=timezone.utc)
        if run_end and run_end > latest_end:
            latest_end = run_end

    # Advance the watermark even on a partial pass: a run that failed to
    # write above is not in processed_runs, so the dedupe check will retry
    # it on the next pass regardless of where the watermark sits. Only move
    # it if we actually saw something newer, so a source hiccup never
    # rewinds it.
    if latest_end > since:
        set_watermark(settings, latest_end, source.name, spark=spark)
    else:
        _touch_watcher_state(settings, source.name, spark=spark)

    log.info("watcher pass complete | raised=%s skipped=%s", raised, skipped)
    return raised