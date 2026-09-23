from __future__ import annotations

import logging

from agentic_ai.config import Settings

log = logging.getLogger(__name__)


# Ordered. Each entry: (leaf table name, column DDL, list of additive ALTER clauses)
TABLES: list[tuple[str, str, list[str]]] = [

    # ---------------------------------------------------------------- watcher
    (
        "watcher_state",
        """
        watcher_name        STRING NOT NULL,
        last_run_end_time   TIMESTAMP,
        last_run_at         TIMESTAMP,
        detection_source    STRING,
        notes               STRING
        """,
        [],
    ),

    # Dedupe ledger. One row per run the watcher has already seen, so a
    # watermark that moves backwards or a source that returns overlapping
    # windows can never raise the same incident twice.
    (
        "processed_runs",
        """
        dedupe_key          STRING NOT NULL,
        run_id              STRING,
        job_id              STRING,
        task_key            STRING,
        incident_id         STRING,
        result_state        STRING,
        run_end_time        TIMESTAMP,
        processed_at        TIMESTAMP,
        detection_source    STRING
        """,
        [],
    ),

    # -------------------------------------------------------------- incidents
    (
        "incidents",
        """
        incident_id         STRING NOT NULL,
        dedupe_key          STRING,
        pipeline_name       STRING,
        job_id              STRING,
        run_id              STRING,
        task_key            STRING,
        result_state        STRING,
        raw_error           STRING,
        run_page_url        STRING,
        run_start_time      TIMESTAMP,
        run_end_time        TIMESTAMP,
        detection_source    STRING,
        status              STRING,
        severity            INT,
        classified_agent    STRING,
        github_issue_number INT,
        created_at          TIMESTAMP,
        updated_at          TIMESTAMP
        """,
        [],
    ),

    # -------------------------------------------------------------- approvals
    (
        "approval_requests",
        """
        request_id          STRING NOT NULL,
        incident_id         STRING,
        pipeline_name       STRING,
        agent_name          STRING,
        priority            STRING,
        severity            INT,
        diagnosis           STRING,
        proposed_actions    STRING,
        fix_complexity      STRING,
        fix_suggestion      STRING,
        requires_human      BOOLEAN,
        status              STRING,
        requested_at        TIMESTAMP,
        expires_at          TIMESTAMP,
        reviewed_at         TIMESTAMP,
        reviewed_by         STRING,
        reviewer_notes      STRING,
        github_issue_number INT
        """,
        [],
    ),

    # ------------------------------------------------------------ remediation
    (
        "remediation_log",
        """
        remediation_id      STRING NOT NULL,
        incident_id         STRING,
        request_id          STRING,
        pipeline_name       STRING,
        agent_name          STRING,
        action_index        INT,
        action_text         STRING,
        action_type         STRING,
        execution_status    STRING,
        execution_output    STRING,
        error_message       STRING,
        complexity          STRING,
        executed_at         TIMESTAMP,
        duration_sec        DOUBLE
        """,
        [],
    ),

    # ------------------------------------------------------------- RAG memory
    (
        "incident_memory",
        """
        incident_id                 STRING NOT NULL,
        pipeline_name               STRING,
        raw_error                   STRING,
        classified_agent            STRING,
        classification_confidence   DOUBLE,
        diagnosis                   STRING,
        severity                    INT,
        priority                    STRING,
        fix_suggestion              STRING,
        fix_complexity              STRING,
        requires_human              BOOLEAN,
        actions_json                STRING,
        tools_used_json             STRING,
        resolution_status           STRING,
        resolution_notes            STRING,
        similar_incidents_json      STRING,
        embedding                   ARRAY<FLOAT>,
        total_latency_sec           DOUBLE,
        created_at                  TIMESTAMP
        """,
        [],
    ),

    # -------------------------------------------------------- execution trace
    (
        "subagent_execution_log",
        """
        incident_id         STRING,
        pipeline_name       STRING,
        step_number         INT,
        node_name           STRING,
        detail              STRING,
        sub_agent_selected  STRING,
        tools_run           STRING,
        duration_sec        DOUBLE,
        logged_at           TIMESTAMP
        """,
        [],
    ),
]


def apply_all(settings: Settings, spark=None) -> dict[str, int]:
    """Create the schema and every table. Idempotent. Returns a small summary."""
    spark = spark or _get_spark()

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {settings.fq_schema}")
    log.info("schema ready: %s", settings.fq_schema)

    created, altered = 0, 0

    for leaf, columns, alters in TABLES:
        fq = settings.table(leaf)
        spark.sql(f"CREATE TABLE IF NOT EXISTS {fq} ({columns}) USING DELTA")
        created += 1
        log.info("table ready: %s", fq)

        for clause in alters:
            try:
                spark.sql(f"ALTER TABLE {fq} ADD COLUMNS IF NOT EXISTS ({clause})")
                altered += 1
            except Exception as exc:
                log.warning("alter skipped on %s (%s): %s", fq, clause, str(exc)[:160])

    summary = {"tables": created, "alters_applied": altered}
    log.info("migrations complete: %s", summary)
    return summary


def _get_spark():
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()