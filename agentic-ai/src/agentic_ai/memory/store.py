"""
Incident memory: similar past incidents and how they were resolved.

  recall          Embed the new incident, compare it with stored incidents by
                  cosine similarity, and return the closest few (above a
                  minimum similarity) as short text lines for the agent prompt.
  remember        After diagnosis, store the incident text, its embedding, the
                  route, the diagnosis and the initial outcome.
  update_outcome  When the outcome changes later (approved and executed,
                  rejected, expired), update the stored row, so later recalls
                  know whether a past fix was accepted.

Embeddings come from the Databricks-hosted model databricks-bge-large-en
(1024 dimensions, pay per use, nothing installed). Rows with any other
dimension -- e.g. old 384-dim MiniLM vectors -- are ignored, never mixed.

Memory is advisory: every failure here is logged and swallowed, and the agent
simply runs without memory.
"""

from __future__ import annotations

import json
import logging
import math
import re

from agentic_ai.config import Settings
from agentic_ai.telemetry.run_context import RunContext

log = logging.getLogger(__name__)

EMBEDDING_MODEL = "databricks-bge-large-en"
EMBEDDING_DIM = 1024
MAX_EMBED_CHARS = 2000          # bge-large reads at most 512 tokens
TOP_K = 3
MIN_SIMILARITY = 0.85
CANDIDATE_LIMIT = 2000          # most recent stored incidents compared

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]+$")
# Run-specific noise that would make identical failures look different.
_NOISE = re.compile(
    r"\b\d{6,}\b|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
    r"|\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\S*",
    re.I,
)


# ------------------------------------------------------------------ text and math

def incident_text(run: RunContext) -> str:
    """What the incident is about, without run ids, timestamps or JVM frames."""
    lines = []
    for t in run.task_errors:
        kind = t.task_type.split(" ")[0]
        lines.append(f"{kind} task failed: {t.error}")
    if not lines:
        lines.append(f"job {run.job_name} failed with no error details")
    if run.tables:
        lines.append(f"tables: {', '.join(run.tables)}")
    return _NOISE.sub("", "\n".join(lines))[:MAX_EMBED_CHARS]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def embed(workspace_client, text: str) -> list[float]:
    resp = workspace_client.serving_endpoints.query(name=EMBEDDING_MODEL, input=[text])
    vec = [float(v) for v in (resp.data[0].embedding or [])]
    if len(vec) != EMBEDDING_DIM:
        raise ValueError(f"{EMBEDDING_MODEL} returned {len(vec)} dimensions, expected {EMBEDDING_DIM}")
    return vec


def _field(row, key):
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]
    except (KeyError, IndexError, TypeError, ValueError):
        return getattr(row, key, None)


def _line(score: float, row) -> str:
    outcome = _field(row, "resolution_status") or "unknown outcome"
    notes = _field(row, "resolution_notes")
    actions = _field(row, "actions_json") or "[]"
    text = (
        f"[similarity {score:.2f}] {_field(row, 'pipeline_name')} "
        f"(agent {_field(row, 'classified_agent')}): {str(_field(row, 'diagnosis') or '')[:300]} "
        f"| fix: {str(_field(row, 'fix_suggestion') or '')[:200]} | actions: {actions[:200]} "
        f"| outcome: {outcome}"
    )
    return text + (f" ({str(notes)[:150]})" if notes else "")


# ------------------------------------------------------------------ recall

def recall(settings: Settings, spark, workspace_client, run: RunContext, exclude_incident: str = "",
           top_k: int = TOP_K, min_similarity: float = MIN_SIMILARITY):
    """Return (prompt lines, matches, embedding). Never raises."""
    try:
        vec = embed(workspace_client, incident_text(run))
    except Exception as exc:
        log.warning("memory: embedding failed, running without memory: %s", str(exc)[:200])
        return [], [], None

    try:
        rows = spark.sql(f"""
            SELECT incident_id, pipeline_name, classified_agent, diagnosis, fix_suggestion,
                   actions_json, resolution_status, resolution_notes, embedding
            FROM {settings.table('incident_memory')}
            WHERE embedding IS NOT NULL AND size(embedding) = {EMBEDDING_DIM}
            ORDER BY created_at DESC LIMIT {CANDIDATE_LIMIT}
        """).collect()
    except Exception as exc:
        log.warning("memory: lookup failed, running without memory: %s", str(exc)[:200])
        return [], [], vec

    scored = []
    for r in rows:
        if exclude_incident and _field(r, "incident_id") == exclude_incident:
            continue
        s = cosine(vec, list(_field(r, "embedding") or []))
        if s >= min_similarity:
            scored.append((s, r))
    scored.sort(key=lambda sr: -sr[0])
    top = scored[:top_k]

    lines = [_line(s, r) for s, r in top]
    matches = [{"incident_id": _field(r, "incident_id"), "similarity": round(s, 3)} for s, r in top]
    return lines, matches, vec


# ------------------------------------------------------------------ remember

def remember(settings: Settings, spark, incident_id: str, pipeline_name: str, run: RunContext,
             diagnosis, route=None, embedding: list[float] | None = None,
             matches: list[dict] | None = None, tools_used: list[str] | None = None,
             resolution_status: str = "", latency_sec: float = 0.0) -> bool:
    """Store (or replace) the memory row for one incident. Never raises."""
    if not _SAFE_ID.match(incident_id or ""):
        log.warning("memory: unsafe incident id %r, not stored", incident_id)
        return False
    if embedding is not None and len(embedding) != EMBEDDING_DIM:
        embedding = None
    try:
        from pyspark.sql.types import (
            ArrayType,
            BooleanType,
            DoubleType,
            FloatType,
            IntegerType,
            StringType,
            StructField,
            StructType,
        )

        schema = StructType([
            StructField("incident_id", StringType()),
            StructField("pipeline_name", StringType()),
            StructField("raw_error", StringType()),
            StructField("classified_agent", StringType()),
            StructField("classification_confidence", DoubleType()),
            StructField("diagnosis", StringType()),
            StructField("severity", IntegerType()),
            StructField("priority", StringType()),
            StructField("fix_suggestion", StringType()),
            StructField("fix_complexity", StringType()),
            StructField("requires_human", BooleanType()),
            StructField("actions_json", StringType()),
            StructField("tools_used_json", StringType()),
            StructField("resolution_status", StringType()),
            StructField("resolution_notes", StringType()),
            StructField("similar_incidents_json", StringType()),
            StructField("embedding", ArrayType(FloatType())),
            StructField("total_latency_sec", DoubleType()),
            StructField("embedding_model", StringType()),
        ])
        row = [(
            incident_id, pipeline_name, incident_text(run),
            diagnosis.agent_name, float(getattr(route, "confidence", 0.0) or 0.0),
            diagnosis.diagnosis, int(diagnosis.severity), diagnosis.fix_complexity,
            diagnosis.fix_suggestion, diagnosis.fix_complexity, bool(diagnosis.requires_human),
            json.dumps(diagnosis.actions), json.dumps(tools_used or []),
            resolution_status, None, json.dumps(matches or []),
            embedding, float(latency_sec),
            EMBEDDING_MODEL if embedding is not None else None,
        )]
        table = settings.table("incident_memory")
        spark.sql(f"DELETE FROM {table} WHERE incident_id = '{incident_id}'")
        df = spark.createDataFrame(row, schema=schema)
        df = df.selectExpr(*[f.name for f in schema.fields], "current_timestamp() AS created_at")
        df.write.format("delta").mode("append").saveAsTable(table)
        return True
    except Exception as exc:
        log.warning("memory: store failed for %s: %s", incident_id, str(exc)[:200])
        return False


def update_outcome(settings: Settings, spark, incident_id: str, status: str, notes: str = "") -> bool:
    """Record the final outcome of a remembered incident. Never raises."""
    if not _SAFE_ID.match(incident_id or "") or not _SAFE_ID.match(status or ""):
        return False
    notes_sql = "'" + str(notes)[:500].replace("\\", "\\\\").replace("'", "\\'") + "'" if notes else "NULL"
    try:
        spark.sql(f"""
            UPDATE {settings.table('incident_memory')}
            SET resolution_status = '{status}', resolution_notes = {notes_sql}
            WHERE incident_id = '{incident_id}'
        """)
        return True
    except Exception as exc:
        log.warning("memory: outcome update failed for %s: %s", incident_id, str(exc)[:200])
        return False