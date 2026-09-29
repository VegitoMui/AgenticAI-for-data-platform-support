"""
Execution trace.

Every step an agent takes -- gathering context, choosing a sub-agent, running
a tool, reasoning -- is recorded as one row in subagent_execution_log, so any
diagnosis can be explained after the fact: what the agent looked at, what it
found, and how long each step took.

Steps are buffered in memory and written in a single append at the end of the
incident (flush), so tracing adds one write per incident, not one per step.
A failed flush is logged and swallowed: tracing must never block remediation.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from agentic_ai.config import Settings

log = logging.getLogger(__name__)

MAX_DETAIL_CHARS = 2000


@dataclass
class TraceStep:
    step_number: int
    node_name: str
    detail: str
    sub_agent_selected: str = ""
    tools_run: str = ""
    duration_sec: float = 0.0


@dataclass
class Trace:
    incident_id: str
    pipeline_name: str
    steps: list[TraceStep] = field(default_factory=list)

    def add(self, node_name: str, detail: str, *, sub_agent: str = "", tools: str = "",
            duration_sec: float = 0.0) -> TraceStep:
        step = TraceStep(
            step_number=len(self.steps) + 1,
            node_name=node_name,
            detail=(detail or "")[:MAX_DETAIL_CHARS],
            sub_agent_selected=sub_agent,
            tools_run=tools,
            duration_sec=round(duration_sec, 3),
        )
        self.steps.append(step)
        log.info("[%s] step %s %s: %s", self.incident_id, step.step_number, node_name, step.detail[:160])
        return step

    @contextmanager
    def timed(self, node_name: str, *, sub_agent: str = "", tools: str = ""):
        """Time a block and record it. Set `holder['detail']` inside the block
        to describe the outcome; exceptions are recorded, then re-raised."""
        holder = {"detail": ""}
        start = time.time()
        try:
            yield holder
        except Exception as exc:
            holder["detail"] = f"FAILED: {str(exc)[:300]}"
            raise
        finally:
            self.add(node_name, holder["detail"], sub_agent=sub_agent, tools=tools,
                     duration_sec=time.time() - start)

    def summary(self) -> str:
        return " > ".join(s.node_name for s in self.steps)

    def flush(self, settings: Settings, spark) -> bool:
        if not self.steps:
            return True
        try:
            from pyspark.sql.types import (
                DoubleType,
                IntegerType,
                StringType,
                StructField,
                StructType,
            )

            schema = StructType([
                StructField("incident_id", StringType()),
                StructField("pipeline_name", StringType()),
                StructField("step_number", IntegerType()),
                StructField("node_name", StringType()),
                StructField("detail", StringType()),
                StructField("sub_agent_selected", StringType()),
                StructField("tools_run", StringType()),
                StructField("duration_sec", DoubleType()),
            ])
            rows = [(
                self.incident_id, self.pipeline_name, s.step_number, s.node_name, s.detail,
                s.sub_agent_selected, s.tools_run, float(s.duration_sec),
            ) for s in self.steps]
            df = spark.createDataFrame(rows, schema=schema)
            df = df.selectExpr(*[f.name for f in schema.fields], "current_timestamp() AS logged_at")
            df.write.format("delta").mode("append").saveAsTable(settings.table("subagent_execution_log"))
            return True
        except Exception as exc:
            log.warning("trace flush failed for %s: %s", self.incident_id, str(exc)[:200])
            return False