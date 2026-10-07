"""
Orchestrator: one incident in, one grounded Diagnosis out.

  1. build_run_context  real task errors, stack traces, tables
  2. route              rules first, LLM only when the signals are ambiguous
  3. recall             similar past incidents from memory (optional)
  4. agent.run          sub-agent selection, read-only tools, one LLM call
  5. trace.flush        every step to subagent_execution_log

The returned Diagnosis is the same object the Phase 2 diagnose() produced,
so approval routing, execution, email and GitHub are unchanged. The caller
stores memory once it knows the initial outcome (see remember_outcome).

Any exception propagates; the reconciler then falls back to the Phase 2
single-call diagnosis, so an agent bug never leaves an incident undiagnosed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from agentic_ai.agents.analytics import AnalyticsAgent
from agentic_ai.agents.base import BaseAgent, ToolContext
from agentic_ai.agents.ingestion import IngestionAgent
from agentic_ai.agents.processing import ProcessingAgent
from agentic_ai.agents.router import Route, route
from agentic_ai.agents.storage import StorageAgent
from agentic_ai.agents.trace import Trace
from agentic_ai.config import Settings
from agentic_ai.llm.client import LLMClient
from agentic_ai.memory import store
from agentic_ai.remediation.diagnosis import Diagnosis
from agentic_ai.telemetry.run_context import RunContext, build_run_context

log = logging.getLogger(__name__)

AGENTS: dict[str, BaseAgent] = {
    "storage": StorageAgent(),
    "processing": ProcessingAgent(),
    "ingestion": IngestionAgent(),
    "analytics": AnalyticsAgent(),
}
DEFAULT_AGENT = "processing"
ANALYSIS_LINE_CHARS = 300


@dataclass
class AgentOutcome:
    diagnosis: Diagnosis
    run: RunContext
    route: Route
    trace: Trace
    memory_lines: list[str] = field(default_factory=list)
    memory_matches: list[dict] = field(default_factory=list)
    embedding: list[float] | None = None
    latency_sec: float = 0.0

    @property
    def tools_used(self) -> list[str]:
        return [s.node_name.split(":", 1)[1] for s in self.trace.steps if s.node_name.startswith("tool:")]

    def analysis_lines(self) -> list[str]:
        """Short, human-readable account of how the diagnosis was reached,
        for the GitHub issue, the email and the App."""
        lines = [f"Routed to the {self.route.agent} agent ({self.route.method}): {self.route.reason}"]
        for s in self.trace.steps:
            if s.node_name == "select_subagent":
                lines.append(f"Sub-agent: {s.detail}")
            elif s.node_name.startswith("tool:"):
                lines.append(s.detail[:ANALYSIS_LINE_CHARS])
        if self.memory_lines:
            lines.append(f"{len(self.memory_lines)} similar past incident(s) were considered:")
            lines += [f"  {m[:ANALYSIS_LINE_CHARS]}" for m in self.memory_lines]
        else:
            lines.append("No sufficiently similar past incidents in memory.")
        return lines


def diagnose_incident(settings: Settings, spark, llm: LLMClient, workspace_client,
                      incident: dict) -> AgentOutcome:
    start = time.time()
    incident_id = incident["incident_id"]
    pipeline = incident.get("pipeline_name") or ""
    trace = Trace(incident_id, pipeline)

    ctx = build_run_context(settings, incident, workspace_client=workspace_client, spark=spark)
    r = route(ctx, llm, trace)
    agent = AGENTS.get(r.agent) or AGENTS[DEFAULT_AGENT]

    memory_lines, matches, embedding = store.recall(
        settings, spark, workspace_client, ctx, exclude_incident=incident_id
    )
    if matches:
        trace.add("memory", ", ".join(f"{m['incident_id']} ({m['similarity']})" for m in matches))

    d = agent.run(llm, ToolContext(settings, ctx, spark=spark, workspace_client=workspace_client),
                  trace, memory_lines)
    trace.flush(settings, spark)

    return AgentOutcome(
        diagnosis=d, run=ctx, route=r, trace=trace, memory_lines=memory_lines,
        memory_matches=matches, embedding=embedding, latency_sec=round(time.time() - start, 2),
    )


def remember_outcome(settings: Settings, spark, incident: dict, outcome: AgentOutcome, status: str) -> bool:
    """Store the incident in memory with its initial outcome. Never raises."""
    return store.remember(
        settings, spark, incident["incident_id"], incident.get("pipeline_name") or "",
        outcome.run, outcome.diagnosis, route=outcome.route, embedding=outcome.embedding,
        matches=outcome.memory_matches, tools_used=outcome.tools_used,
        resolution_status=status, latency_sec=outcome.latency_sec,
    )