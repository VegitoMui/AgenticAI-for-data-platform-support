"""
Agent framework.

Every specialist agent (storage, ingestion, processing, analytics) has the
same shape and runs the same four steps, each recorded in the trace:

  1. select_subagent  Score the incident text against each sub-agent's
                      keywords and pick the best match (first sub-agent is the
                      default when nothing matches). The trace records which
                      keywords matched, so the choice is explainable.
  2. gather           Run that sub-agent's tools. A tool is a plain function
                      that inspects real telemetry and returns a short summary
                      plus raw data. A tool that fails becomes failed evidence;
                      it never stops the other tools.
  3. reason           One LLM call over the incident + evidence (+ similar
                      past incidents once memory exists), which must ground the
                      diagnosis in that evidence.
  4. return           A Diagnosis -- the same object the reconciler already
                      uses, so approvals, execution and email are unchanged.

A concrete agent only declares its sub-agents and tools.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from agentic_ai.agents.trace import Trace
from agentic_ai.config import Settings
from agentic_ai.llm.client import LLMClient
from agentic_ai.remediation.diagnosis import VALID_COMPLEXITY, Diagnosis
from agentic_ai.telemetry.run_context import RunContext

log = logging.getLogger(__name__)

MAX_EVIDENCE_CHARS = 1200      # per tool, in the prompt
MAX_EVIDENCE_TOTAL = 8000      # all tools together, in the prompt


# ------------------------------------------------------------------ types

@dataclass
class ToolContext:
    """Everything a tool may use. Tools must treat it as read-only."""
    settings: Settings
    run: RunContext
    spark: object = None
    workspace_client: object = None

    @property
    def tables(self) -> list[str]:
        return self.run.tables


# A tool returns (summary for the LLM, raw data for the trace/tests).
ToolFn = Callable[[ToolContext], tuple[str, dict]]


@dataclass
class Tool:
    name: str
    description: str
    fn: ToolFn


@dataclass
class Evidence:
    tool: str
    ok: bool
    summary: str
    data: dict = field(default_factory=dict)
    duration_sec: float = 0.0

    def to_prompt(self) -> str:
        status = "ok" if self.ok else "FAILED"
        text = self.summary if len(self.summary) <= MAX_EVIDENCE_CHARS else (
            self.summary[:MAX_EVIDENCE_CHARS] + " ...[truncated]"
        )
        return f"[{self.tool}] ({status}) {text}"


@dataclass
class SubAgent:
    name: str
    focus: str
    keywords: tuple[str, ...]
    tools: tuple[str, ...]


# ------------------------------------------------------------------ parsing

def diagnosis_from_result(result, agent_name: str) -> Diagnosis:
    """Turn an LLM JSON result into a Diagnosis. Anything missing or invalid
    degrades to the safe side: HIGH complexity, requires a human."""
    if not getattr(result, "ok", False) or not getattr(result, "parsed", None):
        return Diagnosis(
            agent_name=agent_name,
            diagnosis="Diagnosis unavailable -- LLM call failed or returned unparseable output.",
            severity=5,
            fix_complexity="HIGH",
            fix_suggestion="Manual review required.",
            requires_human=True,
            actions=[],
            confidence=0.0,
            provider=getattr(result, "provider", "unknown"),
        )

    p = result.parsed
    complexity = str(p.get("fix_complexity", "HIGH")).upper()
    if complexity not in VALID_COMPLEXITY:
        complexity = "HIGH"
    try:
        severity = max(1, min(10, int(p.get("severity", 5))))
    except (TypeError, ValueError):
        severity = 5
    try:
        confidence = max(0.0, min(1.0, float(p.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.0

    return Diagnosis(
        agent_name=agent_name,
        diagnosis=str(p.get("diagnosis", ""))[:800],
        severity=severity,
        fix_complexity=complexity,
        fix_suggestion=str(p.get("fix_suggestion", ""))[:500],
        requires_human=bool(p.get("requires_human", True)),
        actions=[str(a) for a in (p.get("actions") or []) if str(a).strip()][:10],
        confidence=confidence,
        provider=getattr(result, "provider", "unknown"),
    )


# ------------------------------------------------------------------ agent

_SYSTEM = "You are a Databricks platform support engineer. Return ONLY valid JSON."

_RULES = """RULES
- Base the diagnosis on the incident and the evidence above. Name the evidence you relied on.
- Use real table names from the evidence. Never use placeholders such as your_table.
- An executable action is exactly one SQL statement (OPTIMIZE, VACUUM, ANALYZE, ALTER TABLE ...)
  or spark.conf.set('key', 'value'). Anything else is advice for a person.
- If the failure comes from application code (a bug, bad input, an explicit raise), say so,
  propose no data-platform changes, and set requires_human to true.
- If a table is missing, the usual cause is a wrong reference in the job, not a table that should
  be created. If a similarly named table exists, name it. Never propose CREATE or DROP TABLE.
- Set requires_human to true for destructive actions (DROP, TRUNCATE, DELETE, VACUUM with
  retention under 168 hours) or whenever you are not confident."""

_SCHEMA = """Return ONLY this JSON:
{{
  "agent": "{agent}",
  "diagnosis": "root cause in one to three sentences",
  "evidence_used": ["tool names you relied on"],
  "severity": 1-10,
  "fix_complexity": "LOW" | "MEDIUM" | "HIGH",
  "fix_suggestion": "one sentence summary of the fix",
  "actions": ["action 1", "action 2"],
  "requires_human": true | false,
  "confidence": 0.0-1.0
}}"""


class BaseAgent:
    name: str = "base"
    description: str = ""
    sub_agents: list[SubAgent] = []
    tools: dict[str, Tool] = {}

    # -------------------------------------------------------- 1. selection

    def select_subagent(self, run: RunContext) -> tuple[SubAgent, list[str]]:
        text = f"{run.job_name}\n{run.to_prompt()}".lower()
        best, best_hits = self.sub_agents[0], []
        for sub in self.sub_agents:
            hits = [k for k in sub.keywords if k.lower() in text]
            if len(hits) > len(best_hits):
                best, best_hits = sub, hits
        return best, best_hits

    # -------------------------------------------------------- 2. evidence

    def gather(self, tctx: ToolContext, sub: SubAgent, trace: Trace) -> list[Evidence]:
        evidence = []
        for tool_name in sub.tools:
            tool = self.tools.get(tool_name)
            if tool is None:
                trace.add(f"tool:{tool_name}", "not registered on this agent", sub_agent=sub.name)
                continue
            start = time.time()
            try:
                summary, data = tool.fn(tctx)
                ev = Evidence(tool_name, True, summary or "(no findings)", data or {})
            except Exception as exc:
                ev = Evidence(tool_name, False, f"tool failed: {str(exc)[:300]}")
            ev.duration_sec = round(time.time() - start, 3)
            evidence.append(ev)
            trace.add(f"tool:{tool_name}", ev.to_prompt(), sub_agent=sub.name, tools=tool_name,
                      duration_sec=ev.duration_sec)
        return evidence

    # -------------------------------------------------------- 3. reasoning

    def build_prompt(self, run: RunContext, sub: SubAgent, evidence: list[Evidence],
                     memory: list[str] | None = None) -> str:
        blocks, used = [], 0
        for ev in evidence:
            text = ev.to_prompt()
            if used + len(text) > MAX_EVIDENCE_TOTAL:
                blocks.append("[further evidence omitted for length]")
                break
            blocks.append(text)
            used += len(text)

        parts = [
            f"You are the {self.name} specialist agent diagnosing a failed Databricks job.",
            f"Your focus ({sub.name}): {sub.focus}",
            "",
            "INCIDENT",
            run.to_prompt(),
            "",
            "EVIDENCE GATHERED BY YOUR TOOLS",
            "\n".join(blocks) or "No evidence could be gathered.",
        ]
        if memory:
            parts += ["", "SIMILAR PAST INCIDENTS AND HOW THEY WERE RESOLVED", "\n".join(memory)]
        parts += ["", _RULES, "", _SCHEMA.format(agent=self.name)]
        return "\n".join(parts)

    def reason(self, llm: LLMClient, run: RunContext, sub: SubAgent, evidence: list[Evidence],
               trace: Trace, memory: list[str] | None = None) -> Diagnosis:
        start = time.time()
        result = llm.chat_json(
            self.build_prompt(run, sub, evidence, memory),
            system_message=_SYSTEM,
            max_tokens=2000,
        )
        d = diagnosis_from_result(result, self.name)
        d.toolkit_used = f"{self.name}/{sub.name}"

        used = []
        if getattr(result, "parsed", None):
            used = [str(u) for u in (result.parsed.get("evidence_used") or [])]
        trace.add(
            "reason",
            f"{d.fix_complexity}, confidence {d.confidence:.2f}, {len(d.actions)} action(s), "
            f"evidence used: {', '.join(used) or 'none cited'}. {d.fix_suggestion}",
            sub_agent=sub.name,
            duration_sec=time.time() - start,
        )
        return d

    # -------------------------------------------------------- run

    def run(self, llm: LLMClient, tctx: ToolContext, trace: Trace,
            memory: list[str] | None = None) -> Diagnosis:
        sub, hits = self.select_subagent(tctx.run)
        why = f"matched {', '.join(hits)}" if hits else "no keywords matched, using default"
        trace.add("select_subagent", f"{self.name}/{sub.name} ({why})", sub_agent=sub.name)

        evidence = self.gather(tctx, sub, trace)
        return self.reason(llm, tctx.run, sub, evidence, trace, memory)