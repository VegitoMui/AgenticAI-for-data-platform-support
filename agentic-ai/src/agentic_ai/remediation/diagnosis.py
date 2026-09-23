"""
Phase 2 diagnosis: LLM-only classification and fix suggestion.

This is intentionally NOT the full toolkit-backed diagnosis (table stats,
OPTIMIZE history, quality checks, RAG memory). That is Phase 3 scope. This
step exists so the reconciler has a real decision to make about approval vs.
auto-execution, using the LLM's own stated confidence and judgement rather
than toolkit evidence. It is replaced, not wrapped, when Phase 3 lands.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from agentic_ai.llm.client import LLMClient

log = logging.getLogger(__name__)

VALID_AGENTS = ("storage", "ingestion", "processing", "analytics")
VALID_COMPLEXITY = ("LOW", "MEDIUM", "HIGH")


@dataclass
class Diagnosis:
    agent_name: str
    diagnosis: str
    severity: int
    fix_complexity: str
    fix_suggestion: str
    requires_human: bool
    actions: list[str]
    confidence: float
    provider: str
    toolkit_used: str = "llm_only_placeholder"


_PROMPT = """You are diagnosing a failed Databricks pipeline run and proposing a fix.

AGENT CATEGORIES:
- processing: OOM, heap space, GC overhead, shuffle spill, data skew,
  checkpoint failure, streaming failure, executor lost
- ingestion: connection timeout, rate limit, schema drift, duplicate records,
  source unreachable, API failure
- storage: small files, OPTIMIZE, VACUUM, corrupt partition, stale data,
  schema evolution, table properties
- analytics: slow query, full table scan, no predicate pushdown, dashboard
  timeout, permissions, SQL performance

Error: {error_text}
Pipeline: {pipeline_name}

Propose concrete fix actions as SQL statements (OPTIMIZE, VACUUM, ALTER TABLE,
spark.conf.set(...)) where you can, or plain-English instructions where a human
must decide (e.g. "review and confirm before dropping table X").

Set requires_human=true if you are not confident the fix is safe to run
automatically, if it is destructive (DROP, TRUNCATE, data-losing VACUUM), or if
the diagnosis itself is uncertain.

Return ONLY JSON:
{{
  "agent": "storage" | "ingestion" | "processing" | "analytics",
  "diagnosis": "one or two sentence explanation of the root cause",
  "severity": 1-10,
  "fix_complexity": "LOW" | "MEDIUM" | "HIGH",
  "fix_suggestion": "one sentence summary of the proposed fix",
  "actions": ["concrete action 1", "concrete action 2"],
  "requires_human": true | false,
  "confidence": 0.0-1.0
}}"""


def diagnose(llm: LLMClient, raw_error: str, pipeline_name: str) -> Diagnosis:
    result = llm.chat_json(
        _PROMPT.format(error_text=raw_error, pipeline_name=pipeline_name),
        system_message="Pipeline diagnosis assistant. Return ONLY valid JSON.",
        max_tokens=800,
    )

    if not result.ok or not result.parsed:
        log.warning("diagnosis failed or unparseable: %s", result.content[:200])
        return Diagnosis(
            agent_name="storage",
            diagnosis="Diagnosis unavailable -- LLM call failed or returned unparseable output.",
            severity=5,
            fix_complexity="HIGH",
            fix_suggestion="Manual review required.",
            requires_human=True,
            actions=[],
            confidence=0.0,
            provider=result.provider,
        )

    p = result.parsed
    agent = p.get("agent", "storage")
    if agent not in VALID_AGENTS:
        agent = "storage"
    complexity = p.get("fix_complexity", "HIGH")
    if complexity not in VALID_COMPLEXITY:
        complexity = "HIGH"

    return Diagnosis(
        agent_name=agent,
        diagnosis=str(p.get("diagnosis", ""))[:500],
        severity=int(p.get("severity", 5)),
        fix_complexity=complexity,
        fix_suggestion=str(p.get("fix_suggestion", ""))[:500],
        requires_human=bool(p.get("requires_human", True)),
        actions=[str(a) for a in p.get("actions", [])][:10],
        confidence=float(p.get("confidence", 0.5)),
        provider=result.provider,
    )


def needs_approval(d: Diagnosis, confidence_threshold: float = 0.6) -> bool:
    """Approval is required only when the agent itself signals uncertainty --
    not on every request. Three independent triggers, any one is enough:
      - the LLM explicitly said requires_human
      - the fix is classified HIGH complexity
      - the LLM's own confidence in the diagnosis is below threshold
    """
    return (
        d.requires_human
        or d.fix_complexity == "HIGH"
        or d.confidence < confidence_threshold
    )