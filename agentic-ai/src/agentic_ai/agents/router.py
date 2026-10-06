"""
Router: decides which specialist agent handles an incident.

Two stages, so the common cases cost nothing and the hard ones get judgement:

  1. Rules.  Score the real error, stack trace and task type against weighted
             signals for each agent. Strong signals (e.g. OutOfMemoryError,
             ConcurrentAppendException, HTTP 429) score 3, weaker ones 1. If
             one agent clearly wins, it is chosen with no LLM call.
  2. LLM.    If the result is ambiguous (low top score, or a close second),
             one short LLM call decides, with the rule scores as hints.
             If that call fails, the top rule score wins, and with no signal
             at all the default is processing -- job failures most often
             originate in the code or compute that ran them.

Every decision is recorded in the trace with its method and reason.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from agentic_ai.agents.trace import Trace
from agentic_ai.llm.client import LLMClient
from agentic_ai.telemetry.run_context import RunContext

log = logging.getLogger(__name__)

AGENTS = ("storage", "ingestion", "processing", "analytics")
DEFAULT_AGENT = "processing"

STRONG, WEAK = 3, 1
MIN_CLEAR_SCORE = 3     # top score needed to skip the LLM
MIN_CLEAR_MARGIN = 2    # lead over the runner-up needed to skip the LLM

# (pattern, weight). Patterns are case-insensitive regexes over the error,
# stack trace and job name. Short tokens use word boundaries: Spark Connect
# stack traces are full of library paths (grpc ssl_credentials, jdbc drivers)
# that would otherwise score for the wrong agent.
SIGNALS: dict[str, list[tuple[str, int]]] = {
    "processing": [
        (r"outofmemory|java heap space|gc overhead", STRONG),
        (r"executor(s)? lost|executorlostfailure|container killed", STRONG),
        (r"shuffle|spill|data skew|skewed", STRONG),
        (r"streamingqueryexception|checkpoint", STRONG),
        (r"sparkexception|job aborted due to stage failure", WEAK),
        (r"py4jjavaerror|traceback \(most recent call last\)", WEAK),
        (r"spark\.conf|driver.*(crash|restart)", WEAK),
    ],
    "storage": [
        (r"concurrentappendexception|concurrentmodification|concurrentdelete", STRONG),
        (r"deltaanalysisexception|protocolchangedexception|delta_\w+", STRONG),
        (r"table_or_view_not_found|table or view not found|table .* does not exist", STRONG),
        (r"small files|too many files|optimize|vacuum|zorder", STRONG),
        (r"corrupt|checksum|_delta_log|file not found in delta", STRONG),
        (r"merge into|schema (evolution|mismatch)|cannot write to", WEAK),
        (r"partition", WEAK),
    ],
    "ingestion": [
        (r"\b429\b|rate limit|too many requests|throttl", STRONG),
        (r"connection (refused|reset|timed out)|unreachable|unknownhost|name resolution", STRONG),
        (r"name or service not known|nodename nor servname|getaddrinfo failed|urlopen error", STRONG),
        (r"sockettimeout|read timed out|certificate (verify|expired)", STRONG),
        (r"\bssl(exception| handshake|handshakeexception)\b", STRONG),
        (r"\bcloudfiles\b|\bauto ?loader\b|\bjdbc\b|\bkafka\b|\beventhubs?\b", STRONG),
        (r"path does not exist|filenotfoundexception|no such file", WEAK),
        (r"\bapi\b|http(s)?://|status code", WEAK),
    ],
    "analytics": [
        (r"insufficient_permissions|permission_denied|access denied|forbidden|\b403\b", STRONG),
        (r"sql warehouse|dashboard|lakeview|genie", STRONG),
        (r"full (table )?scan|query (timed out|timeout)|statement timeout", STRONG),
        (r"unresolved_(column|routine|field|map_key)|cannot be resolved|cannot resolve\s+['`]", STRONG),
        (r"parse_syntax_error|datatype_mismatch|cast_invalid_input|ambiguous_reference", STRONG),
        (r"sqlstate:?\s*(42703|42601|42883|42k09)", WEAK),
    ],
}

_COMPILED = {agent: [(re.compile(p, re.I), w) for p, w in sigs] for agent, sigs in SIGNALS.items()}


@dataclass
class Route:
    agent: str
    confidence: float
    method: str          # rules | llm | fallback | default
    reason: str
    scores: dict[str, int] = field(default_factory=dict)


def score(run: RunContext) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Weighted signal score per agent, plus the patterns that matched."""
    text = "\n".join([run.job_name] + [f"{t.error}\n{t.error_trace}" for t in run.task_errors])
    scores = {a: 0 for a in AGENTS}
    hits: dict[str, list[str]] = {a: [] for a in AGENTS}
    for agent, sigs in _COMPILED.items():
        for pattern, weight in sigs:
            m = pattern.search(text)
            if m:
                scores[agent] += weight
                hits[agent].append(m.group(0).lower())

    # Task type is a hint, not a verdict.
    for t in run.task_errors:
        if t.task_type.startswith("sql"):
            scores["analytics"] += WEAK
        elif t.task_type.startswith("pipeline"):
            scores["ingestion"] += WEAK
    return scores, hits


_PROMPT = """Route this failed Databricks job to exactly one specialist agent.

AGENTS
- processing: the code or compute that ran the job -- Spark errors, memory, skew, shuffle,
  streaming, and application code errors (bugs, bad input, explicit raises)
- storage: Delta tables themselves -- missing tables, concurrent writes, corruption,
  small files, OPTIMIZE / VACUUM, schema evolution on write
- ingestion: getting data in -- source connectivity, timeouts, rate limits, Auto Loader,
  JDBC / Kafka, missing landing files
- analytics: querying and access -- SQL warehouse queries, dashboards, permissions,
  slow or full-scan queries

INCIDENT
{incident}

Rule-based signal scores (hints only, may be wrong): {scores}

Return ONLY JSON:
{{"agent": "processing" | "storage" | "ingestion" | "analytics",
  "confidence": 0.0-1.0,
  "reason": "one sentence"}}"""


def route(run: RunContext, llm: LLMClient | None, trace: Trace) -> Route:
    start = time.time()
    scores, hits = score(run)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (top, top_score), (_, second_score) = ranked[0], ranked[1]
    matched = ", ".join(hits[top]) or "none"

    if top_score >= MIN_CLEAR_SCORE and top_score - second_score >= MIN_CLEAR_MARGIN:
        confidence = min(0.95, 0.6 + 0.05 * (top_score - second_score))
        r = Route(top, round(confidence, 2), "rules", f"clear signals: {matched}", scores)
    else:
        r = _route_with_llm(run, llm, scores, top, top_score, matched)

    matched_all = "; ".join(f"{a}: {', '.join(h)}" for a, h in hits.items() if h) or "none"
    trace.add(
        "classify",
        f"{r.agent} via {r.method} (confidence {r.confidence:.2f}): {r.reason} "
        f"| scores {r.scores} | matched {matched_all}",
        sub_agent=r.agent,
        duration_sec=time.time() - start,
    )
    return r


def _route_with_llm(run, llm, scores, top, top_score, matched) -> Route:
    if llm is not None:
        try:
            result = llm.chat_json(
                _PROMPT.format(incident=run.to_prompt(), scores=scores),
                system_message="You route incidents. Return ONLY valid JSON.",
                max_tokens=600,
            )
            p = result.parsed if result.ok else None
            if p and p.get("agent") in AGENTS:
                conf = max(0.0, min(1.0, float(p.get("confidence", 0.5))))
                return Route(p["agent"], round(conf, 2), "llm", str(p.get("reason", ""))[:300], scores)
        except Exception as exc:
            log.warning("router LLM call failed: %s", str(exc)[:200])

    if top_score > 0:
        return Route(top, 0.4, "fallback", f"LLM unavailable; strongest signals: {matched}", scores)
    return Route(DEFAULT_AGENT, 0.3, "default", "no signals and no LLM decision", scores)