import pytest

from agentic_ai.agents import orchestrator
from agentic_ai.agents.router import Route
from agentic_ai.config import Settings
from agentic_ai.notify.mailer import decision_needed, fix_applied
from agentic_ai.remediation import reconciler
from agentic_ai.remediation.diagnosis import Diagnosis
from agentic_ai.telemetry.run_context import RunContext

INCIDENT = {"incident_id": "INC-1", "job_id": "11", "run_id": "22", "pipeline_name": "orders",
            "raw_error": "boom"}


def _diag(agent="analytics"):
    return Diagnosis(agent_name=agent, diagnosis="typo in column", severity=4, fix_complexity="LOW",
                     fix_suggestion="fix the column name", requires_human=True, actions=[],
                     confidence=0.9, provider="fake", toolkit_used=f"{agent}/QUERY_ERRORS")


class FakeAgent:
    def __init__(self):
        self.memory = None

    def run(self, llm, tctx, trace, memory):
        self.memory = memory
        trace.add("select_subagent", "analytics/QUERY_ERRORS (matched unresolved_column)")
        trace.add("tool:column_check", "[column_check] (ok) custmer_id NOT FOUND")
        return _diag()


@pytest.fixture
def wired(monkeypatch):
    agent = FakeAgent()
    flushed = []
    monkeypatch.setattr(orchestrator, "build_run_context",
                        lambda s, inc, workspace_client, spark: RunContext("11", "22", "orders"))
    monkeypatch.setattr(orchestrator, "route",
                        lambda ctx, llm, trace: Route("analytics", 0.8, "rules", "clear signals"))
    monkeypatch.setattr(orchestrator.store, "recall", lambda *a, **k: (
        ["[similarity 0.93] orders (agent analytics): typo | outcome: APPROVED_EXECUTED"],
        [{"incident_id": "INC-0", "similarity": 0.93}], [0.1]))
    monkeypatch.setitem(orchestrator.AGENTS, "analytics", agent)
    monkeypatch.setattr(orchestrator.Trace, "flush", lambda self, s, spark: flushed.append(self) or True)
    return agent, flushed


def test_diagnose_incident_runs_the_routed_agent_with_memory(wired):
    agent, flushed = wired
    out = orchestrator.diagnose_incident(Settings(), None, None, None, INCIDENT)

    assert out.diagnosis.agent_name == "analytics" and out.route.method == "rules"
    assert agent.memory[0].startswith("[similarity 0.93]")
    assert out.tools_used == ["column_check"]
    assert [s.node_name for s in out.trace.steps] == ["memory", "select_subagent", "tool:column_check"]
    assert len(flushed) == 1

    lines = out.analysis_lines()
    assert lines[0] == "Routed to the analytics agent (rules): clear signals"
    assert "Sub-agent: analytics/QUERY_ERRORS (matched unresolved_column)" in lines
    assert "[column_check] (ok) custmer_id NOT FOUND" in lines
    assert "1 similar past incident(s) were considered:" in lines


def test_unknown_route_uses_default_agent(wired, monkeypatch):
    monkeypatch.setattr(orchestrator, "route", lambda ctx, llm, trace: Route("weird", 0.1, "llm", "?"))
    default = FakeAgent()
    monkeypatch.setitem(orchestrator.AGENTS, orchestrator.DEFAULT_AGENT, default)
    orchestrator.diagnose_incident(Settings(), None, None, None, INCIDENT)
    assert default.memory is not None


def test_reconciler_falls_back_to_single_call_diagnosis(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("agent bug")

    monkeypatch.setattr(reconciler, "diagnose_incident", broken)
    monkeypatch.setattr(reconciler, "diagnose", lambda llm, err, pipe: _diag("processing"))
    d, outcome = reconciler._diagnose(Settings(), None, None, None, INCIDENT)
    assert d.agent_name == "processing" and outcome is None


def test_issue_body_and_emails_carry_the_analysis():
    analysis = ["Routed to the analytics agent (rules): clear signals", "[column_check] (ok) NOT FOUND"]
    body = reconciler._issue_body(INCIDENT, _diag(), analysis)
    assert "analytics/QUERY_ERRORS" in body and "**How the agent reached this:**" in body
    assert "[column_check] (ok) NOT FOUND" in body
    assert "How the agent reached this" not in reconciler._issue_body(INCIDENT, _diag())

    email = decision_needed(INCIDENT, _diag(), "APR-1", "https://app", 72, analysis=analysis)
    assert "How the agent reached this:\n  Routed to the analytics agent" in email.body
    email = fix_applied(INCIDENT, _diag(), [], analysis=analysis)
    assert "  [column_check] (ok) NOT FOUND" in email.body


def test_app_shows_only_the_latest_trace_pass():
    pytest.importorskip("fastapi")
    pytest.importorskip("jinja2")
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "agentic_ai" / "app"))
    import main

    old, new = "2026-10-07 10:00:00", "2026-10-07 11:00:00"
    rows = [
        {"step_number": 1, "node_name": "classify", "detail": "old pass", "logged_at": old},
        {"step_number": 2, "node_name": "tool:column_check", "detail": "[column_check] (ok) found",
         "logged_at": new},
        {"step_number": 1, "node_name": "classify", "detail": "analytics via rules", "logged_at": new},
    ]
    steps = main.shape_trace(rows)
    assert [s["label"] for s in steps] == ["Routing", "Tool: column_check"]
    assert steps[1]["detail"] == "(ok) found" and steps[1]["failed"] is False
    assert main.shape_trace([]) == []