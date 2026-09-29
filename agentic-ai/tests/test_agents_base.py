from types import SimpleNamespace as NS

import pytest

from agentic_ai.agents.base import BaseAgent, SubAgent, Tool, ToolContext, diagnosis_from_result
from agentic_ai.agents.trace import Trace
from agentic_ai.config import Settings
from agentic_ai.telemetry.run_context import RunContext, TaskError


def _ok_tool(tctx):
    return f"table {tctx.tables[0]} has 18400 files", {"files": 18400}


def _broken_tool(tctx):
    raise RuntimeError("permission denied")


class ToyAgent(BaseAgent):
    name = "storage"
    sub_agents = [
        SubAgent("TABLE_HEALTH", "table metadata", ("corrupt",), ("detail",)),
        SubAgent("OPTIMIZATION", "file layout", ("small files", "optimize"), ("detail", "broken", "missing")),
    ]
    tools = {
        "detail": Tool("detail", "DESCRIBE DETAIL", _ok_tool),
        "broken": Tool("broken", "always fails", _broken_tool),
    }


class FakeLLM:
    def __init__(self, parsed):
        self.parsed = parsed
        self.prompts = []

    def chat_json(self, prompt, system_message="", max_tokens=0):
        self.prompts.append(prompt)
        return NS(ok=True, parsed=self.parsed, content="", provider="fake")


def _ctx(error):
    run = RunContext(job_id="1", run_id="2", job_name="orders_etl",
                     task_errors=[TaskError("load", "3", "FAILED", "notebook /x", error=error)],
                     tables_from_code=["main.sales.orders"])
    return ToolContext(settings=Settings(), run=run)


PARSED = {
    "diagnosis": "Too many small files on main.sales.orders.",
    "evidence_used": ["detail"],
    "severity": 6, "fix_complexity": "low", "fix_suggestion": "Compact the table",
    "actions": ["OPTIMIZE main.sales.orders"], "requires_human": False, "confidence": 0.85,
}


def test_selects_subagent_by_keywords_and_records_why():
    trace = Trace("INC-1", "orders_etl")
    d = ToyAgent().run(FakeLLM(PARSED), _ctx("Query slow: too many small files"), trace)

    assert d.toolkit_used == "storage/OPTIMIZATION"
    assert trace.steps[0].node_name == "select_subagent"
    assert "matched small files" in trace.steps[0].detail


def test_default_subagent_when_nothing_matches():
    trace = Trace("INC-1", "orders_etl")
    d = ToyAgent().run(FakeLLM(PARSED), _ctx("something odd"), trace)
    assert d.toolkit_used == "storage/TABLE_HEALTH"
    assert "using default" in trace.steps[0].detail


def test_failed_and_missing_tools_do_not_stop_the_agent():
    trace = Trace("INC-1", "orders_etl")
    llm = FakeLLM(PARSED)
    d = ToyAgent().run(llm, _ctx("small files everywhere"), trace)

    names = [s.node_name for s in trace.steps]
    assert names == ["select_subagent", "tool:detail", "tool:broken", "tool:missing", "reason"]
    assert "FAILED" in trace.steps[2].detail and "permission denied" in trace.steps[2].detail
    # evidence reaches the prompt, and the result is normalised
    assert "18400 files" in llm.prompts[0]
    assert d.fix_complexity == "LOW" and d.confidence == 0.85
    assert "evidence used: detail" in trace.steps[-1].detail


def test_unparseable_llm_result_degrades_to_human_review():
    d = diagnosis_from_result(NS(ok=False, parsed=None, provider="all_failed"), "storage")
    assert d.requires_human and d.fix_complexity == "HIGH" and d.confidence == 0.0


def test_out_of_range_values_are_clamped():
    d = diagnosis_from_result(
        NS(ok=True, parsed={"severity": 42, "confidence": 7, "fix_complexity": "?"}, provider="p"), "x",
    )
    assert d.severity == 10 and d.confidence == 1.0 and d.fix_complexity == "HIGH"


def test_trace_timed_records_failures_and_reraises():
    trace = Trace("INC-1", "p")
    with pytest.raises(ValueError):
        with trace.timed("gather_context"):
            raise ValueError("boom")
    assert trace.steps[0].detail.startswith("FAILED: boom")


def test_trace_flush_failure_is_swallowed():
    trace = Trace("INC-1", "p")
    trace.add("step", "detail")
    broken_spark = NS(createDataFrame=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no spark")))
    assert trace.flush(Settings(), broken_spark) is False