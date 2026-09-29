from types import SimpleNamespace as NS

from agentic_ai.agents.router import route, score
from agentic_ai.agents.trace import Trace
from agentic_ai.telemetry.run_context import RunContext, TaskError


def _run(error, trace="", task_type="notebook /x"):
    return RunContext(job_id="1", run_id="2", job_name="nightly_load",
                      task_errors=[TaskError("t", "3", "FAILED", task_type, error=error, error_trace=trace)])


class FakeLLM:
    def __init__(self, parsed=None, ok=True):
        self.parsed, self.ok, self.calls = parsed, ok, 0

    def chat_json(self, prompt, system_message="", max_tokens=0):
        self.calls += 1
        return NS(ok=self.ok, parsed=self.parsed, content="", provider="fake")


def test_clear_signal_routes_by_rules_without_llm():
    llm = FakeLLM()
    trace = Trace("INC", "p")
    r = route(_run("java.lang.OutOfMemoryError: Java heap space"), llm, trace)

    assert r.agent == "processing" and r.method == "rules"
    assert llm.calls == 0
    assert trace.steps[0].node_name == "classify" and "via rules" in trace.steps[0].detail


def test_each_agent_has_a_clear_signal():
    cases = {
        "storage": "io.delta.exceptions.ConcurrentAppendException: files were added",
        "ingestion": "HTTP 429 Too Many Requests from source API",
        "analytics": "[INSUFFICIENT_PERMISSIONS] User does not have SELECT",
    }
    for expected, error in cases.items():
        assert route(_run(error), FakeLLM(), Trace("INC", "p")).agent == expected


def test_ambiguous_case_asks_the_llm():
    llm = FakeLLM({"agent": "storage", "confidence": 0.7, "reason": "table missing"})
    r = route(_run("something went wrong"), llm, Trace("INC", "p"))
    assert llm.calls == 1 and r.agent == "storage" and r.method == "llm"


def test_llm_failure_falls_back_to_strongest_signal_then_default():
    weak = _run("SparkException: Job aborted due to stage failure")  # processing, score 1 -> ambiguous
    r = route(weak, FakeLLM(ok=False), Trace("INC", "p"))
    assert r.agent == "processing" and r.method == "fallback"

    r = route(_run("unclear"), FakeLLM(ok=False), Trace("INC", "p"))
    assert r.agent == "processing" and r.method == "default"


def test_invalid_llm_agent_is_ignored():
    r = route(_run("unclear"), FakeLLM({"agent": "networking", "confidence": 0.9}), Trace("INC", "p"))
    assert r.method == "default"


def test_sql_task_type_nudges_analytics():
    scores, _ = score(_run("unclear", task_type="sql"))
    assert scores["analytics"] == 1



def test_missing_table_routes_to_storage_despite_grpc_ssl_in_trace():
    error = "[TABLE_OR_VIEW_NOT_FOUND] The table or view `c`.`s`.`t` cannot be found. SQLSTATE: 42P01"
    trace = "File /site-packages/grpc/_channel.py in ssl_credentials\nAnalysisException"
    r = route(_run(error, trace=trace), FakeLLM(), Trace("INC", "p"))
    assert r.agent == "storage" and r.method == "rules"
    assert r.scores["ingestion"] == 0


def test_real_ingestion_ssl_error_still_scores():
    scores, _ = score(_run("javax.net.ssl.SSLHandshakeException: certificate expired"))
    assert scores["ingestion"] >= 3