import time
from types import SimpleNamespace as NS

from agentic_ai.agents.base import ToolContext
from agentic_ai.agents.processing import ProcessingAgent, compute_profile, error_analysis, run_history
from agentic_ai.agents.trace import Trace
from agentic_ai.config import Settings
from agentic_ai.telemetry.run_context import RunContext, TaskError

NOW_MS = int(time.time() * 1000)
HOUR_MS = 3_600_000

APP_TRACE = (
    "ValueError                                Traceback (most recent call last)\n"
    "File <command-6276623696594002>, line 2\n"
    "      1 n = spark.table('databricks_ws.agentic_ai_dev.incidents').count()\n"
    "----> 2 raise ValueError(f'probe v2: read {n} incident rows, then failed on purpose')\n"
    "ValueError: probe v2: read 10 incident rows, then failed on purpose"
)
ENGINE_ERROR = (
    "pyspark.errors.exceptions.connect.AnalysisException: [TABLE_OR_VIEW_NOT_FOUND] "
    "The table or view `c`.`s`.`t` cannot be found. SQLSTATE: 42P01"
)


def _tctx(error="", trace="", attempts=1, w=None, job_id="1", run_id="100"):
    run = RunContext(job_id=job_id, run_id=run_id, job_name="orders_etl",
                     task_errors=[TaskError("load", "7", "FAILED", "notebook /x", error=error,
                                            error_trace=trace, attempts=attempts)])
    return ToolContext(settings=Settings(), run=run, workspace_client=w)


def _run(run_id, state, start_h_ago, dur_s):
    start = NOW_MS - int(start_h_ago * HOUR_MS)
    return NS(run_id=run_id, start_time=start, end_time=start + dur_s * 1000,
              state=NS(result_state=NS(value=state)))


# ------------------------------------------------------------ error_analysis

def test_error_analysis_identifies_application_code():
    summary, data = error_analysis(_tctx(error="ValueError: probe", trace=APP_TRACE, attempts=2))
    assert "RAISED BY APPLICATION CODE" in summary
    assert "raise ValueError" in data["load"]["user_line"]
    assert "not transient" in summary


def test_error_analysis_identifies_engine_error_class_and_sqlstate():
    summary, data = error_analysis(_tctx(error=ENGINE_ERROR))
    assert "raised by the Spark engine" in summary
    assert data["load"]["error_class"] == "TABLE_OR_VIEW_NOT_FOUND"
    assert data["load"]["sqlstate"] == "42P01"
    assert data["load"]["exception"].endswith("AnalysisException")


def test_error_analysis_without_error_text():
    run = RunContext(job_id="1", run_id="2", job_name="x")
    summary, _ = error_analysis(ToolContext(settings=Settings(), run=run))
    assert "No error text" in summary


# ------------------------------------------------------------ run_history

class FakeJobs:
    def __init__(self, runs=(), run=None):
        self.runs, self.run = list(runs), run

    def list_runs(self, **kwargs):
        return iter(self.runs)

    def get_run(self, run_id):
        return self.run


def test_first_failure_after_successes_signals_a_recent_change():
    runs = [_run(100, "FAILED", 1, 600)] + [_run(90 - i, "SUCCESS", 24 * (i + 1), 120) for i in range(5)]
    summary, data = run_history(_tctx(w=NS(jobs=FakeJobs(runs))))
    assert "FIRST FAILURE after 5 recent successes" in summary
    assert "600s vs typical successful 120s" in summary
    assert data["failure_streak"] == 1


def test_consistent_and_never_succeeded_patterns():
    failing = [_run(100 - i, "FAILED", i + 1, 60) for i in range(3)] + [_run(50, "SUCCESS", 100, 60)]
    assert "FAILING CONSISTENTLY: last 3 runs failed" in run_history(_tctx(w=NS(jobs=FakeJobs(failing))))[0]

    never = [_run(100 - i, "FAILED", i + 1, 60) for i in range(4)]
    assert "NEVER SUCCEEDED" in run_history(_tctx(w=NS(jobs=FakeJobs(never))))[0]


# ------------------------------------------------------------ compute_profile

def test_compute_profile_serverless_and_job_cluster():
    serverless = NS(tasks=[NS(task_key="load", new_cluster=None, job_cluster_key=None,
                              existing_cluster_id=None)], job_clusters=[])
    summary, data = compute_profile(_tctx(w=NS(jobs=FakeJobs(run=serverless))))
    assert "serverless" in summary and data["load"]["kind"] == "serverless"

    spec = NS(node_type_id="Standard_DS3_v2", driver_node_type_id=None, num_workers=2,
              autoscale=None, spark_version="15.4.x", spark_conf={"spark.sql.shuffle.partitions": "8"})
    cluster = NS(tasks=[NS(task_key="load", new_cluster=None, job_cluster_key="main",
                           existing_cluster_id=None)],
                 job_clusters=[NS(job_cluster_key="main", new_cluster=spec)])
    summary, data = compute_profile(_tctx(w=NS(jobs=FakeJobs(run=cluster))))
    assert "Standard_DS3_v2" in summary and "spark.sql.shuffle.partitions=8" in summary


# ------------------------------------------------------------ agent

class FakeLLM:
    def __init__(self):
        self.prompts = []

    def chat_json(self, prompt, system_message="", max_tokens=0):
        self.prompts.append(prompt)
        return NS(ok=True, provider="fake", content="", parsed={
            "diagnosis": "Explicit raise in notebook.", "evidence_used": ["error_analysis"],
            "severity": 4, "fix_complexity": "LOW", "fix_suggestion": "Fix the code",
            "actions": [], "requires_human": True, "confidence": 0.95,
        })


def test_agent_uses_code_subagent_for_app_errors_and_grounds_prompt():
    runs = [_run(100, "FAILED", 1, 60), _run(99, "SUCCESS", 25, 60)]
    serverless = NS(tasks=[NS(task_key="load", new_cluster=None, job_cluster_key=None,
                              existing_cluster_id=None)], job_clusters=[])
    w = NS(jobs=FakeJobs(runs, serverless))
    llm, trace = FakeLLM(), Trace("INC", "orders_etl")
    d = ProcessingAgent().run(llm, _tctx(error="ValueError: probe", trace=APP_TRACE, w=w), trace)

    assert d.toolkit_used == "processing/CODE_AND_CONFIG"
    assert [s.node_name for s in trace.steps] == [
        "select_subagent", "tool:error_analysis", "tool:run_history", "tool:compute_profile", "reason",
    ]
    assert "RAISED BY APPLICATION CODE" in llm.prompts[0]
    assert "FIRST FAILURE" in llm.prompts[0]


def test_oom_selects_memory_subagent():
    agent = ProcessingAgent()
    sub, hits = agent.select_subagent(_tctx(error="java.lang.OutOfMemoryError: Java heap space").run)
    assert sub.name == "MEMORY" and "outofmemory" in hits



def test_compute_profile_reports_retried_task_once():
    attempt = NS(task_key="load", new_cluster=None, job_cluster_key=None, existing_cluster_id=None)
    run = NS(tasks=[attempt, attempt], job_clusters=[])
    summary, _ = compute_profile(_tctx(w=NS(jobs=FakeJobs(run=run))))
    assert summary.count("task 'load'") == 1



def test_failing_line_comes_from_user_frame_not_library():
    trace = (
        "URLError                                  Traceback (most recent call last)\n"
        "File <command-123>, line 2\n"
        "      1 import urllib.request\n"
        "----> 2 urllib.request.urlopen('https://orders-api.example.invalid/v1', timeout=10)\n"
        "File /usr/lib/python3.12/urllib/request.py:215, in urlopen(url)\n"
        "--> 215 return opener.open(url, data, timeout)\n"
        "File /usr/lib/python3.12/urllib/request.py:1347, in do_open\n"
        "-> 1347 raise URLError(err)\n"
        "URLError: <urlopen error [Errno -2] Name or service not known>"
    )
    summary, data = error_analysis(_tctx(error="URLError", trace=trace))
    assert data["load"]["user_line"].startswith("urllib.request.urlopen(")
    assert "raised by a library called from the job's code" in summary