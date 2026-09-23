from agentic_ai.config import Settings


def test_table_normalises_regardless_of_prefix():
    s = Settings(catalog="databricks_ws", schema="agentic_ai")
    expected = "databricks_ws.agentic_ai.remediation_log"
    assert s.table("remediation_log") == expected
    assert s.table("agentic_ai.remediation_log") == expected
    assert s.table("databricks_ws.agentic_ai.remediation_log") == expected


def test_fq_schema():
    assert Settings(catalog="c", schema="s").fq_schema == "c.s"


# The two detection sources spell some states differently:
#   system.lakeflow.job_run_timeline -> SUCCEEDED, FAILED, ERROR, CANCELLED
#   Jobs API (SDK RunResultState)    -> SUCCESS, FAILED, TIMED_OUT, CANCELED
# failed_states is shared by both, so it must be a subset of the union.
VALID_RESULT_STATES = {
    "SUCCEEDED", "SUCCESS", "FAILED", "ERROR", "TIMED_OUT",
    "CANCELLED", "CANCELED",
    "EXCLUDED", "MAXIMUM_CONCURRENT_RUNS_REACHED",
}


def test_failed_states_are_valid_result_states():
    assert set(Settings().failed_states) <= VALID_RESULT_STATES


def test_failed_states_cover_both_sources():
    states = set(Settings().failed_states)
    # system tables vocabulary
    assert {"FAILED", "ERROR", "CANCELLED"} <= states
    # Jobs API vocabulary
    assert {"FAILED", "TIMED_OUT", "CANCELED"} <= states


def test_detection_source_defaults_to_auto():
    assert Settings().detection_source == "auto"


def test_approval_expiry_is_72_hours():
    assert Settings().approval_expiry_hours == 72