from agentic_ai.config import Settings


def test_table_normalises_regardless_of_prefix():
    s = Settings(catalog="databricks_ws", schema="agentic_ai")
    expected = "databricks_ws.agentic_ai.remediation_log"
    assert s.table("remediation_log") == expected
    assert s.table("agentic_ai.remediation_log") == expected
    assert s.table("databricks_ws.agentic_ai.remediation_log") == expected


def test_fq_schema():
    assert Settings(catalog="c", schema="s").fq_schema == "c.s"
