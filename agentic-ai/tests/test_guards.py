"""Which proposed actions must wait for a person, and which approved ones can run."""

import pytest

from agentic_ai.agents.base import needs_person
from agentic_ai.remediation.executor import classify_action

T = "databricks_ws.agentic_probe.orders_daily"


@pytest.mark.parametrize("action", [
    f"ALTER TABLE {T} ADD COLUMNS (discount DOUBLE)",
    f"ALTER TABLE {T} ADD COLUMN discount DOUBLE",
    f"ALTER TABLE {T} ALTER COLUMN status SET DEFAULT 'new'",
    f"ALTER TABLE {T} RENAME COLUMN amount TO total",
    f"ALTER TABLE {T} RENAME TO databricks_ws.agentic_probe.orders_old",
    f"ALTER TABLE {T} SET TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported')",
    f"ALTER TABLE {T} SET TBLPROPERTIES ('delta.minWriterVersion' = '7')",
    "DROP TABLE databricks_ws.agentic_probe.customers_bkp",
])
def test_schema_protocol_and_drop_changes_need_a_person(action):
    assert needs_person(action)


@pytest.mark.parametrize("action", [
    f"OPTIMIZE {T}",
    f"ANALYZE TABLE {T} COMPUTE STATISTICS",
    f"VACUUM {T} RETAIN 168 HOURS",
    f"ALTER TABLE {T} SET TBLPROPERTIES ('delta.autoOptimize.autoCompact' = 'true')",
])
def test_safe_maintenance_still_runs_unattended(action):
    assert not needs_person(action)


def test_approved_drop_table_is_executable():
    assert classify_action("DROP TABLE databricks_ws.agentic_probe.customers_bkp") == "SQL_DDL"