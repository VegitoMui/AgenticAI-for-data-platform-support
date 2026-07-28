"""
Wheel entry points referenced by resources/jobs.yml.

Each is a thin shell: parse args into Settings, configure logging, delegate.
Phase 1 fills in watcher, Phase 2 fills in reconciler.
"""

from __future__ import annotations

import logging
import sys


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        stream=sys.stdout,
    )


def watcher_main() -> None:
    _setup_logging()
    from agentic_ai.config import settings_from_args
    from agentic_ai.telemetry.watcher import run_once

    settings = settings_from_args()
    count = run_once(settings)
    logging.getLogger("watcher").info("raised %s incident(s)", count)


def reconciler_main() -> None:
    _setup_logging()
    from agentic_ai.config import settings_from_args
    from agentic_ai.remediation.reconciler import run_once

    settings = settings_from_args()
    stats = run_once(settings)
    logging.getLogger("reconciler").info("reconciled: %s", stats)


def migrate_main() -> None:
    _setup_logging()
    from agentic_ai.config import settings_from_args
    from agentic_ai.migrations import apply_all

    settings = settings_from_args()
    apply_all(settings)
    logging.getLogger("migrate").info("schema up to date in %s", settings.fq_schema)
