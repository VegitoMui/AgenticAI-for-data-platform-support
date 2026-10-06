"""
Analytics agent: querying and access.

Sub-agents
  QUERY_ERRORS       the query itself is wrong: unresolved columns, syntax
                     errors, unknown functions, datatype mismatches. The default.
  QUERY_PERFORMANCE  the query is too slow: timeouts, full scans, spill.
  ACCESS_CONTROL     the identity running the job lacks a privilege.

Tools
  error_analysis    shared with the Processing agent: exception, error
                    class, SQLSTATE, the failing line in the job's own code.
  run_history       shared with the Processing agent: new regression,
                    consistent or flaky failure.
  query_history     statements this job sent in the last 2 days, from
                    system.query.history: failed statements and their errors,
                    the slowest statements, files read vs pruned, spill.
                    Analysis-time errors and classic-cluster runs are often
                    not recorded, so an empty result is reported as such and
                    is not evidence that the queries were healthy.
  column_check      parses the unresolved column name from the error,
                    DESCRIBEs the tables involved and suggests the closest
                    existing column names.
  access_check      the identity the run executed as, the privilege the
                    error says is missing, and SHOW GRANTS on the table,
                    schema and catalog.
  table_layout      reuses the Storage agent's DESCRIBE DETAIL tool, for
                    partitioning and file counts behind slow queries.

All tools are read-only. Permission changes (GRANT / REVOKE) are only ever
proposed, and always require a human (enforced in diagnosis_from_result).
"""

from __future__ import annotations

import difflib
import re

from agentic_ai.agents.base import BaseAgent, SubAgent, Tool, ToolContext
from agentic_ai.agents.processing import _workspace, error_analysis, run_history
from agentic_ai.agents.storage import _rows, quote, table_detail, target_tables

QUERY_HISTORY_DAYS = 2
QUERY_HISTORY_LIMIT = 50
FULL_SCAN_MIN_FILES = 50
SLOW_MIN_MS = 10_000            # only statements at least this slow (or flagged) are listed
SLOWEST_SHOWN = 3
FAILED_SHOWN = 3
TEXT_CHARS = 160

_JOB_ID = re.compile(r"^\d+$")

# "with name `custmer_id` cannot be resolved", "with name `i`.`custmer_id`",
# older "cannot resolve '`custmer_id`' given input columns".
_UNRESOLVED = re.compile(
    r"(?:with name|cannot resolve)\s+'?((?:`[^`]+`\.?)+|[A-Za-z_][\w.]*)'?", re.I
)
_SUGGESTIONS = re.compile(r"Did you mean one of the following\?\s*\[([^\]]*)\]", re.I)
_MISSING_PRIV = re.compile(
    r"does not have\s+(?:the\s+)?(?:privilege\s+)?([A-Z][A-Z_ ]*?)\s+on\s+"
    r"(table|view|schema|catalog|volume|function)\s+['`\"]?([\w.`]+)",
    re.I,
)


# ------------------------------------------------------------------ helpers

def _error_text(tctx: ToolContext) -> str:
    return "\n".join(f"{t.error}\n{t.error_trace}" for t in tctx.run.task_errors)


def _field(obj, *path):
    """Nested lookup that works on dicts and Spark Rows, returning None when absent."""
    for key in path:
        if obj is None:
            return None
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            try:
                obj = obj[key]
            except (KeyError, IndexError, TypeError, ValueError):
                obj = getattr(obj, key, None)
    return obj


def _int(v) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def _one_line(text, limit: int = TEXT_CHARS) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[:limit] + "..."


def _mb(n) -> str:
    return f"{(n or 0) / (1024 * 1024):,.1f} MB"


def unresolved_columns(text: str) -> list[str]:
    """Column names the engine could not resolve (last name part, deduplicated)."""
    out: list[str] = []
    for raw in _UNRESOLVED.findall(text):
        name = raw.replace("`", "").split(".")[-1].strip()
        if name and name not in out:
            out.append(name)
    return out


def engine_suggestions(text: str) -> list[str]:
    m = _SUGGESTIONS.search(text)
    if not m:
        return []
    return [s.strip().strip("`").split("`.`")[-1] for s in m.group(1).split(",") if s.strip()]


def missing_privilege(text: str) -> dict:
    m = _MISSING_PRIV.search(text)
    if not m:
        return {}
    return {"privilege": " ".join(m.group(1).upper().split()), "object_type": m.group(2).upper(),
            "object": m.group(3).replace("`", "").rstrip(".")}


# ------------------------------------------------------------------ tools

def query_history(tctx: ToolContext) -> tuple[str, dict]:
    job_id = str(tctx.run.job_id or "")
    if not _JOB_ID.match(job_id):
        return "No valid job id; query history unavailable.", {}
    rows = _rows(
        tctx.spark,
        "SELECT * FROM system.query.history "
        f"WHERE query_source.job_info.job_id = '{job_id}' "
        f"AND start_time >= current_timestamp() - INTERVAL {QUERY_HISTORY_DAYS} DAYS "
        f"ORDER BY start_time DESC LIMIT {QUERY_HISTORY_LIMIT}",
    )
    if not rows:
        return (
            f"No statements from this job in system.query.history in the last {QUERY_HISTORY_DAYS} "
            "days. Analysis-time errors (unresolved columns, syntax) and classic-cluster runs are "
            "often not recorded, so this is not evidence that the queries were healthy.",
            {"statements": 0},
        )

    run_id = str(tctx.run.run_id or "")
    this_run = [r for r in rows if str(_field(r, "query_source", "job_info", "job_run_id") or "") == run_id]
    failed = [r for r in rows if str(_field(r, "execution_status") or "").upper() == "FAILED"]

    stats = []
    for r in rows:
        read_files, pruned = _int(_field(r, "read_files")), _int(_field(r, "pruned_files"))
        spilled = _int(_field(r, "spilled_local_bytes"))
        stats.append({
            "statement_id": str(_field(r, "statement_id") or ""),
            "status": str(_field(r, "execution_status") or ""),
            "duration_ms": _int(_field(r, "total_duration_ms")),
            "read_files": read_files,
            "pruned_files": pruned,
            "read_bytes": _int(_field(r, "read_bytes")),
            "spilled_bytes": spilled,
            "full_scan": read_files >= FULL_SCAN_MIN_FILES and pruned == 0,
            "text": _one_line(_field(r, "statement_text")),
        })

    parts = [f"{len(rows)} statements in the last {QUERY_HISTORY_DAYS} days "
             f"({len(this_run)} from this run), {len(failed)} failed"]
    for r in failed[:FAILED_SHOWN]:
        parts.append(f"FAILED: {_one_line(_field(r, 'error_message'))} | statement: "
                     f"{_one_line(_field(r, 'statement_text'), 100)}")
    notable = [s for s in stats if s["duration_ms"] >= SLOW_MIN_MS or s["full_scan"] or s["spilled_bytes"]]
    if not notable:
        parts.append(f"no statement ran {SLOW_MIN_MS // 1000}s or longer, scanned fully or spilled")
    for s in sorted(notable, key=lambda s: -s["duration_ms"])[:SLOWEST_SHOWN]:
        flags = []
        if s["full_scan"]:
            flags.append("FULL SCAN")
        if s["spilled_bytes"]:
            flags.append(f"SPILLED {_mb(s['spilled_bytes'])}")
        parts.append(
            f"slow: {s['duration_ms'] / 1000:.1f}s, {s['read_files']} files read, "
            f"{s['pruned_files']} pruned, {_mb(s['read_bytes'])} read"
            + (f", {', '.join(flags)}" if flags else "") + f" | {_one_line(s['text'], 100)}"
        )
    full_scans = sum(s["full_scan"] for s in stats)
    spills = sum(bool(s["spilled_bytes"]) for s in stats)
    if full_scans:
        parts.append(f"{full_scans} statement(s) read {FULL_SCAN_MIN_FILES}+ files with none pruned")
    if spills:
        parts.append(f"{spills} statement(s) spilled to disk")
    return "; ".join(parts), {"statements": len(rows), "this_run": len(this_run), "failed": len(failed),
                              "full_scans": full_scans, "spills": spills, "stats": stats}


def column_check(tctx: ToolContext) -> tuple[str, dict]:
    text = _error_text(tctx)
    names = unresolved_columns(text)
    if not names:
        return "No unresolved column name found in the error.", {}
    suggested = engine_suggestions(text)

    tables = target_tables(tctx)
    findings, data = [], {"unresolved": names, "engine_suggestions": suggested, "tables": {}}
    for t in tables:
        try:
            cols = []
            for r in _rows(tctx.spark, f"DESCRIBE TABLE {quote(t)}"):
                name = str(r.get("col_name") or "")
                if not name or name.startswith("#"):
                    break
                cols.append(name)
        except Exception as exc:
            findings.append(f"{t}: DESCRIBE failed ({str(exc).splitlines()[0][:120]})")
            continue
        lower = {c.lower(): c for c in cols}
        per_col = {}
        for n in names:
            if n.lower() in lower:
                findings.append(f"{t}: column {n} EXISTS (so the reference or its qualifier is wrong)")
                per_col[n] = {"exists": True, "closest": []}
                continue
            closest = [lower[c] for c in difflib.get_close_matches(n.lower(), list(lower), n=3, cutoff=0.6)]
            hint = f"closest: {', '.join(closest)}" if closest else "no similar column"
            findings.append(f"{t}: column {n} NOT FOUND among {len(cols)} columns, {hint}")
            per_col[n] = {"exists": False, "closest": closest}
        data["tables"][t] = {"columns": len(cols), "matches": per_col}

    head = f"unresolved column(s): {', '.join(names)}"
    if suggested:
        head += f"; engine suggested: {', '.join(suggested[:5])}"
    if not tables:
        findings.append("no tables identified to check against")
    return "; ".join([head] + findings), data


def access_check(tctx: ToolContext) -> tuple[str, dict]:
    text = _error_text(tctx)
    missing = missing_privilege(text)

    identity = ""
    if tctx.run.run_id:
        try:
            run = _workspace(tctx).jobs.get_run(run_id=int(tctx.run.run_id))
            identity = getattr(run, "run_as_user_name", None) or getattr(run, "creator_user_name", None) or ""
        except Exception as exc:
            missing["identity_error"] = str(exc)[:120]

    # Objects to inspect: the one named in the error first, then the tables involved.
    tables = []
    if missing.get("object_type") in ("TABLE", "VIEW") and quote(missing["object"]):
        tables.append(missing["object"])
    tables += [t for t in target_tables(tctx) if t not in tables]
    tables = tables[:2]

    securables: list[tuple[str, str]] = []
    for t in tables:
        cat, sch, _ = t.split(".")
        for kind, name in (("TABLE", t), ("SCHEMA", f"{cat}.{sch}"), ("CATALOG", cat)):
            if (kind, name) not in securables:
                securables.append((kind, name))
    if missing.get("object_type") in ("SCHEMA", "CATALOG"):
        obj = missing["object"]
        parts = obj.split(".")
        ok = len(parts) == (2 if missing["object_type"] == "SCHEMA" else 1)
        if ok and (missing["object_type"], obj) not in securables:
            securables.insert(0, (missing["object_type"], obj))

    findings = [f"run executed as {identity or 'unknown identity'}"]
    if missing.get("privilege"):
        findings.append(f"error says missing {missing['privilege']} on "
                        f"{missing['object_type'].lower()} {missing['object']}")
    data: dict = {"identity": identity, "missing": missing, "grants": {}}

    for kind, name in securables:
        q = ".".join(f"`{p}`" for p in name.split("."))
        if not all(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", p) for p in name.split(".")):
            continue
        try:
            rows = _rows(tctx.spark, f"SHOW GRANTS ON {kind} {q}")
        except Exception as exc:
            findings.append(f"{kind.lower()} {name}: grants not visible ({str(exc).splitlines()[0][:100]})")
            continue
        held = sorted({
            str(r.get("ActionType") or r.get("action_type") or "").upper()
            for r in rows
            if identity and str(r.get("Principal") or r.get("principal") or "").lower() == identity.lower()
        } - {""})
        principals = {str(r.get("Principal") or r.get("principal") or "") for r in rows}
        findings.append(
            f"{kind.lower()} {name}: identity holds directly: {', '.join(held) or 'nothing'} "
            f"({len(rows)} grants across {len(principals)} principal(s))"
        )
        data["grants"][f"{kind} {name}"] = {"held_directly": held, "grant_rows": len(rows)}

    findings.append("grants held through groups are not expanded, so 'nothing' means not granted "
                    "directly, not necessarily missing")
    return "; ".join(findings), data


# ------------------------------------------------------------------ agent

class AnalyticsAgent(BaseAgent):
    name = "analytics"
    description = "Query errors, slow queries, and access control."

    sub_agents = [
        SubAgent(
            "QUERY_ERRORS",
            "Is the query itself wrong? Name the exact column, function or syntax at fault and "
            "the closest correct name from the evidence. This is a fix to the job's code, not a "
            "data-platform change.",
            ("unresolved_column", "cannot be resolved", "cannot resolve", "parse_syntax_error",
             "syntax error", "unresolved_routine", "datatype_mismatch", "ambiguous_reference",
             "cast_invalid_input", "42703", "42601"),
            ("error_analysis", "column_check", "query_history", "run_history"),
        ),
        SubAgent(
            "QUERY_PERFORMANCE",
            "Is the query too slow? Use query history for full scans, pruning and spill, and "
            "table layout for partitioning and clustering.",
            ("timeout", "timed out", "full scan", "spill", "slow", "query_timeout",
             "statement_timeout", "exceeded"),
            ("query_history", "table_layout", "run_history", "error_analysis"),
        ),
        SubAgent(
            "ACCESS_CONTROL",
            "Is a privilege missing? Name the identity, the privilege and the object. Any GRANT is "
            "a proposal for a person to approve.",
            ("insufficient_permissions", "permission_denied", "insufficient privileges",
             "does not have", "access denied", "forbidden", "privilege", "403"),
            ("error_analysis", "access_check", "run_history"),
        ),
    ]

    tools = {
        "error_analysis": Tool("error_analysis", "Exception, error class, failing line, origin",
                               error_analysis),
        "run_history": Tool("run_history", "Recent runs: first failure, consistent or flaky",
                            run_history),
        "query_history": Tool("query_history", "system.query.history: failures, slow, scans, spill",
                              query_history),
        "column_check": Tool("column_check", "Unresolved column vs real columns, closest names",
                             column_check),
        "access_check": Tool("access_check", "Run-as identity, missing privilege, SHOW GRANTS",
                             access_check),
        "table_layout": Tool("table_layout", "DESCRIBE DETAIL of the tables involved", table_detail),
    }