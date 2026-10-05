"""
Ingestion agent: getting data in.

Sub-agents
  CONNECTIVITY          the job cannot reach a source: connection refused or
                        reset, timeouts, DNS, SSL, JDBC / Kafka. The default.
  RATE_LIMIT_RECOVERY   the source is pushing back: HTTP 429, throttling,
                        quota errors. The fix is usually retry and backoff.
  LANDING_FILES         file-based ingestion: Auto Loader, missing paths,
                        no new files arriving, source schema changes.

Tools
  error_analysis      shared with the Processing agent: exception, error
                      class, failing line, retries.
  run_history         shared with the Processing agent: transient, flaky or
                      persistent failure.
  source_endpoints    pulls the sources out of the error text: URLs, JDBC
                      URLs, hosts and ports, storage paths, HTTP status codes,
                      Retry-After hints. No external calls.
  reachability        a TCP connect test (no data sent) from Databricks
                      compute to each host found, to tell "down right now"
                      from "was down at the time".
  landing_path_check  LISTs each storage path found: does it exist, how many
                      files, and how fresh is the newest one.

All tools are read-only.
"""

from __future__ import annotations

import re
import socket
import time

from agentic_ai.agents.base import BaseAgent, SubAgent, Tool, ToolContext
from agentic_ai.agents.processing import error_analysis, run_history

MAX_HOSTS = 3
MAX_PATHS = 3
CONNECT_TIMEOUT_SEC = 3
LIST_LIMIT = 1000
DEFAULT_PORTS = {"https": 443, "http": 80, "kafka": 9092, "sqlserver": 1433, "postgresql": 5432,
                 "mysql": 3306, "oracle": 1521, "snowflake": 443}

_URL = re.compile(r"\b(?:https?|jdbc:[a-z0-9]+|kafka)://[^\s'\"<>),;]+", re.I)
_HOST_HINT = re.compile(
    r"(?:host|server|endpoint|connect(?:ing|ion)? to|resolve)\s*[=:]?\s*['\"]?"
    r"([a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,})(?::(\d{2,5}))?", re.I,
)
_STATUS = re.compile(
    r"\b(?:status(?: code)?|http(?:/\d(?:\.\d)?)?|response code)\s*[:=]?\s*([1-5]\d\d)\b", re.I,
)
_RETRY_AFTER = re.compile(r"retry[- ]after\D{0,5}(\d+)", re.I)
_PATH = re.compile(r"(?:abfss?|wasbs?|s3a?|gs|dbfs)://[^\s'\"<>),;`]+|/Volumes/[^\s'\"<>),;`]+")
_SAFE_PATH = re.compile(r"^[\w./:@%+=~-]+$")
_FILE_SUFFIXES = (".py", ".java", ".scala", ".json", ".txt", ".csv", ".parquet", ".log", ".ipynb")


def _error_text(tctx: ToolContext) -> str:
    return "\n".join(f"{t.error}\n{t.error_trace}" for t in tctx.run.task_errors)


def find_endpoints(text: str) -> dict:
    """Pure parsing; shared by the tools below and by the tests."""
    hosts: dict[str, int] = {}
    urls = sorted(set(_URL.findall(text)))
    for url in urls:
        m = re.match(r"(?:jdbc:)?([a-z0-9]+)://(?:[^@/]*@)?([^:/?;]+)(?::(\d+))?", url, re.I)
        if m:
            scheme, host, port = m.group(1).lower(), m.group(2).lower(), m.group(3)
            hosts.setdefault(host, int(port) if port else DEFAULT_PORTS.get(scheme, 443))
    for host, port in _HOST_HINT.findall(text):
        host = host.lower()
        if host.endswith(_FILE_SUFFIXES):
            continue
        hosts.setdefault(host, int(port) if port else 443)

    return {
        "urls": urls,
        "hosts": hosts,
        "paths": sorted({p.rstrip("./") for p in _PATH.findall(text)}),
        "status_codes": sorted(set(_STATUS.findall(text))),
        "retry_after_sec": sorted({int(x) for x in _RETRY_AFTER.findall(text)}),
    }


# ------------------------------------------------------------------ tools

def source_endpoints(tctx: ToolContext) -> tuple[str, dict]:
    found = find_endpoints(_error_text(tctx))
    parts = []
    if found["hosts"]:
        parts.append("hosts: " + ", ".join(f"{h}:{p}" for h, p in found["hosts"].items()))
    if found["paths"]:
        parts.append("storage paths: " + ", ".join(found["paths"]))
    if found["status_codes"]:
        parts.append("HTTP status codes: " + ", ".join(found["status_codes"]))
    if found["retry_after_sec"]:
        parts.append("source asked to retry after " + ", ".join(f"{s}s" for s in found["retry_after_sec"]))
    return ("; ".join(parts) or "No source hosts, paths or status codes found in the error."), found


def reachability(tctx: ToolContext) -> tuple[str, dict]:
    hosts = list(find_endpoints(_error_text(tctx))["hosts"].items())[:MAX_HOSTS]
    if not hosts:
        return "No source hosts found in the error to test.", {}
    findings, data = [], {}
    for host, port in hosts:
        start = time.time()
        try:
            with socket.create_connection((host, port), timeout=CONNECT_TIMEOUT_SEC):
                ms = (time.time() - start) * 1000
            findings.append(f"{host}:{port} reachable now ({ms:.0f} ms)")
            data[host] = {"port": port, "reachable": True, "ms": round(ms)}
        except OSError as exc:
            reason = type(exc).__name__ + (f": {exc}" if str(exc) else "")
            findings.append(f"{host}:{port} NOT reachable now ({reason[:120]})")
            data[host] = {"port": port, "reachable": False, "reason": reason[:200]}
    return "; ".join(findings) + " (tested from Databricks compute)", data


def landing_path_check(tctx: ToolContext) -> tuple[str, dict]:
    paths = [p for p in find_endpoints(_error_text(tctx))["paths"] if _SAFE_PATH.match(p)][:MAX_PATHS]
    if not paths:
        return "No storage paths found in the error to check.", {}
    findings, data = [], {}
    for path in paths:
        try:
            rows = [r.asDict() if hasattr(r, "asDict") else dict(r)
                    for r in tctx.spark.sql(f"LIST '{path}' LIMIT {LIST_LIMIT}").collect()]
        except Exception as exc:
            reason = str(exc).splitlines()[0][:160]
            findings.append(f"{path}: NOT LISTABLE ({reason})")
            data[path] = {"exists": False, "reason": reason}
            continue
        files = [r for r in rows if not str(r.get("name", "")).endswith("/")]
        newest = max((r.get("modification_time") or 0 for r in files), default=0)
        age_h = (time.time() * 1000 - newest) / 3_600_000 if newest else None
        capped = " (listing capped)" if len(rows) >= LIST_LIMIT else ""
        if not files:
            note = "exists but contains NO FILES"
        elif age_h is not None and age_h > 24:
            note = f"{len(files)} files{capped}, newest {age_h / 24:.1f} days old -- NO NEW DATA ARRIVING"
        elif age_h is not None:
            note = f"{len(files)} files{capped}, newest {age_h:.1f} h old"
        else:
            note = f"{len(files)} files{capped}"
        findings.append(f"{path}: {note}")
        data[path] = {"exists": True, "files": len(files), "newest_age_hours": age_h}
    return "; ".join(findings), data


# ------------------------------------------------------------------ agent

class IngestionAgent(BaseAgent):
    name = "ingestion"
    description = "Source connectivity, rate limits and file-based landing data."

    sub_agents = [
        SubAgent(
            "CONNECTIVITY",
            "Can the job reach its source? Tell a source that is down now from one that was "
            "briefly unavailable, and network or credential problems from source outages.",
            ("connection refused", "connection reset", "timed out", "timeout", "unreachable",
             "unknownhost", "name resolution", "name or service not known", "urlopen error",
             "ssl", "certificate", "jdbc", "kafka", "socket"),
            ("error_analysis", "source_endpoints", "reachability", "run_history"),
        ),
        SubAgent(
            "RATE_LIMIT_RECOVERY",
            "Is the source throttling the job? Recommend retry with backoff and request pacing, "
            "using any Retry-After hint, rather than infrastructure changes.",
            ("429", "rate limit", "too many requests", "throttl", "quota", "retry-after"),
            ("error_analysis", "source_endpoints", "run_history"),
        ),
        SubAgent(
            "LANDING_FILES",
            "Is file-based input missing, stale or changed? Look at whether the landing path "
            "exists, whether new files are arriving, and source schema changes.",
            ("cloudfiles", "autoloader", "auto loader", "path does not exist",
             "filenotfoundexception", "no such file", "schema", "/volumes/", "abfss://"),
            ("error_analysis", "source_endpoints", "landing_path_check", "run_history"),
        ),
    ]

    tools = {
        "error_analysis": Tool("error_analysis", "Exception, error class, failing line", error_analysis),
        "run_history": Tool("run_history", "Transient, flaky or persistent", run_history),
        "source_endpoints": Tool("source_endpoints", "Hosts, paths, status codes from the error",
                                 source_endpoints),
        "reachability": Tool("reachability", "TCP connect test to each source host", reachability),
        "landing_path_check": Tool("landing_path_check", "LIST landing paths: exists, freshness",
                                   landing_path_check),
    }