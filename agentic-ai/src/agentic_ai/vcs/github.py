"""
GitHub integration.

Three responsibilities:
  1. create_issue   -- one issue per incident, linked to the approval request
  2. close_issue    -- called by the reconciler after execution or rejection
  3. commits_since  -- commit history used to enrich diagnosis with likely cause

Authenticates with a fine-grained personal access token or a GitHub App
installation token. Required permissions: Issues (read/write), Contents (read).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import requests

from agentic_ai.config import Settings

log = logging.getLogger(__name__)

API_ROOT = "https://api.github.com"


@dataclass
class Commit:
    sha: str
    author: str
    date: str
    message: str
    url: str

    def summary(self) -> str:
        first_line = self.message.splitlines()[0] if self.message else ""
        return f"{self.sha[:8]} {self.date} {self.author}: {first_line[:120]}"


class GitHubClient:
    def __init__(self, owner: str, repo: str, token: str):
        self.owner = owner
        self.repo = repo
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })

    @classmethod
    def from_settings(cls, settings: Settings) -> GitHubClient:
        cfg = settings.github
        return cls(cfg["owner"], cfg["repo"], cfg["token"])

    @property
    def _base(self) -> str:
        return f"{API_ROOT}/repos/{self.owner}/{self.repo}"

    # ----------------------------------------------------------------- issues

    def create_issue(
        self,
        title: str,
        body: str,
        incident_id: str,
        severity: int,
        approval_url: str = "",
    ) -> dict | None:
        """Create an issue. Returns the created issue, or None on failure.

        Failure here must never block remediation, so errors are logged
        and swallowed.
        """
        if approval_url:
            body = f"{body}\n\n---\n\n[Review and approve this remediation]({approval_url})"

        payload = {
            "title": title[:255],
            "body": body,
            "labels": ["agentic-ai", _severity_label(severity), incident_id],
        }
        try:
            resp = self._session.post(f"{self._base}/issues", json=payload, timeout=30)
            resp.raise_for_status()
            issue = resp.json()
            log.info("created issue #%s for %s", issue.get("number"), incident_id)
            return issue
        except Exception as exc:
            log.error("issue creation failed for %s: %s", incident_id, exc)
            return None

    def comment(self, issue_number: int, body: str) -> bool:
        try:
            resp = self._session.post(
                f"{self._base}/issues/{issue_number}/comments",
                json={"body": body},
                timeout=30,
            )
            resp.raise_for_status()
            return True
        except Exception as exc:
            log.error("comment on issue #%s failed: %s", issue_number, exc)
            return False

    def close_issue(self, issue_number: int, comment: str, reason: str = "completed") -> bool:
        """Close an issue. reason is 'completed' or 'not_planned'."""
        if comment:
            self.comment(issue_number, comment)
        try:
            resp = self._session.patch(
                f"{self._base}/issues/{issue_number}",
                json={"state": "closed", "state_reason": reason},
                timeout=30,
            )
            resp.raise_for_status()
            return True
        except Exception as exc:
            log.error("closing issue #%s failed: %s", issue_number, exc)
            return False

    # ---------------------------------------------------------------- commits

    def commits_since(
        self,
        since: datetime,
        path: str | None = None,
        branch: str | None = None,
        per_page: int = 20,
    ) -> list[Commit]:
        """Commits since a timestamp, optionally scoped to a path or branch.

        The watcher passes the last successful run time of the failed job so
        the agent can correlate a failure with what changed.
        """
        params: dict[str, object] = {
            "since": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "per_page": per_page,
        }
        if path:
            params["path"] = path
        if branch:
            params["sha"] = branch

        try:
            resp = self._session.get(f"{self._base}/commits", params=params, timeout=30)
            resp.raise_for_status()
            out = []
            for c in resp.json():
                commit = c.get("commit", {})
                author = commit.get("author", {}) or {}
                out.append(Commit(
                    sha=c.get("sha", ""),
                    author=author.get("name", ""),
                    date=author.get("date", ""),
                    message=commit.get("message", ""),
                    url=c.get("html_url", ""),
                ))
            return out
        except Exception as exc:
            log.error("commit lookup failed: %s", exc)
            return []


def _severity_label(severity: int) -> str:
    if severity >= 9:
        return "severity:critical"
    if severity >= 7:
        return "severity:high"
    if severity >= 4:
        return "severity:medium"
    return "severity:low"
