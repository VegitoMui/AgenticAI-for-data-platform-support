"""
Email notifications.

Two messages, one per outcome of the approval gate:

  fix_applied      The agent was confident and ran the fix on its own. The
                   person is told what went wrong, what ran, and the result.
  decision_needed  The agent paused. The email explains why and links to the
                   approvals App. Decisions are made in the App, never by
                   replying to the email.

Building a message is pure (easy to test). Sending failures are logged and
swallowed: a notification must never block remediation or approvals.
"""

from __future__ import annotations

import logging
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage

from agentic_ai.config import Settings

log = logging.getLogger(__name__)

SUBJECT_PREFIX = "[agentic-ai]"


@dataclass
class Email:
    subject: str
    body: str


class EmailNotifier:
    def __init__(self, host: str, port: int, user: str, password: str,
                 sender: str, recipient: str):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.sender = sender
        self.recipient = recipient

    @classmethod
    def from_settings(cls, settings: Settings) -> EmailNotifier | None:
        """Returns None (email disabled) rather than raising if SMTP secrets
        are missing, so the reconciler keeps working without email."""
        try:
            smtp = settings.smtp
        except Exception as exc:
            log.warning("email disabled, SMTP settings unavailable: %s", str(exc)[:160])
            return None
        required = ("user", "password", "sender", "approver")
        if not all(smtp.get(k) for k in required):
            log.warning("email disabled, SMTP settings incomplete")
            return None
        return cls(
            host=smtp["host"], port=int(smtp["port"]), user=smtp["user"],
            password=smtp["password"], sender=smtp["sender"], recipient=smtp["approver"],
        )

    def send(self, email: Email) -> bool:
        msg = EmailMessage()
        msg["Subject"] = email.subject
        msg["From"] = self.sender
        msg["To"] = self.recipient
        msg.set_content(email.body)
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(self.host, self.port, context=context, timeout=20) as server:
                server.login(self.user, self.password)
                server.send_message(msg)
            log.info("email sent: %s", email.subject)
            return True
        except Exception as exc:
            log.warning("email send failed (%s): %s", email.subject, str(exc)[:200])
            return False


# ------------------------------------------------------------------ messages

def _steps(lines: list[str], actions: list[str]) -> None:
    if not actions:
        lines.append("  No steps were proposed.")
        return
    for i, action in enumerate(actions, start=1):
        lines.append(f"  {i}. {action}")


def fix_applied(incident: dict, d, results: list[dict], issue_url: str = "") -> Email:
    succeeded = sum(1 for r in results if r["execution_status"] == "SUCCESS")
    failed = sum(1 for r in results if r["execution_status"] == "FAILED")
    skipped = len(results) - succeeded - failed
    pipeline = incident["pipeline_name"]

    headline = "Fix applied" if failed == 0 else "Fix applied with failures"
    lines = [
        f"The {d.agent_name} agent fixed a failure in {pipeline} without waiting for approval,",
        "because it was confident the fix was safe.",
        "",
        f"What went wrong: {d.diagnosis}",
        "",
        "What it did:",
    ]
    if results:
        for r in results:
            lines.append(f"  {r['action_index'] + 1}. [{r['execution_status']}] {r['action_text']}")
            if r.get("error_message"):
                lines.append(f"     {r['error_message']}")
    else:
        lines.append("  No steps were proposed.")
    lines += [
        "",
        f"Result: {succeeded} succeeded, {failed} failed, {skipped} skipped.",
        f"Incident: {incident['incident_id']}",
    ]
    if issue_url:
        lines.append(f"Ticket: {issue_url}")
    return Email(subject=f"{SUBJECT_PREFIX} {headline}: {pipeline}", body="\n".join(lines))


def decision_needed(incident: dict, d, request_id: str, app_url: str, expiry_hours: int,
                    confidence_threshold: float = 0.6, issue_url: str = "") -> Email:
    pipeline = incident["pipeline_name"]

    reasons = []
    if d.requires_human:
        reasons.append("the agent asked for a human check")
    if d.fix_complexity == "HIGH":
        reasons.append("the fix is high complexity")
    if d.confidence < confidence_threshold:
        reasons.append(f"the agent is only {round(d.confidence * 100)}% confident")

    lines = [
        f"The {d.agent_name} agent diagnosed a failure in {pipeline} and is waiting",
        "for your decision before it changes anything.",
        "",
        f"Why it paused: {', '.join(reasons) or 'approval required'}.",
        "",
        f"What went wrong: {d.diagnosis}",
        f"Proposed fix: {d.fix_suggestion}",
        "",
        "Proposed steps:",
    ]
    _steps(lines, d.actions)
    lines.append("")
    if app_url:
        lines.append(f"Review and decide: {app_url}")
    else:
        lines.append("Open the approvals App to review and decide.")
    lines += [
        f"If nobody decides within {expiry_hours} hours, the request expires and nothing runs.",
        "",
        f"Request: {request_id}",
        f"Incident: {incident['incident_id']}",
    ]
    if issue_url:
        lines.append(f"Ticket: {issue_url}")
    return Email(subject=f"{SUBJECT_PREFIX} Decision needed: {pipeline}", body="\n".join(lines))