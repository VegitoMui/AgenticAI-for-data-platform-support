"""
Central configuration.

Nothing in this package reads a literal credential. Everything resolves through
Settings, which pulls from the Databricks secret scope when running on a cluster
and falls back to environment variables for local test runs.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from functools import cached_property

from agentic_ai.secrets import SecretResolver


@dataclass
class LLMProvider:
    name: str
    api_key: str
    base_url: str
    model: str


@dataclass
class Settings:
    catalog: str = "databricks_ws"
    schema: str = "agentic_ai"
    secret_scope: str = "agentic-ai"

    # Telemetry
    watcher_lookback_hours: int = 24
    watcher_batch_limit: int = 25
    failed_states: tuple[str, ...] = ("FAILED", "TIMED_OUT", "ERROR")

    # Approval
    approval_expiry_hours: int = 24
    auto_approve_enabled: bool = False

    _resolver: SecretResolver = field(default_factory=SecretResolver, repr=False)

    # ---------------------------------------------------------------- naming

    @property
    def fq_schema(self) -> str:
        return f"{self.catalog}.{self.schema}"

    def table(self, name: str) -> str:
        """Fully-qualified table name. Use this everywhere instead of f-strings.

        Guards against the double-schema-prefix bug from the notebook version:
        table('agentic_ai.remediation_log') and table('remediation_log') both
        resolve to catalog.schema.remediation_log.
        """
        leaf = name.split(".")[-1]
        return f"{self.catalog}.{self.schema}.{leaf}"

    # ------------------------------------------------------------ credentials

    @cached_property
    def llm_providers(self) -> list[LLMProvider]:
        scope = self.secret_scope
        get = self._resolver.get
        return [
            LLMProvider(
                name="OpenRouter (Primary)",
                api_key=get(scope, "openrouter-api-key"),
                base_url="https://openrouter.ai/api/v1",
                model="nvidia/nemotron-3-super-120b-a12b:free",
            ),
            LLMProvider(
                name="Groq Fallback-1",
                api_key=get(scope, "groq-api-key-1"),
                base_url="https://api.groq.com/openai/v1",
                model="llama-3.3-70b-versatile",
            ),
            LLMProvider(
                name="Groq Fallback-2",
                api_key=get(scope, "groq-api-key-2"),
                base_url="https://api.groq.com/openai/v1",
                model="llama-3.3-70b-versatile",
            ),
        ]

    @cached_property
    def smtp(self) -> dict[str, str]:
        get = self._resolver.get
        return {
            "host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
            "port": os.getenv("SMTP_PORT", "465"),
            "user": get(self.secret_scope, "smtp-user"),
            "password": get(self.secret_scope, "smtp-password"),
            "sender": get(self.secret_scope, "smtp-sender"),
            "approver": get(self.secret_scope, "approver-email"),
        }

    @cached_property
    def github(self) -> dict[str, str]:
        get = self._resolver.get
        return {
            "owner": get(self.secret_scope, "github-owner"),
            "repo": get(self.secret_scope, "github-repo"),
            "token": get(self.secret_scope, "github-token"),
        }

    @cached_property
    def app_base_url(self) -> str:
        """Base URL of the approvals App. Used to build links in emails."""
        return self._resolver.get(self.secret_scope, "app-base-url", required=False) or ""


def settings_from_args(argv: list[str] | None = None) -> Settings:
    """Parse the standard job arguments into Settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default=os.getenv("AGENTIC_CATALOG", "databricks_ws"))
    parser.add_argument("--schema", default=os.getenv("AGENTIC_SCHEMA", "agentic_ai"))
    parser.add_argument("--secret-scope", default=os.getenv("AGENTIC_SECRET_SCOPE", "agentic-ai"))
    known, _ = parser.parse_known_args(argv)
    return Settings(
        catalog=known.catalog,
        schema=known.schema,
        secret_scope=known.secret_scope,
    )
