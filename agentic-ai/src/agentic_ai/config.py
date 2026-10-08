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
from urllib.parse import urlparse

from agentic_ai.secrets import SecretResolver

# Fallback models. Each can be overridden without a code change by setting the
# optional secret named alongside it, e.g. when a provider retires a model.
# Groq retired llama-3.3-70b-versatile on 2026-08-16.
DEFAULT_MODELS = {
    "model-groq-1": "openai/gpt-oss-120b",
    "model-groq-2": "qwen/qwen3.6-27b",
    "model-openrouter": "nvidia/nemotron-3-super-120b-a12b:free",
}


@dataclass
class LLMProvider:
    name: str
    api_key: str
    base_url: str
    model: str
    # Reasoning models (recent OpenAI models on Azure AI Foundry) take
    # max_completion_tokens instead of max_tokens and reject a custom temperature.
    reasoning: bool = False


class NoLLMConfigured(RuntimeError):
    pass


def foundry_base_url(endpoint: str) -> str:
    """The OpenAI-compatible v1 base URL for an Azure AI Foundry resource.

    Accepts whatever the Foundry portal shows: the resource endpoint
    (https://x.openai.azure.com or https://x.services.ai.azure.com), the
    .../openai/v1 URL, or a full deployment "target URI" with a path and
    api-version. Only the scheme and host are kept.
    """
    parsed = urlparse(endpoint.strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"not a valid Foundry endpoint URL: {endpoint!r}")
    return f"{parsed.scheme}://{parsed.netloc}/openai/v1/"


@dataclass
class Settings:
    catalog: str = "databricks_ws"
    schema: str = "agentic_ai"
    secret_scope: str = "agentic-ai"

    # Telemetry
    watcher_lookback_hours: int = 24
    watcher_batch_limit: int = 25
    # The two detection sources spell states differently:
    #   system.lakeflow.job_run_timeline -> FAILED, ERROR, CANCELLED
    #   Jobs API (SDK RunResultState)    -> FAILED, TIMED_OUT, CANCELED
    failed_states: tuple[str, ...] = ("FAILED", "CANCELED", "CANCELLED", "ERROR", "TIMED_OUT")
    detection_source: str = "auto"  # system_tables | jobs_api | auto

    # Approval
    approval_expiry_hours: int = 72
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

    def _model(self, key: str) -> str:
        override = self._resolver.get(self.secret_scope, key, required=False)
        return override or DEFAULT_MODELS[key]

    @cached_property
    def llm_providers(self) -> list[LLMProvider]:
        """Tried in order; the client fails over to the next on any error or
        empty reply.

        Azure AI Foundry is primary when its three secrets exist
        (foundry-endpoint, foundry-api-key, foundry-deployment). Groq and
        OpenRouter follow as fallbacks, each only if its key is still in the
        scope, so removing a key removes that provider.
        """
        scope = self.secret_scope

        def opt(key: str) -> str:
            return self._resolver.get(scope, key, required=False)

        providers: list[LLMProvider] = []
        endpoint, key, deployment = opt("foundry-endpoint"), opt("foundry-api-key"), opt("foundry-deployment")
        if endpoint and key and deployment:
            providers.append(LLMProvider(
                name="Azure AI Foundry",
                api_key=key,
                base_url=foundry_base_url(endpoint),
                model=deployment,
                reasoning=opt("foundry-reasoning").lower() not in ("false", "0", "no"),
            ))
        for name, key_name, base_url, model_key in (
            ("Groq Primary", "groq-api-key-1", "https://api.groq.com/openai/v1", "model-groq-1"),
            ("Groq Fallback", "groq-api-key-2", "https://api.groq.com/openai/v1", "model-groq-2"),
            ("OpenRouter Fallback", "openrouter-api-key", "https://openrouter.ai/api/v1", "model-openrouter"),
        ):
            api_key = opt(key_name)
            if api_key:
                providers.append(LLMProvider(name, api_key, base_url, self._model(model_key)))

        if not providers:
            raise NoLLMConfigured(
                f"No LLM provider configured in secret scope '{scope}'. Set foundry-endpoint, "
                "foundry-api-key and foundry-deployment (or a Groq / OpenRouter key)."
            )
        return providers

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
    parser.add_argument(
        "--detection-source",
        default=os.getenv("AGENTIC_DETECTION_SOURCE", "auto"),
        choices=["system_tables", "jobs_api", "auto"],
    )
    known, _ = parser.parse_known_args(argv)
    return Settings(
        catalog=known.catalog,
        schema=known.schema,
        secret_scope=known.secret_scope,
        detection_source=known.detection_source,
    )