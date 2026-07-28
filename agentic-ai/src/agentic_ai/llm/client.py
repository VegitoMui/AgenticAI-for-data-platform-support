"""Multi-provider LLM client with automatic failover."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from openai import OpenAI

from agentic_ai.config import LLMProvider, Settings

log = logging.getLogger(__name__)


@dataclass
class LLMResult:
    status: str
    content: str
    latency_sec: float
    tokens_used: int | str
    model: str
    provider: str
    parsed: dict | None = None
    parse_error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "success"


def _strip_fences(text: str) -> str:
    if "```json" in text:
        return text.split("```json")[1].split("```")[0].strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            return parts[1].strip()
    return text.strip()


class LLMClient:
    def __init__(self, providers: list[LLMProvider]):
        self._providers = [
            (p, OpenAI(api_key=p.api_key, base_url=p.base_url)) for p in providers
        ]

    @classmethod
    def from_settings(cls, settings: Settings) -> LLMClient:
        return cls(settings.llm_providers)

    def chat(
        self,
        user_message: str,
        system_message: str = "You are a helpful assistant.",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResult:
        last_error = "no providers configured"

        for provider, client in self._providers:
            start = time.time()
            try:
                response = client.chat.completions.create(
                    model=provider.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as exc:
                last_error = str(exc)
                log.warning("provider %s failed (%s), falling back", provider.name, _code(last_error))
                continue

            return LLMResult(
                status="success",
                content=response.choices[0].message.content or "",
                latency_sec=round(time.time() - start, 2),
                tokens_used=response.usage.total_tokens if response.usage else "N/A",
                model=provider.model,
                provider=provider.name,
            )

        return LLMResult(
            status="error",
            content=last_error,
            latency_sec=0.0,
            tokens_used=0,
            model="none",
            provider="all_failed",
        )

    def chat_json(
        self,
        user_message: str,
        system_message: str = "Return ONLY valid JSON, no markdown.",
        max_tokens: int = 1024,
    ) -> LLMResult:
        result = self.chat(user_message, system_message, max_tokens, temperature=0.3)
        if not result.ok:
            return result
        try:
            result.parsed = json.loads(_strip_fences(result.content))
        except json.JSONDecodeError as exc:
            result.parsed = None
            result.parse_error = str(exc)
        return result

    def health_check(self) -> list[dict]:
        out = []
        for provider, client in self._providers:
            start = time.time()
            try:
                client.chat.completions.create(
                    model=provider.model,
                    messages=[{"role": "user", "content": "Respond with only the word OK."}],
                    max_tokens=10,
                    temperature=0,
                )
                out.append({"name": provider.name, "status": "pass",
                            "latency_sec": round(time.time() - start, 2)})
            except Exception as exc:
                out.append({"name": provider.name, "status": "fail", "error": str(exc)[:150]})
        return out


def _code(error: str) -> str:
    if "429" in error:
        return "RATE_LIMITED"
    if "401" in error:
        return "AUTH_FAILED"
    if "404" in error:
        return "MODEL_NOT_FOUND"
    return "ERROR"
