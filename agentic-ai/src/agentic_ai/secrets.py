"""
Secret resolution.

Order of resolution for key `k` in scope `s`:
  1. dbutils.secrets.get(s, k)          -- when running on Databricks
  2. environment variable AGENTIC_<K>   -- k uppercased, dashes to underscores
  3. raise, unless required=False
"""

from __future__ import annotations

import os


def _env_name(key: str) -> str:
    return "AGENTIC_" + key.upper().replace("-", "_")


class MissingSecret(RuntimeError):
    pass


class SecretResolver:
    def __init__(self) -> None:
        self._dbutils = self._find_dbutils()
        self._cache: dict[tuple[str, str], str] = {}

    @staticmethod
    def _find_dbutils():
        try:
            from pyspark.dbutils import DBUtils  # type: ignore
            from pyspark.sql import SparkSession

            return DBUtils(SparkSession.builder.getOrCreate())
        except Exception:
            return globals().get("dbutils")

    def get(self, scope: str, key: str, required: bool = True) -> str:
        cache_key = (scope, key)
        if cache_key in self._cache:
            return self._cache[cache_key]

        value = None
        if self._dbutils is not None:
            try:
                value = self._dbutils.secrets.get(scope=scope, key=key)
            except Exception:
                value = None

        if not value:
            value = os.getenv(_env_name(key))

        if not value:
            if required:
                raise MissingSecret(
                    f"Secret '{key}' not found in scope '{scope}' or env var {_env_name(key)}. "
                    f"Create it with: databricks secrets put-secret {scope} {key}"
                )
            return ""

        self._cache[cache_key] = value
        return value
