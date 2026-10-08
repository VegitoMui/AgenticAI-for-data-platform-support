from agentic_ai.config import Settings


def test_table_normalises_regardless_of_prefix():
    s = Settings(catalog="databricks_ws", schema="agentic_ai")
    expected = "databricks_ws.agentic_ai.remediation_log"
    assert s.table("remediation_log") == expected
    assert s.table("agentic_ai.remediation_log") == expected
    assert s.table("databricks_ws.agentic_ai.remediation_log") == expected


def test_fq_schema():
    assert Settings(catalog="c", schema="s").fq_schema == "c.s"


# The two detection sources spell some states differently:
#   system.lakeflow.job_run_timeline -> SUCCEEDED, FAILED, ERROR, CANCELLED
#   Jobs API (SDK RunResultState)    -> SUCCESS, FAILED, TIMED_OUT, CANCELED
# failed_states is shared by both, so it must be a subset of the union.
VALID_RESULT_STATES = {
    "SUCCEEDED", "SUCCESS", "FAILED", "ERROR", "TIMED_OUT",
    "CANCELLED", "CANCELED",
    "EXCLUDED", "MAXIMUM_CONCURRENT_RUNS_REACHED",
}


def test_failed_states_are_valid_result_states():
    assert set(Settings().failed_states) <= VALID_RESULT_STATES


def test_failed_states_cover_both_sources():
    states = set(Settings().failed_states)
    # system tables vocabulary
    assert {"FAILED", "ERROR", "CANCELLED"} <= states
    # Jobs API vocabulary
    assert {"FAILED", "TIMED_OUT", "CANCELED"} <= states


def test_detection_source_defaults_to_auto():
    assert Settings().detection_source == "auto"


def test_approval_expiry_is_72_hours():
    assert Settings().approval_expiry_hours == 72



# ------------------------------------------------------------------ LLM providers

class _Secrets:
    def __init__(self, values):
        self.values = values

    def get(self, scope, key, required=True):
        return self.values.get(key, "")


def _settings(**values):
    return Settings(_resolver=_Secrets(values))


FOUNDRY = {"foundry-endpoint": "https://airo-ai.openai.azure.com/",
           "foundry-api-key": "k", "foundry-deployment": "gpt-6-1-luna"}


def test_foundry_base_url_accepts_any_portal_form():
    from agentic_ai.config import foundry_base_url

    expected = "https://airo-ai.openai.azure.com/openai/v1/"
    assert foundry_base_url("https://airo-ai.openai.azure.com") == expected
    assert foundry_base_url("https://airo-ai.openai.azure.com/openai/v1/") == expected
    assert foundry_base_url("https://airo-ai.openai.azure.com/openai/deployments/luna/chat/completions"
                            "?api-version=2025-01-01-preview") == expected
    assert foundry_base_url("https://airo.services.ai.azure.com/api/projects/p") == \
        "https://airo.services.ai.azure.com/openai/v1/"


def test_foundry_is_primary_and_fallbacks_are_optional():
    providers = _settings(**FOUNDRY, **{"groq-api-key-1": "g"}).llm_providers
    assert [p.name for p in providers] == ["Azure AI Foundry", "Groq Primary"]
    assert providers[0].model == "gpt-6-1-luna" and providers[0].reasoning is True
    assert providers[1].reasoning is False

    only = _settings(**FOUNDRY, **{"foundry-reasoning": "false"}).llm_providers
    assert len(only) == 1 and only[0].reasoning is False
def test_no_provider_configured_raises():
    import pytest

    from agentic_ai.config import NoLLMConfigured

    with pytest.raises(NoLLMConfigured):
        _ = _settings().llm_providers
    with pytest.raises(NoLLMConfigured):            # Foundry needs all three secrets
        _ = _settings(**{"foundry-endpoint": "https://x.openai.azure.com"}).llm_providers




# ------------------------------------------------------------------ LLM providers

class _Secrets:
    def __init__(self, values):
        self.values = values

    def get(self, scope, key, required=True):
        return self.values.get(key, "")


def _settings(**values):
    return Settings(_resolver=_Secrets(values))


FOUNDRY = {"foundry-endpoint": "https://airo-ai.openai.azure.com/",
           "foundry-api-key": "k", "foundry-deployment": "gpt-6-1-luna"}


def test_foundry_base_url_accepts_any_portal_form():
    from agentic_ai.config import foundry_base_url

    expected = "https://airo-ai.openai.azure.com/openai/v1/"
    assert foundry_base_url("https://airo-ai.openai.azure.com") == expected
    assert foundry_base_url("https://airo-ai.openai.azure.com/openai/v1/") == expected
    assert foundry_base_url("https://airo-ai.openai.azure.com/openai/deployments/luna/chat/completions"
                            "?api-version=2025-01-01-preview") == expected
    assert foundry_base_url("https://airo.services.ai.azure.com/api/projects/p") == \
        "https://airo.services.ai.azure.com/openai/v1/"


def test_foundry_is_primary_and_fallbacks_are_optional():
    providers = _settings(**FOUNDRY, **{"groq-api-key-1": "g"}).llm_providers
    assert [p.name for p in providers] == ["Azure AI Foundry", "Groq Primary"]
    assert providers[0].model == "gpt-6-1-luna" and providers[0].reasoning is True
    assert providers[1].reasoning is False

    only = _settings(**FOUNDRY, **{"foundry-reasoning": "false"}).llm_providers
    assert len(only) == 1 and only[0].reasoning is False


def test_no_provider_configured_raises():
    import pytest

    from agentic_ai.config import NoLLMConfigured

    with pytest.raises(NoLLMConfigured):
        _ = _settings().llm_providers
    with pytest.raises(NoLLMConfigured):            # Foundry needs all three secrets
        _ = _settings(**{"foundry-endpoint": "https://x.openai.azure.com"}).llm_providers


def test_request_args_follow_the_provider_shape():
    from agentic_ai.config import LLMProvider
    from agentic_ai.llm.client import REASONING_MIN_TOKENS, _request_args

    reasoning = LLMProvider("f", "k", "u", "m", reasoning=True)
    assert _request_args(reasoning, 2000, 0.3) == {"max_completion_tokens": REASONING_MIN_TOKENS}
    assert _request_args(reasoning, 8000, 0.3) == {"max_completion_tokens": 8000}
    plain = LLMProvider("g", "k", "u", "m")
    assert _request_args(plain, 2000, 0.3) == {"max_tokens": 2000, "temperature": 0.3}
