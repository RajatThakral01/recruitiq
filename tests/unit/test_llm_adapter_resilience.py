import json

import pytest

from src.adapters.outbound.llm.claude_adapter import BaseLLMAdapter
from src.infrastructure.exceptions import LLMTimeoutException


class _FakeCompletions:
    def __init__(self, side_effects):
        self.side_effects = list(side_effects)

    def create(self, **kwargs):
        effect = self.side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect

        class _Message:
            content = effect

        class _Choice:
            message = _Message()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class _FakeClient:
    def __init__(self, side_effects):
        self.chat = type("_Chat", (), {"completions": _FakeCompletions(side_effects)})()


def _adapter_with_client(side_effects):
    adapter = BaseLLMAdapter.__new__(BaseLLMAdapter)
    adapter.client = _FakeClient(side_effects)
    adapter.model = "fake-model"
    return adapter


def test_call_llm_json_falls_back_on_malformed_json(monkeypatch):
    adapter = _adapter_with_client(["not-json", "also-not-json"])

    result = adapter._call_llm_json(
        system_prompt="sys",
        user_content="user",
        max_tokens=50,
        label="test",
        validator=lambda payload: payload,
        default_payload={"skills_score": 50.0},
        parse_retries=2,
    )

    assert result == {"skills_score": 50.0}


def test_call_llm_retries_and_raises_timeout(monkeypatch):
    from src.infrastructure import config

    monkeypatch.setattr(config.settings, "LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(config.settings, "LLM_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(config.settings, "LLM_BACKOFF_BASE_SECONDS", 0.0)

    adapter = _adapter_with_client([
        RuntimeError("429 too many requests"),
        RuntimeError("500 provider error"),
    ])

    with pytest.raises(LLMTimeoutException):
        adapter._call_llm("sys", "user", max_tokens=20)


def test_call_llm_json_handles_missing_fields_with_validator_defaults():
    adapter = _adapter_with_client([json.dumps({"skills_score": 88})])

    def validator(payload):
        return {
            "skills_score": float(payload.get("skills_score", 50.0)),
            "reasoning": str(payload.get("reasoning", "")),
        }

    result = adapter._call_llm_json(
        system_prompt="sys",
        user_content="user",
        max_tokens=50,
        label="test",
        validator=validator,
        default_payload={"skills_score": 50.0, "reasoning": "fallback"},
    )

    assert result["skills_score"] == 88.0
    assert result["reasoning"] == ""
