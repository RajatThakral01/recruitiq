import pytest

from src.services.llm_response_parser import LLMResponseParser


def test_parse_json_object_from_markdown_fence() -> None:
    raw = """```json
{"skills_score": 88, "reasoning": "good"}
```"""
    parsed = LLMResponseParser.parse_json_object(raw)
    assert parsed["skills_score"] == 88
    assert parsed["reasoning"] == "good"


def test_as_score_clamps_and_defaults() -> None:
    assert LLMResponseParser.as_score(120) == 100.0
    assert LLMResponseParser.as_score(-5) == 0.0
    assert LLMResponseParser.as_score("oops", default=42.0) == 42.0


def test_parse_json_object_raises_for_non_object() -> None:
    with pytest.raises(ValueError):
        LLMResponseParser.parse_json_object("[1,2,3]")
