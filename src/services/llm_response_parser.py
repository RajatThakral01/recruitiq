import json
from typing import Any


class LLMResponseParser:
    """Central utility for robustly parsing and normalizing LLM JSON outputs."""

    @staticmethod
    def strip_markdown(text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned.startswith("```"):
            return cleaned
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        return "\n".join(lines).strip()

    @staticmethod
    def extract_json_object(text: str) -> str:
        cleaned = LLMResponseParser.strip_markdown(text)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return cleaned[start : end + 1]
        return cleaned

    @staticmethod
    def parse_json_object(text: str) -> dict[str, Any]:
        candidate = LLMResponseParser.extract_json_object(text)
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            raise ValueError("Top-level JSON must be an object")
        return parsed

    @staticmethod
    def as_string(value: Any, default: str = "") -> str:
        if value is None:
            return default
        return str(value).strip()

    @staticmethod
    def as_string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        normalized: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized

    @staticmethod
    def as_float(value: Any, default: float = 0.0, minimum: float | None = None) -> float:
        try:
            result = float(value if value is not None else default)
        except (TypeError, ValueError):
            result = float(default)

        if minimum is not None:
            result = max(minimum, result)
        return result

    @staticmethod
    def as_score(value: Any, default: float = 50.0) -> float:
        score = LLMResponseParser.as_float(value, default=default)
        return max(0.0, min(100.0, score))
