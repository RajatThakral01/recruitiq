from src.infrastructure.config import Settings


def test_scoring_mode_accepts_supported_values() -> None:
    assert Settings(DATABASE_URL="postgresql://x", GROK_API_KEY="k", SCORING_MODE="legacy").SCORING_MODE == "legacy"
    assert Settings(DATABASE_URL="postgresql://x", GROK_API_KEY="k", SCORING_MODE="hybrid").SCORING_MODE == "hybrid"
    assert Settings(DATABASE_URL="postgresql://x", GROK_API_KEY="k", SCORING_MODE="llm_first").SCORING_MODE == "llm_first"


def test_scoring_mode_normalizes_case_and_invalid_values() -> None:
    assert Settings(DATABASE_URL="postgresql://x", GROK_API_KEY="k", SCORING_MODE=" HYBRID ").SCORING_MODE == "hybrid"
    assert Settings(DATABASE_URL="postgresql://x", GROK_API_KEY="k", SCORING_MODE="unknown_mode").SCORING_MODE == "legacy"
