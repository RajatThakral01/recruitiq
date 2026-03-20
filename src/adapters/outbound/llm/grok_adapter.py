import openai

from src.adapters.outbound.llm.claude_adapter import BaseLLMAdapter
from src.infrastructure.config import settings
from src.infrastructure.exceptions import ScoringException
from src.infrastructure.logger import logger


class GrokAdapter(BaseLLMAdapter):
    """OpenAI-compatible adapter supporting Groq and Mistral providers."""

    def __init__(self, provider: str | None = None) -> None:
        selected_provider = (provider or settings.LLM_PROVIDER or "groq").strip().lower()

        def _normalize_key(raw: str) -> str:
            key = (raw or "").strip()
            if not key or key.startswith("your_"):
                return ""
            return key

        if selected_provider == "groq":
            api_key = _normalize_key(settings.GROQ_API_KEY or "")
            base_url = settings.GROQ_BASE_URL
            model = settings.GROQ_MODEL
            if not api_key:
                raise ScoringException(
                    "GROQ_API_KEY is missing.",
                    detail="Set GROQ_API_KEY in your .env file.",
                )

        elif selected_provider == "mistral":
            api_key = _normalize_key(settings.MISTRAL_API_KEY or "")
            base_url = settings.MISTRAL_BASE_URL
            model = settings.MISTRAL_MODEL
            if not api_key:
                raise ScoringException(
                    "MISTRAL_API_KEY is missing.",
                    detail="Set MISTRAL_API_KEY in your .env file.",
                )

        else:
            raise ScoringException(
                f"Unsupported provider: {selected_provider}",
                detail="LLM_PROVIDER must be one of: groq, mistral.",
            )

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self._auth_failed = False

        logger.info(
            f"GrokAdapter initialized: provider={selected_provider} model={self.model}"
        )
