from src.core.ports.llm_port import LLMPort
from src.infrastructure.config import settings
from src.infrastructure.logger import logger


def get_llm_adapter() -> LLMPort:
    """Return the correct LLM adapter based on runtime or config provider setting."""
    from src.infrastructure import runtime_config
    provider = runtime_config.get_provider()

    mistral_key = (settings.MISTRAL_API_KEY or "").strip()
    groq_key = (settings.GROQ_API_KEY or "").strip()

    if provider == "mistral" and not mistral_key:
        logger.warning("MISTRAL_API_KEY missing, falling back to groq.")
        provider = "groq"

    if provider == "groq" and not groq_key:
        logger.warning("GROQ_API_KEY missing, falling back to mistral.")
        provider = "mistral"

    from src.adapters.outbound.llm.grok_adapter import GrokAdapter

    logger.info(f"LLM adapter selected: provider={provider}")
    return GrokAdapter(provider=provider)
