from src.infrastructure.config import settings
from src.infrastructure.logger import logger

_current_provider: str = ""


def get_provider() -> str:
    """Get the currently active LLM provider for this session."""
    return _current_provider or settings.LLM_PROVIDER


def set_provider(provider: str) -> None:
    """Set the LLM provider for this session without restarting."""
    global _current_provider
    _current_provider = provider.strip().lower()
    logger.info(f"Runtime provider set to: {_current_provider}")
