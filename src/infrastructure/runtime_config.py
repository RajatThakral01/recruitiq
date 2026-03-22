from src.infrastructure.config import settings
from src.infrastructure.logger import logger

_current_provider: str = ""


def get_provider() -> str:
    """Get the currently active LLM provider for this session.
    Always falls back to environment setting if no runtime override is set.
    """
    if _current_provider and _current_provider.strip():
        return _current_provider
    return (settings.LLM_PROVIDER or "mistral").strip().lower()


def set_provider(provider: str) -> None:
    """Set the LLM provider for this session without restarting."""
    global _current_provider
    _current_provider = provider.strip().lower()
    logger.info(f"Runtime provider set to: {_current_provider}")


def reset_provider() -> None:
    """Reset runtime override — falls back to LLM_PROVIDER env variable."""
    global _current_provider
    _current_provider = ""
    logger.info("Runtime provider reset to environment default.")
