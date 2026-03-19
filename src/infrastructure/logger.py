import sys
from pathlib import Path
from loguru import logger
from src.infrastructure.config import settings

def setup_logger():
    """
    Configures loguru logger with console and rotating file sinks.
    """
    # Remove default loguru handler
    logger.remove()

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Console sink
    logger.add(
        sys.stdout, 
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", 
        level=settings.LOG_LEVEL.upper(),
        enqueue=True
    )

    # File sink (rotating/retention)
    logger.add(
        "logs/app.log", 
        rotation="10 MB", 
        retention="7 days",
        level=settings.LOG_LEVEL.upper(),
        enqueue=True
    )

    return logger

# Export configured logger instance
logger = setup_logger()
