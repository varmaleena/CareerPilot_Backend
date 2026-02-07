import sys
from loguru import logger
from app.config import settings

def setup_logging():
    """Configure Loguru logger."""
    logger.remove()
    
    logger.add(
        sys.stderr,
        level="DEBUG" if settings.DEBUG else "INFO",
        format="{time} {level} {message}",
        serialize=not settings.DEBUG,  # JSON logs in production
    )
