from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger
import sentry_sdk
from app.config import settings


async def error_handler_middleware(request: Request, call_next):
    """Global error handling middleware."""
    try:
        return await call_next(request)
    except Exception as exc:
        # Log error
        logger.exception(f"Unhandled error: {exc}")
        
        # Report to Sentry
        sentry_sdk.capture_exception(exc)
        
        # Return generic error (don't expose internals)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.DEBUG else None,
            }
        )
