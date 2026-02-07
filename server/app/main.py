from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sentry_sdk

from app.config import settings
from app.api.v1.router import api_router
from app.db.session import engine
from app.middleware.error_handler import error_handler_middleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.utils.logger import setup_logging

# Initialize Sentry
if settings.SENTRY_DSN:
    sentry_sdk.init(dsn=settings.SENTRY_DSN, traces_sample_rate=0.1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    setup_logging()
    yield
    # Shutdown
    await engine.dispose()


app = FastAPI(
    title="CareerPilot AI API",
    description="Multi-Agent LLM Career Assistant",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
app.middleware("http")(error_handler_middleware)
app.add_middleware(RateLimitMiddleware)

# Routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0"}
