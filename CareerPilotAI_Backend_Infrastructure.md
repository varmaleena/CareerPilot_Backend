# CareerPilot AI - Backend Infrastructure

> Detailed FastAPI Backend Architecture for Multi-Agent LLM System

---

## Table of Contents

1. [Technology Stack](#1-technology-stack)
2. [Project Structure](#2-project-structure)
3. [FastAPI Application Setup](#3-fastapi-application-setup)
4. [Pydantic Models](#4-pydantic-models)
5. [Database Layer](#5-database-layer)
6. [Authentication](#6-authentication)
7. [API Endpoints](#7-api-endpoints)
8. [Middleware](#8-middleware)
9. [Caching Layer](#9-caching-layer)
10. [LLM Gateway](#10-llm-gateway)
11. [Environment Configuration](#11-environment-configuration)
12. [Deployment](#12-deployment)

---

## 1. Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Framework** | FastAPI 0.109+ | High-performance async API |
| **Server** | Uvicorn + Gunicorn | ASGI server |
| **Validation** | Pydantic v2 | Request/response validation |
| **Database** | PostgreSQL (Supabase) | Persistent storage |
| **ORM** | SQLAlchemy 2.0 | Async database operations |
| **Cache** | Redis (Upstash) | Caching + rate limiting |
| **Auth** | python-jose + Supabase | JWT authentication |
| **AI/Agents** | LangGraph + LangChain | Multi-agent orchestration |
| **LLM** | google-generativeai | Gemini API |
| **Queue** | Celery + Redis | Background jobs |
| **Monitoring** | Sentry + Loguru | Error tracking + logging |

### Core Dependencies (pyproject.toml)

```toml
[tool.poetry.dependencies]
python = "^3.11"

# Web Framework
fastapi = "^0.109.0"
uvicorn = {extras = ["standard"], version = "^0.27.0"}
python-multipart = "^0.0.6"

# Validation & Settings
pydantic = "^2.5.0"
pydantic-settings = "^2.1.0"

# Database
sqlalchemy = {extras = ["asyncio"], version = "^2.0.25"}
asyncpg = "^0.29.0"
alembic = "^1.13.0"

# Cache & Queue
redis = "^5.0.0"
celery = "^5.3.0"

# Authentication
python-jose = {extras = ["cryptography"], version = "^3.3.0"}
passlib = {extras = ["bcrypt"], version = "^1.7.4"}

# AI/LLM
langchain = "^0.1.0"
langgraph = "^0.0.20"
google-generativeai = "^0.3.0"

# Utilities
httpx = "^0.26.0"
loguru = "^0.7.2"
sentry-sdk = {extras = ["fastapi"], version = "^1.39.0"}
```

---

## 2. Project Structure

```
server/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application entry point
│   ├── config.py                  # Settings & environment variables
│   │
│   ├── api/                       # API Layer
│   │   ├── __init__.py
│   │   ├── deps.py                # Dependency injection
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── router.py          # Main API router
│   │       ├── analyze.py         # POST /analyze
│   │       ├── plan.py            # POST /plan
│   │       ├── interview.py       # POST /interview/*
│   │       ├── resume.py          # POST /resume/*
│   │       ├── projects.py        # POST /projects
│   │       └── auth.py            # POST /auth/*
│   │
│   ├── agents/                    # Multi-Agent System
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── orchestrator.py    # LangGraph state machine
│   │   │   ├── decision_engine.py # Decision point evaluation
│   │   │   ├── message_bus.py     # Agent communication
│   │   │   └── base_agent.py      # Abstract agent class
│   │   │
│   │   ├── masters/               # Master agents (expensive)
│   │   │   ├── __init__.py
│   │   │   ├── strategist.py
│   │   │   ├── evaluator.py
│   │   │   └── resolver.py
│   │   │
│   │   ├── helpers/               # Helper agents (cheap)
│   │   │   ├── __init__.py
│   │   │   ├── extractor.py
│   │   │   ├── generator.py
│   │   │   ├── validator.py
│   │   │   └── formatter.py
│   │   │
│   │   ├── workflows/             # LangGraph workflow definitions
│   │   │   ├── __init__.py
│   │   │   ├── resume_analysis.py
│   │   │   ├── interview.py
│   │   │   ├── learning_plan.py
│   │   │   └── project_ideas.py
│   │   │
│   │   └── prompts/               # Prompt templates
│   │       ├── strategist/
│   │       ├── evaluator/
│   │       └── helpers/
│   │
│   ├── models/                    # Pydantic Models
│   │   ├── __init__.py
│   │   ├── requests.py            # API request schemas
│   │   ├── responses.py           # API response schemas
│   │   ├── domain.py              # Business domain models
│   │   └── agents.py              # Agent-related models
│   │
│   ├── db/                        # Database Layer
│   │   ├── __init__.py
│   │   ├── session.py             # SQLAlchemy async session
│   │   ├── base.py                # Base model class
│   │   └── models/                # SQLAlchemy ORM models
│   │       ├── __init__.py
│   │       ├── user.py
│   │       ├── analysis.py
│   │       ├── session.py
│   │       └── usage.py
│   │
│   ├── repositories/              # Data Access Layer
│   │   ├── __init__.py
│   │   ├── base.py                # Base repository
│   │   ├── user.py
│   │   ├── analysis.py
│   │   └── session.py
│   │
│   ├── services/                  # Business Logic
│   │   ├── __init__.py
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   ├── gateway.py         # LLM abstraction
│   │   │   ├── gemini.py          # Gemini implementation
│   │   │   ├── key_manager.py     # API key rotation
│   │   │   ├── model_router.py    # Cost-aware routing
│   │   │   └── token_counter.py   # Token tracking
│   │   │
│   │   ├── cache/
│   │   │   ├── __init__.py
│   │   │   ├── redis_client.py    # Redis connection
│   │   │   ├── semantic_cache.py  # LLM response caching
│   │   │   └── session_store.py   # Interview sessions
│   │   │
│   │   └── quota/
│   │       ├── __init__.py
│   │       ├── manager.py         # Per-user limits
│   │       └── tracker.py         # Usage tracking
│   │
│   ├── middleware/                # Custom Middleware
│   │   ├── __init__.py
│   │   ├── auth.py                # JWT verification
│   │   ├── rate_limit.py          # Rate limiting
│   │   ├── error_handler.py       # Global error handling
│   │   └── cost_tracker.py        # LLM cost tracking
│   │
│   └── utils/                     # Utilities
│       ├── __init__.py
│       ├── logger.py              # Loguru configuration
│       ├── json_repair.py         # Fix malformed JSON
│       └── hash.py                # Hashing utilities
│
├── migrations/                    # Alembic migrations
│   ├── versions/
│   └── env.py
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── alembic.ini
└── .env.example
```

---

## 3. FastAPI Application Setup

### Main Application (app/main.py)

```python
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
```

### Configuration (app/config.py)

```python
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    # App
    APP_ENV: str = "development"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str
    DATABASE_POOL_SIZE: int = 10
    
    # Redis
    REDIS_URL: str
    
    # Supabase
    SUPABASE_URL: str
    SUPABASE_KEY: str
    SUPABASE_JWT_SECRET: str
    
    # Gemini API Keys (7 for rotation)
    GEMINI_API_KEYS: list[str]
    
    # Security
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:5173"]
    
    # Monitoring
    SENTRY_DSN: str | None = None
    
    # Rate Limits
    RATE_LIMIT_ANALYZE: int = 5      # per hour
    RATE_LIMIT_INTERVIEW: int = 100  # per minute
    RATE_LIMIT_PLAN: int = 10        # per hour

    class Config:
        env_file = ".env"
        
    @property
    def gemini_keys_list(self) -> list[str]:
        if isinstance(self.GEMINI_API_KEYS, str):
            return self.GEMINI_API_KEYS.split(",")
        return self.GEMINI_API_KEYS


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
```

---

## 4. Pydantic Models

### Request Models (app/models/requests.py)

```python
from pydantic import BaseModel, Field, EmailStr
from enum import Enum


class InterviewType(str, Enum):
    BEHAVIORAL = "behavioral"
    TECHNICAL = "technical"
    SYSTEM_DESIGN = "system_design"
    DSA = "dsa"


class AnalyzeRequest(BaseModel):
    """Resume analysis request."""
    resume_text: str = Field(..., min_length=100, max_length=50000)
    target_role: str = Field(..., min_length=2, max_length=200)
    target_company: str | None = Field(None, max_length=200)

    class Config:
        json_schema_extra = {
            "example": {
                "resume_text": "Software Engineer with 5 years...",
                "target_role": "Senior Software Engineer",
                "target_company": "Google"
            }
        }


class InterviewStartRequest(BaseModel):
    """Start interview session."""
    interview_type: InterviewType
    difficulty: str = Field("medium", pattern="^(easy|medium|hard)$")
    duration_minutes: int = Field(30, ge=10, le=60)


class InterviewMessageRequest(BaseModel):
    """Send message in interview."""
    session_id: str
    message: str = Field(..., min_length=1, max_length=5000)


class PlanRequest(BaseModel):
    """Learning plan generation request."""
    target_role: str
    current_skills: list[str] = Field(default_factory=list)
    timeline_weeks: int = Field(12, ge=4, le=52)
    hours_per_week: int = Field(10, ge=5, le=40)


class ResumeOptimizeRequest(BaseModel):
    """Resume optimization request."""
    resume_text: str = Field(..., min_length=100)
    target_role: str
    optimization_focus: list[str] = Field(
        default_factory=lambda: ["ats_keywords", "impact_metrics"]
    )
```

### Response Models (app/models/responses.py)

```python
from pydantic import BaseModel
from datetime import datetime
from typing import Any


class SkillGap(BaseModel):
    skill: str
    current_level: int  # 0-100
    required_level: int
    priority: str  # high, medium, low


class AnalysisResponse(BaseModel):
    """Resume analysis response."""
    analysis_id: str
    readiness_score: int  # 0-100
    skill_gaps: list[SkillGap]
    strengths: list[str]
    weaknesses: list[str]
    recommendations: list[str]
    ats_score: int
    created_at: datetime


class InterviewSessionResponse(BaseModel):
    """Interview session info."""
    session_id: str
    interview_type: str
    status: str
    messages: list[dict[str, Any]]
    started_at: datetime


class InterviewFeedback(BaseModel):
    """Interview feedback response."""
    overall_score: int
    communication_score: int
    technical_score: int
    strengths: list[str]
    improvements: list[str]
    detailed_feedback: str


class LearningPlanResponse(BaseModel):
    """Learning plan response."""
    plan_id: str
    weeks: list[dict[str, Any]]
    resources: list[dict[str, Any]]
    milestones: list[str]


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str | None = None
    code: str | None = None
```

### Agent Models (app/models/agents.py)

```python
from pydantic import BaseModel
from enum import Enum
from typing import Any


class AgentRole(str, Enum):
    STRATEGIST = "strategist"
    EVALUATOR = "evaluator"
    RESOLVER = "resolver"
    EXTRACTOR = "extractor"
    GENERATOR = "generator"
    VALIDATOR = "validator"
    FORMATTER = "formatter"


class Complexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class Verdict(str, Enum):
    APPROVE = "approve"
    REVISE = "revise"
    REJECT = "reject"
    ESCALATE = "escalate"


class ExecutionStep(BaseModel):
    """Single step in execution plan."""
    step: int
    agent: AgentRole
    task: str
    fallback_if: str | None = None
    timeout_seconds: int = 30


class StrategistOutput(BaseModel):
    """Output from Strategist agent."""
    complexity: Complexity
    confidence: int  # 0-100
    execution_plan: list[ExecutionStep]
    needs_clarification: bool = False
    clarification_question: str | None = None


class EvaluatorOutput(BaseModel):
    """Output from Evaluator agent."""
    overall_score: int  # 0-100
    verdict: Verdict
    issues: list[str] = []
    revision_instructions: str | None = None


class AgentMessage(BaseModel):
    """Message passed between agents."""
    from_agent: AgentRole
    to_agent: AgentRole
    payload: dict[str, Any]
    timestamp: datetime
```

---

## 5. Database Layer

### Session Setup (app/db/session.py)

```python
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from app.config import settings

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
    pool_size=settings.DATABASE_POOL_SIZE,
    pool_pre_ping=True,
    echo=settings.DEBUG,
)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncSession:
    """Dependency for getting database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

### Database Models (app/db/models/user.py)

```python
from sqlalchemy import String, DateTime, Boolean, JSON
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
import uuid

from app.db.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[str | None] = mapped_column(String(255))
    profile_data: Mapped[dict] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    resume_hash: Mapped[str] = mapped_column(String(64))
    target_role: Mapped[str] = mapped_column(String(255))
    result: Mapped[dict] = mapped_column(JSON)
    tokens_used: Mapped[int] = mapped_column(default=0)
    cost_usd: Mapped[float] = mapped_column(default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    interview_type: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(20), default="active")
    messages: Mapped[list] = mapped_column(JSON, default=list)
    feedback: Mapped[dict | None] = mapped_column(JSON)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime)


class UsageLog(Base):
    __tablename__ = "usage_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    endpoint: Mapped[str] = mapped_column(String(100))
    tokens_used: Mapped[int] = mapped_column(default=0)
    cost_usd: Mapped[float] = mapped_column(default=0.0)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

### Repository Pattern (app/repositories/base.py)

```python
from typing import TypeVar, Generic, Type
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from app.db.base import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Base repository with common CRUD operations."""

    def __init__(self, model: Type[ModelType], session: AsyncSession):
        self.model = model
        self.session = session

    async def get(self, id: str) -> ModelType | None:
        result = await self.session.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalar_one_or_none()

    async def get_by_user(self, user_id: str, limit: int = 100) -> list[ModelType]:
        result = await self.session.execute(
            select(self.model)
            .where(self.model.user_id == user_id)
            .order_by(self.model.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def create(self, **data) -> ModelType:
        instance = self.model(**data)
        self.session.add(instance)
        await self.session.flush()
        return instance

    async def update(self, id: str, **data) -> ModelType | None:
        await self.session.execute(
            update(self.model).where(self.model.id == id).values(**data)
        )
        return await self.get(id)

    async def delete(self, id: str) -> bool:
        result = await self.session.execute(
            delete(self.model).where(self.model.id == id)
        )
        return result.rowcount > 0
```

---

## 6. Authentication

### JWT Middleware (app/middleware/auth.py)

```python
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from app.config import settings

security = HTTPBearer()


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """Verify JWT token from Supabase."""
    token = credentials.credentials
    
    try:
        payload = jwt.decode(
            token,
            settings.SUPABASE_JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
            audience="authenticated",
        )
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid authentication token: {e}",
        )


async def get_current_user(payload: dict = Depends(verify_token)) -> dict:
    """Extract user info from verified token."""
    user_id = payload.get("sub")
    email = payload.get("email")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    
    return {
        "id": user_id,
        "email": email,
        "role": payload.get("role", "user"),
    }
```

### Dependency Injection (app/api/deps.py)

```python
from typing import Annotated
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.middleware.auth import get_current_user
from app.services.llm.gateway import LLMGateway
from app.services.cache.redis_client import get_redis

# Typed dependencies
DBSession = Annotated[AsyncSession, Depends(get_db)]
CurrentUser = Annotated[dict, Depends(get_current_user)]
Redis = Annotated[object, Depends(get_redis)]


async def get_llm_gateway() -> LLMGateway:
    """Get LLM gateway instance."""
    return LLMGateway()


LLM = Annotated[LLMGateway, Depends(get_llm_gateway)]
```

---

## 7. API Endpoints

### Router Setup (app/api/v1/router.py)

```python
from fastapi import APIRouter
from app.api.v1 import analyze, plan, interview, resume, projects, auth

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(analyze.router, prefix="/analyze", tags=["Analysis"])
api_router.include_router(plan.router, prefix="/plan", tags=["Learning Plan"])
api_router.include_router(interview.router, prefix="/interview", tags=["Interview"])
api_router.include_router(resume.router, prefix="/resume", tags=["Resume"])
api_router.include_router(projects.router, prefix="/projects", tags=["Projects"])
```

### Analyze Endpoint (app/api/v1/analyze.py)

```python
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.api.deps import DBSession, CurrentUser, LLM
from app.models.requests import AnalyzeRequest
from app.models.responses import AnalysisResponse
from app.agents.workflows.resume_analysis import ResumeAnalysisWorkflow
from app.repositories.analysis import AnalysisRepository
from app.services.quota.manager import QuotaManager

router = APIRouter()


@router.post("", response_model=AnalysisResponse)
async def analyze_resume(
    request: AnalyzeRequest,
    db: DBSession,
    user: CurrentUser,
    llm: LLM,
    background_tasks: BackgroundTasks,
):
    """Analyze resume and generate insights."""
    
    # Check quota
    quota = QuotaManager(db, user["id"])
    if not await quota.can_use("analyze"):
        raise HTTPException(429, "Daily analysis limit reached")
    
    # Run analysis workflow
    workflow = ResumeAnalysisWorkflow(llm)
    result = await workflow.run(
        resume_text=request.resume_text,
        target_role=request.target_role,
        target_company=request.target_company,
    )
    
    # Save to database
    repo = AnalysisRepository(db)
    analysis = await repo.create(
        user_id=user["id"],
        resume_hash=hash(request.resume_text),
        target_role=request.target_role,
        result=result.model_dump(),
        tokens_used=result.tokens_used,
        cost_usd=result.cost_usd,
    )
    
    # Track usage in background
    background_tasks.add_task(quota.record_usage, "analyze", result.tokens_used)
    
    return AnalysisResponse(
        analysis_id=analysis.id,
        **result.model_dump(),
        created_at=analysis.created_at,
    )
```

### Interview Endpoints (app/api/v1/interview.py)

```python
from fastapi import APIRouter, HTTPException, WebSocket
from app.api.deps import DBSession, CurrentUser, LLM
from app.models.requests import InterviewStartRequest, InterviewMessageRequest
from app.models.responses import InterviewSessionResponse, InterviewFeedback
from app.agents.workflows.interview import InterviewWorkflow
from app.services.cache.session_store import SessionStore

router = APIRouter()


@router.post("/start", response_model=InterviewSessionResponse)
async def start_interview(
    request: InterviewStartRequest,
    db: DBSession,
    user: CurrentUser,
    llm: LLM,
):
    """Start a new interview session."""
    workflow = InterviewWorkflow(llm)
    session = await workflow.start(
        user_id=user["id"],
        interview_type=request.interview_type,
        difficulty=request.difficulty,
        duration=request.duration_minutes,
    )
    return session


@router.post("/message", response_model=InterviewSessionResponse)
async def send_message(
    request: InterviewMessageRequest,
    db: DBSession,
    user: CurrentUser,
    llm: LLM,
):
    """Send message in interview session."""
    workflow = InterviewWorkflow(llm)
    session = await workflow.process_message(
        session_id=request.session_id,
        user_id=user["id"],
        message=request.message,
    )
    return session


@router.post("/{session_id}/end", response_model=InterviewFeedback)
async def end_interview(
    session_id: str,
    db: DBSession,
    user: CurrentUser,
    llm: LLM,
):
    """End interview and get feedback."""
    workflow = InterviewWorkflow(llm)
    feedback = await workflow.end(
        session_id=session_id,
        user_id=user["id"],
    )
    return feedback


@router.websocket("/ws/{session_id}")
async def interview_websocket(
    websocket: WebSocket,
    session_id: str,
):
    """WebSocket for real-time interview."""
    await websocket.accept()
    
    session_store = SessionStore()
    session = await session_store.get(session_id)
    
    if not session:
        await websocket.close(code=4004)
        return
    
    try:
        while True:
            data = await websocket.receive_json()
            # Process message through workflow
            response = await process_interview_message(session_id, data)
            await websocket.send_json(response)
    except Exception:
        await websocket.close()
```

---

## 8. Middleware

### Rate Limiting (app/middleware/rate_limit.py)

```python
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from app.services.cache.redis_client import redis_client
from app.config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-user, per-endpoint rate limiting."""
    
    LIMITS = {
        "/api/v1/analyze": (settings.RATE_LIMIT_ANALYZE, 3600),
        "/api/v1/interview/message": (settings.RATE_LIMIT_INTERVIEW, 60),
        "/api/v1/plan": (settings.RATE_LIMIT_PLAN, 3600),
    }
    
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        
        if path not in self.LIMITS:
            return await call_next(request)
        
        # Get user ID from JWT (simplified)
        user_id = request.state.user_id if hasattr(request.state, "user_id") else "anon"
        limit, window = self.LIMITS[path]
        
        key = f"rate:{user_id}:{path}"
        current = await redis_client.incr(key)
        
        if current == 1:
            await redis_client.expire(key, window)
        
        if current > limit:
            raise HTTPException(
                429,
                detail=f"Rate limit exceeded. Try again in {window} seconds."
            )
        
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, limit - current))
        return response
```

### Error Handler (app/middleware/error_handler.py)

```python
from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger
import sentry_sdk
import traceback


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
```

---

## 9. Caching Layer

### Redis Client (app/services/cache/redis_client.py)

```python
import redis.asyncio as redis
from app.config import settings


class RedisClient:
    """Async Redis client wrapper."""
    
    def __init__(self):
        self._client: redis.Redis | None = None
    
    async def connect(self) -> redis.Redis:
        if not self._client:
            self._client = await redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._client
    
    async def get(self, key: str) -> str | None:
        client = await self.connect()
        return await client.get(key)
    
    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        client = await self.connect()
        await client.setex(key, ttl, value)
    
    async def incr(self, key: str) -> int:
        client = await self.connect()
        return await client.incr(key)
    
    async def expire(self, key: str, seconds: int) -> None:
        client = await self.connect()
        await client.expire(key, seconds)


redis_client = RedisClient()


async def get_redis() -> RedisClient:
    return redis_client
```

### Semantic Cache (app/services/cache/semantic_cache.py)

```python
import hashlib
import json
from app.services.cache.redis_client import redis_client


class SemanticCache:
    """Cache LLM responses based on prompt similarity."""
    
    TTL_CONFIG = {
        "validate": 86400,    # 24h
        "extract": 43200,     # 12h
        "generate": 21600,    # 6h
        "interview": 7200,    # 2h
    }
    
    def _hash_prompt(self, prompt: str) -> str:
        """Create hash of prompt for cache key."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
    
    async def get(self, prompt: str, operation: str) -> str | None:
        """Get cached response."""
        key = f"llm:{operation}:{self._hash_prompt(prompt)}"
        cached = await redis_client.get(key)
        return cached
    
    async def set(
        self, prompt: str, response: str, operation: str
    ) -> None:
        """Cache response."""
        key = f"llm:{operation}:{self._hash_prompt(prompt)}"
        ttl = self.TTL_CONFIG.get(operation, 3600)
        await redis_client.set(key, response, ttl)
    
    async def get_or_compute(
        self,
        prompt: str,
        operation: str,
        compute_fn,
    ) -> str:
        """Get from cache or compute and cache."""
        cached = await self.get(prompt, operation)
        if cached:
            return cached
        
        result = await compute_fn()
        await self.set(prompt, result, operation)
        return result
```

---

## 10. LLM Gateway

### Gateway (app/services/llm/gateway.py)

```python
from typing import Literal
import google.generativeai as genai
from app.services.llm.key_manager import KeyManager
from app.services.llm.model_router import ModelRouter
from app.services.llm.token_counter import TokenCounter
from app.services.cache.semantic_cache import SemanticCache


class LLMGateway:
    """Unified LLM interface with cost optimization."""
    
    def __init__(self):
        self.key_manager = KeyManager()
        self.model_router = ModelRouter()
        self.token_counter = TokenCounter()
        self.cache = SemanticCache()
    
    async def generate(
        self,
        prompt: str,
        task: Literal["validate", "extract", "generate", "reason"],
        max_tokens: int | None = None,
        use_cache: bool = True,
    ) -> dict:
        """Generate LLM response with optimizations."""
        
        # Try cache first
        if use_cache:
            cached = await self.cache.get(prompt, task)
            if cached:
                return {"text": cached, "cached": True, "tokens": 0, "cost": 0}
        
        # Get optimal model and key
        model_config = self.model_router.route(task)
        api_key = self.key_manager.get_next_key()
        
        # Configure client
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_config.model)
        
        # Generate
        response = await model.generate_content_async(
            prompt,
            generation_config={
                "max_output_tokens": max_tokens or model_config.max_tokens,
                "temperature": model_config.temperature,
            }
        )
        
        text = response.text
        
        # Calculate tokens and cost
        tokens = self.token_counter.count(prompt, text)
        cost = self.token_counter.calculate_cost(tokens, model_config.model)
        
        # Cache result
        if use_cache:
            await self.cache.set(prompt, text, task)
        
        return {
            "text": text,
            "cached": False,
            "tokens": tokens,
            "cost": cost,
            "model": model_config.model,
        }
```

### Model Router (app/services/llm/model_router.py)

```python
from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    model: str
    max_tokens: int
    temperature: float
    cost_per_1k_tokens: float


class ModelRouter:
    """Cost-aware model selection."""
    
    MODELS = {
        "lite": ModelConfig(
            model="gemini-2.0-flash-lite",
            max_tokens=500,
            temperature=0.3,
            cost_per_1k_tokens=0.0001,
        ),
        "flash": ModelConfig(
            model="gemini-2.5-flash",
            max_tokens=2000,
            temperature=0.7,
            cost_per_1k_tokens=0.001,
        ),
        "flash-thinking": ModelConfig(
            model="gemini-2.5-flash-thinking",
            max_tokens=2000,
            temperature=0.5,
            cost_per_1k_tokens=0.002,
        ),
        "pro": ModelConfig(
            model="gemini-2.5-pro",
            max_tokens=4000,
            temperature=0.7,
            cost_per_1k_tokens=0.005,
        ),
    }
    
    TASK_ROUTING = {
        "validate": "lite",
        "extract": "lite",
        "format": "lite",
        "generate": "flash",
        "evaluate": "flash-thinking",
        "reason": "pro",
        "resolve": "pro",
    }
    
    def route(
        self, task: Literal["validate", "extract", "generate", "evaluate", "reason", "resolve", "format"]
    ) -> ModelConfig:
        """Get optimal model config for task."""
        model_key = self.TASK_ROUTING.get(task, "flash")
        return self.MODELS[model_key]
```

### Key Manager (app/services/llm/key_manager.py)

```python
import asyncio
from app.config import settings


class KeyManager:
    """Rotate Gemini API keys to avoid rate limits."""
    
    def __init__(self):
        self.keys = settings.gemini_keys_list
        self.current_index = 0
        self.lock = asyncio.Lock()
        self.failed_keys: set[int] = set()
    
    def get_next_key(self) -> str:
        """Get next available API key (round-robin)."""
        # Skip failed keys
        attempts = 0
        while attempts < len(self.keys):
            key_index = self.current_index % len(self.keys)
            self.current_index += 1
            
            if key_index not in self.failed_keys:
                return self.keys[key_index]
            attempts += 1
        
        # All keys failed, reset and try first
        self.failed_keys.clear()
        return self.keys[0]
    
    def mark_failed(self, key: str) -> None:
        """Mark key as temporarily failed."""
        try:
            index = self.keys.index(key)
            self.failed_keys.add(index)
        except ValueError:
            pass
    
    def reset_failed(self) -> None:
        """Reset all failed keys."""
        self.failed_keys.clear()
```

---

## 11. Environment Configuration

### .env.example

```env
# Application
APP_ENV=development
DEBUG=true

# Database (Supabase)
DATABASE_URL=postgresql://postgres:password@db.xxxx.supabase.co:5432/postgres
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
SUPABASE_JWT_SECRET=your-jwt-secret

# Redis (Upstash)
REDIS_URL=redis://default:xxxx@xxxx.upstash.io:6379

# Gemini API Keys (comma-separated, 7 keys for rotation)
GEMINI_API_KEYS=key1,key2,key3,key4,key5,key6,key7

# Security
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS
CORS_ORIGINS=["http://localhost:5173","https://careerpilot.ai"]

# Monitoring
SENTRY_DSN=https://xxxx@sentry.io/xxxx

# Rate Limits
RATE_LIMIT_ANALYZE=5
RATE_LIMIT_INTERVIEW=100
RATE_LIMIT_PLAN=10
```

---

## 12. Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction

# Copy application
COPY app/ ./app/
COPY migrations/ ./migrations/
COPY alembic.ini ./

# Run migrations and start
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
```

### docker-compose.yml (Development)

```yaml
version: "3.9"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./app:/app/app  # Hot reload
    command: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  worker:
    build: .
    env_file:
      - .env
    command: celery -A app.worker worker --loglevel=info
    depends_on:
      - redis

volumes:
  redis_data:
```

### Railway Deployment

```toml
# railway.toml
[build]
builder = "dockerfile"

[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "on_failure"

[[services]]
name = "api"
```

---

## Quick Reference

### API Routes Summary

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/v1/analyze` | Analyze resume | ✅ |
| POST | `/api/v1/plan` | Generate learning plan | ✅ |
| POST | `/api/v1/interview/start` | Start interview | ✅ |
| POST | `/api/v1/interview/message` | Send message | ✅ |
| POST | `/api/v1/interview/{id}/end` | End interview | ✅ |
| POST | `/api/v1/resume/optimize` | Optimize resume | ✅ |
| POST | `/api/v1/projects` | Generate project ideas | ✅ |
| POST | `/api/v1/auth/login` | Login | ❌ |
| POST | `/api/v1/auth/signup` | Register | ❌ |
| GET | `/health` | Health check | ❌ |

### Cost per Model

| Model | Use Case | Cost/1K tokens |
|-------|----------|----------------|
| gemini-2.0-flash-lite | Validation, extraction | $0.0001 |
| gemini-2.5-flash | Content generation | $0.001 |
| gemini-2.5-flash-thinking | Evaluation | $0.002 |
| gemini-2.5-pro | Complex reasoning | $0.005 |

---

*Document Version: 1.0*
*Created: February 7, 2026*
