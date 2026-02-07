# CareerPilot_Backend
This repository contains the backend service for an AI-driven platform built using Node.js, TypeScript, Fastify, and a multi-agent LLM architecture.
The system is designed with clear separation of concerns, scalability, and cost-aware AI orchestration as first-class principles.

📁 Repository Structure (Backend)
server/
├── src/
│   ├── index.ts
│   ├── app.ts
│
│   ├── routes/
│   ├── middleware/
│   ├── agents/
│   ├── services/
│   ├── db/
│   ├── migrations/
│   ├── jobs/
│   ├── utils/
│   ├── types/
│   └── tests/
│
├── Dockerfile
├── package.json
├── tsconfig.json
└── .env.example

 Entry Points
index.ts

Application entry point

Boots the Fastify server

Handles environment loading and graceful shutdown

app.ts

Fastify instance configuration

Registers plugins, middleware, routes, and hooks

Central place for HTTP-level concerns

 API Layer
routes/

Defines all HTTP API endpoints.
Each file represents a domain-specific route group.

routes/
├── analyze.ts       # Resume analysis endpoints
├── plan.ts          # Learning plan generation
├── interview.ts     # Interview session APIs
├── resume.ts        # Resume processing
├── projects.ts      # Project recommendation APIs
├── auth.ts          # Authentication-related routes
└── webhooks.ts      # External service webhooks


Responsibilities

Request/response handling

Input validation

Delegation to domain services

No business logic

 Middleware Layer
middleware/

Reusable Fastify middleware applied across routes.

middleware/
├── auth.ts          # JWT verification (Supabase)
├── rateLimit.ts    # Per-user rate limiting
├── validation.ts   # Zod-based schema validation
├── errorHandler.ts # Centralized error handling
└── costTracker.ts  # LLM cost tracking per request


Responsibilities

Cross-cutting concerns

Security, validation, observability

Enforced consistently across APIs

 Multi-Agent System
agents/

Core of the AI reasoning architecture.
Implements a multi-agent, workflow-driven system.

agents/core/ – Agent Infrastructure
core/
├── orchestrator.ts     # Workflow runner / state machine
├── decision-engine.ts # Agent decision logic
├── message-bus.ts     # Inter-agent communication
├── agent-factory.ts   # Agent instantiation
└── types.ts           # Agent contracts & interfaces


Responsibilities

Controls execution order

Routes data between agents

Manages workflow state

agents/masters/ – High-Cost Reasoning Agents
masters/
├── strategist.ts   # Planning & complex reasoning
├── evaluator.ts    # Quality & output assessment
└── resolver.ts     # Conflict & ambiguity resolution


Used only when deep reasoning is required.

agents/helpers/ – Lightweight Task Agents
helpers/
├── extractor.ts    # Structured data extraction
├── generator.ts    # Content generation
├── validator.ts    # Output validation
└── formatter.ts    # Response formatting


Optimized for low latency and cost efficiency.

agents/workflows/ – Workflow Definitions
workflows/
├── resume-analysis.ts
├── interview.ts
├── learning-plan.ts
└── project-ideas.ts


Each workflow:

Defines agent sequence

Controls branching and retries

Acts as the unit of AI execution

agents/prompts/ – Prompt Templates
prompts/
├── strategist/
├── evaluator/
└── helpers/


Version-controlled prompt files

Keeps prompts out of code

Enables safe iteration and optimization

 Service Layer
services/

Contains core business logic and infrastructure services.

services/llm/ – LLM Infrastructure
llm/
├── gateway.ts        # Unified LLM interface
├── gemini.ts         # Provider implementation
├── key-manager.ts   # API key rotation
├── model-router.ts  # Cost-aware model selection
├── token-counter.ts # Token usage tracking
└── prompt-builder.ts# Prompt optimization


Responsibilities

Abstracts LLM providers

Tracks usage & cost

Enables future provider swaps

services/cache/ – Caching Layer
cache/
├── redis.ts
├── semantic-cache.ts
└── session-store.ts


Used for:

Repeated LLM responses

Interview session persistence

Performance optimization

services/quota/ – Usage & Billing
quota/
├── quota-manager.ts
├── usage-tracker.ts
└── billing.ts


Controls:

Per-user limits

Token accounting

Cost calculations

services/domain/ – Business Logic
domain/
├── resume.service.ts
├── interview.service.ts
├── plan.service.ts
└── project.service.ts


Responsibilities

Core business rules

Orchestrates workflows

Independent of HTTP layer

 Database Layer
db/
db/
├── client.ts        # Supabase client setup
├── schema.sql       # Base schema
└── repositories/
    ├── user.repo.ts
    ├── analysis.repo.ts
    └── session.repo.ts


Pattern Used

Repository pattern

No raw queries outside repositories

Database-agnostic business logic

 Migrations
migrations/
migrations/
├── 001_initial.sql
└── 002_add_usage.sql


Versioned schema evolution

Safe production rollouts

 Background Jobs
jobs/
jobs/
├── queue.ts
├── usage-report.job.ts
└── cleanup.job.ts


Uses BullMQ for:

Async tasks

Scheduled cleanup

Usage reporting

🔹 Utilities & Shared Types
utils/
utils/
├── logger.ts
├── json-repair.ts
└── hash.ts

types/
types/
├── index.ts
├── api.ts
├── agents.ts
└── domain.ts


Ensures strong typing and consistency across layers.

 Testing
tests/
tests/
├── unit/
├── integration/
└── e2e/


Unit → agents & services

Integration → routes

E2E → full workflows
