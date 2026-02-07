# CareerPilot Backend

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-purple.svg)
![Tests](https://img.shields.io/badge/Tests-58%20passed-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**AI-powered career development platform with multi-agent workflows**

[Features](#features) • [Quick Start](#quick-start) • [API Reference](#api-reference) • [Architecture](#architecture)

</div>

---

## Features

🎯 **Resume Analysis** — Deep analysis with skill gap identification and ATS scoring  
📝 **Resume Optimization** — Keyword optimization and bullet rewriting for target roles  
📚 **Learning Plans** — Personalized skill development roadmaps with resources  
🎤 **Mock Interviews** — Real-time AI interviews with behavioral/technical modes  
💼 **Job Matching** — Find jobs that match your skills and experience  
📊 **Market Insights** — Salary benchmarks and industry trend analysis  
📄 **PDF/LaTeX Export** — Professional resume export in multiple formats

## Quick Start

### Prerequisites

- Python 3.11+
- Redis (for caching and rate limiting)
- PostgreSQL (for data persistence)

### Installation

```bash
# Clone and navigate
cd CareerPilot/server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials
```

### Environment Variables

```env
# Required
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/careerdb
REDIS_URL=redis://localhost:6379
GEMINI_API_KEYS=your-api-key-1,your-api-key-2

# Supabase Auth
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_JWT_SECRET=your-jwt-secret

# Optional
SENTRY_DSN=https://xxx@sentry.io/xxx
```

### Run Development Server

```bash
# Start the API
uvicorn app.main:app --reload --port 8000

# Open API docs
open http://localhost:8000/docs
```

### Run Tests

```bash
pytest tests/ -v
# Output: ================== 58 passed, 2 skipped ===================
```

## API Reference

### Resume Analysis
```http
POST /api/v1/analyze
Content-Type: application/json
Authorization: Bearer <token>

{
  "resume_text": "Your resume content...",
  "target_role": "Senior Software Engineer",
  "target_company": "Google"
}
```

### Resume Optimization
```http
POST /api/v1/optimize

{
  "resume_text": "Your resume...",
  "job_description": "Job posting...",
  "target_role": "Software Engineer",
  "optimization_focus": ["ats", "bullets", "keywords"]
}
```

### Learning Plan
```http
POST /api/v1/plan

{
  "target_role": "Data Scientist",
  "current_skills": ["Python", "SQL"],
  "timeline_weeks": 12,
  "hours_per_week": 10
}
```

### Mock Interview
```http
POST /api/v1/interview/start

{
  "interview_type": "technical",  // behavioral, technical, system_design
  "difficulty": "medium",
  "duration_minutes": 30
}
```

### Job Search & Market
```http
POST /api/v1/jobs/search
GET /api/v1/jobs/market/{role}
GET /api/v1/jobs/market/salary/{role}?experience_years=5
```

### Export
```http
POST /api/v1/export/resume

{
  "resume_data": { ... },
  "format": "pdf"  // or "latex"
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Gateway                          │
├─────────────────────────────────────────────────────────────────┤
│  Auth Middleware  │  Rate Limiter  │  Error Handler  │  CORS   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    LangGraph Workflows                    │   │
│  │  ┌──────────┐  ┌───────────┐  ┌────────┐  ┌──────────┐   │   │
│  │  │ Resume   │  │ Interview │  │ Learn  │  │ Optimize │   │   │
│  │  │ Analysis │  │ Workflow  │  │ Plan   │  │ Resume   │   │   │
│  │  └──────────┘  └───────────┘  └────────┘  └──────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌────────────────────┐  ┌───────────────────────────────────┐  │
│  │   Master Agents    │  │         Helper Agents             │  │
│  │  ├─ Strategist     │  │  ├─ Extractor  ├─ Generator      │  │
│  │  ├─ Evaluator      │  │  ├─ Validator  └─ Formatter      │  │
│  │  └─ Resolver       │  │                                   │  │
│  └────────────────────┘  └───────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                      LLM Gateway                          │   │
│  │  Key Rotation  │  Model Router  │  Semantic Cache        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│     PostgreSQL      │      Redis       │     Gemini API        │
└─────────────────────────────────────────────────────────────────┘
```

### Agent Hierarchy

| Agent | Role | Model Tier |
|-------|------|------------|
| **Strategist** | Plans execution, routes requests | Pro |
| **Evaluator** | Quality assessment, scoring | Flash-Thinking |
| **Resolver** | Conflict resolution, fallbacks | Pro |
| **Extractor** | Data extraction from documents | Lite |
| **Generator** | Content generation | Flash |
| **Validator** | Input validation | Lite |
| **Formatter** | Output formatting | Lite |

### Cost Optimization

- **Model tiering**: Uses cheapest model capable of each task
- **Semantic caching**: Caches similar prompts to reduce API calls
- **Key rotation**: Distributes load across multiple API keys
- **Rate limiting**: Protects against abuse

## Project Structure

```
server/
├── app/
│   ├── agents/
│   │   ├── core/           # Base agent classes
│   │   ├── masters/        # Strategist, Evaluator, Resolver
│   │   ├── helpers/        # Extractor, Generator, Validator, Formatter
│   │   └── workflows/      # LangGraph workflow definitions
│   ├── api/
│   │   └── v1/             # API endpoints
│   ├── services/
│   │   ├── llm/            # LLM Gateway, Key Manager, Model Router
│   │   ├── cache/          # Redis client, Semantic Cache
│   │   ├── export/         # PDF/LaTeX generation
│   │   ├── jobs/           # Job fetching service
│   │   └── market/         # Market insights service
│   ├── middleware/         # Auth, Rate limit, Error handler
│   ├── models/             # Pydantic models
│   └── main.py             # FastAPI application
├── tests/                  # Test suite (58 tests)
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Docker Deployment

```bash
# Build and run
docker-compose up --build

# Or build image only
docker build -t careerpilot-api .
docker run -p 8000:8000 --env-file .env careerpilot-api
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Built with ❤️ using FastAPI, LangGraph, and Gemini
</div>
