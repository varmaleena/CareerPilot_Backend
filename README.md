# CareerPilot Backend

Multi-Agent LLM Career Assistant Backend built with FastAPI and LangGraph.

## Quick Start

1. **Set up environment**
```bash
cd server
cp .env.example .env
# Edit .env with your API keys
```

2. **Install dependencies**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Run the server**
```bash
uvicorn app.main:app --reload
```

4. **Open API docs**
Navigate to http://localhost:8000/docs

## Docker

```bash
docker-compose up --build
```

## Architecture

- **FastAPI** - High-performance async API
- **LangGraph** - Multi-agent orchestration
- **Gemini** - LLM provider with key rotation
- **Redis** - Caching and rate limiting
- **PostgreSQL** (Supabase) - Database

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/analyze` | Analyze resume |
| POST | `/api/v1/plan` | Generate learning plan |
| POST | `/api/v1/interview/start` | Start interview |
| POST | `/api/v1/interview/message` | Send message |
| POST | `/api/v1/interview/{id}/end` | End interview |
| GET | `/health` | Health check |
