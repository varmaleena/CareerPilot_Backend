import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport
from fastapi.testclient import TestClient

from app.main import app
from app.services.llm.gateway import LLMGateway


@pytest.fixture
def client():
    """Sync test client for FastAPI."""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Async test client for FastAPI."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def mock_llm():
    """Mock LLM gateway."""
    mock = AsyncMock(spec=LLMGateway)
    mock.generate.return_value = {
        "text": '{"content": "mock response", "key_points": ["point1"]}',
        "cached": False,
        "tokens": 100,
        "cost": 0.001,
        "model": "gemini-2.5-flash",
    }
    return mock


@pytest.fixture
def sample_resume_data():
    """Sample resume data for testing."""
    return {
        "name": "John Doe",
        "email": "john@example.com",
        "phone": "555-123-4567",
        "summary": "Experienced software engineer with 5 years of experience",
        "skills": ["Python", "JavaScript", "AWS", "Docker", "React"],
        "experience": [
            {
                "company": "Tech Corp",
                "title": "Senior Software Engineer",
                "start": "2020-01",
                "end": "Present",
                "highlights": [
                    "Led team of 5 engineers",
                    "Improved system performance by 40%",
                ],
            },
            {
                "company": "Startup Inc",
                "title": "Software Engineer",
                "start": "2018-06",
                "end": "2019-12",
                "highlights": [
                    "Built REST API serving 1M requests/day",
                ],
            },
        ],
        "education": [
            {
                "institution": "MIT",
                "degree": "Bachelor of Science",
                "field": "Computer Science",
                "year": "2018",
            },
        ],
        "certifications": ["AWS Solutions Architect"],
        "projects": [
            {
                "name": "Open Source Project",
                "description": "Contributed to major open source project",
                "technologies": ["Python", "Go"],
            },
        ],
    }


@pytest.fixture
def sample_job_description():
    """Sample job description for testing."""
    return """
    Senior Software Engineer at Google
    
    Requirements:
    - 5+ years of experience in software development
    - Proficiency in Python, Java, or Go
    - Experience with distributed systems
    - Strong problem-solving skills
    
    Preferred:
    - Experience with Kubernetes
    - Cloud platform experience (GCP, AWS, Azure)
    - Machine learning background
    """


@pytest.fixture
def mock_user():
    """Mock authenticated user."""
    return {
        "id": "user-123",
        "email": "test@example.com",
        "role": "user",
    }
