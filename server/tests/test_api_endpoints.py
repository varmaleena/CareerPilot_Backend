import pytest
from unittest.mock import patch, AsyncMock


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check(self, client):
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert "version" in response.json()


class TestAnalyzeEndpoint:
    """Tests for /analyze endpoint."""
    
    @pytest.mark.asyncio
    async def test_analyze_requires_auth(self, async_client):
        """Test analyze endpoint requires authentication."""
        response = await async_client.post(
            "/api/v1/analyze",
            json={
                "resume_text": "John Doe, Software Engineer...",
                "target_role": "Senior SWE",
            },
        )
        
        # Should fail without auth
        assert response.status_code in [401, 403]
    
    @pytest.mark.skip(reason="Requires Redis connection")
    def test_analyze_request_validation(self, client):
        """Test request validation."""
        # Resume too short
        response = client.post(
            "/api/v1/analyze",
            json={
                "resume_text": "short",  # Less than 100 chars
                "target_role": "SWE",
            },
            headers={"Authorization": "Bearer fake-token"},
        )
        
        assert response.status_code in [401, 422]  # Validation or auth error


class TestInterviewEndpoint:
    """Tests for /interview endpoints."""
    
    @pytest.mark.asyncio
    async def test_interview_start_requires_auth(self, async_client):
        """Test interview start requires authentication."""
        response = await async_client.post(
            "/api/v1/interview/start",
            json={
                "interview_type": "technical",
                "difficulty": "medium",
                "duration_minutes": 30,
            },
        )
        
        assert response.status_code in [401, 403]


class TestPlanEndpoint:
    """Tests for /plan endpoint."""
    
    @pytest.mark.skip(reason="Requires Redis connection")
    @pytest.mark.asyncio
    async def test_plan_requires_auth(self, async_client):
        """Test plan endpoint requires authentication."""
        response = await async_client.post(
            "/api/v1/plan",
            json={
                "target_role": "Data Scientist",
                "current_skills": ["Python"],
                "timeline_weeks": 12,
            },
        )
        
        assert response.status_code in [401, 403]


class TestOptimizeEndpoint:
    """Tests for /optimize endpoint."""
    
    @pytest.mark.asyncio
    async def test_optimize_requires_auth(self, async_client):
        """Test optimize endpoint requires authentication."""
        response = await async_client.post(
            "/api/v1/optimize",
            json={
                "resume_text": "A" * 150,  # Minimum length
                "job_description": "B" * 100,
                "target_role": "SWE",
            },
        )
        
        assert response.status_code in [401, 403]


class TestExportEndpoint:
    """Tests for /export endpoint."""
    
    @pytest.mark.asyncio
    async def test_export_requires_auth(self, async_client, sample_resume_data):
        """Test export endpoint requires authentication."""
        response = await async_client.post(
            "/api/v1/export/resume",
            json={
                "resume_data": sample_resume_data,
                "format": "latex",
            },
        )
        
        assert response.status_code in [401, 403]


class TestJobsEndpoint:
    """Tests for /jobs endpoints."""
    
    @pytest.mark.asyncio
    async def test_job_search_requires_auth(self, async_client):
        """Test job search requires authentication."""
        response = await async_client.post(
            "/api/v1/jobs/search",
            json={
                "query": "Software Engineer",
                "location": "Remote",
            },
        )
        
        assert response.status_code in [401, 403]
    
    @pytest.mark.asyncio
    async def test_market_insights_requires_auth(self, async_client):
        """Test market insights requires authentication."""
        response = await async_client.get("/api/v1/jobs/market/software-engineer")
        
        assert response.status_code in [401, 403]
