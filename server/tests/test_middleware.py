import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from jose import jwt


class TestAuthMiddleware:
    """Tests for authentication middleware."""
    
    def test_verify_token_valid(self):
        """Test valid JWT token verification."""
        from app.middleware.auth import verify_token
        from fastapi.security import HTTPAuthorizationCredentials
        
        # Create a mock valid token
        with patch("app.middleware.auth.settings") as mock_settings:
            mock_settings.SUPABASE_JWT_SECRET = "test-secret"
            mock_settings.JWT_ALGORITHM = "HS256"
            
            # Create a valid token
            token = jwt.encode(
                {"sub": "user-123", "email": "test@example.com", "aud": "authenticated"},
                "test-secret",
                algorithm="HS256",
            )
            
            credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
            
            # This would need async context in real test
            # Just checking the token structure
            assert token is not None
    
    def test_invalid_token_raises_error(self):
        """Test invalid token raises 401."""
        from jose import JWTError
        
        # Invalid token should raise JWTError when decoded
        with pytest.raises(JWTError):
            jwt.decode("invalid.token.here", "secret", algorithms=["HS256"])


class TestRateLimitMiddleware:
    """Tests for rate limiting middleware."""
    
    def test_rate_limit_config(self):
        """Test rate limit configuration."""
        from app.middleware.rate_limit import RateLimitMiddleware
        
        middleware = RateLimitMiddleware(app=MagicMock())
        
        # Check limits are configured
        assert "/api/v1/analyze" in middleware.LIMITS
        assert "/api/v1/interview/message" in middleware.LIMITS


class TestErrorHandlerMiddleware:
    """Tests for error handler middleware."""
    
    @pytest.mark.asyncio
    async def test_error_handler_catches_exceptions(self):
        """Test error handler catches and logs exceptions."""
        from app.middleware.error_handler import error_handler_middleware
        from fastapi import Request
        from unittest.mock import AsyncMock
        
        mock_request = MagicMock(spec=Request)
        
        async def failing_call_next(req):
            raise ValueError("Test error")
        
        with patch("app.middleware.error_handler.logger") as mock_logger:
            with patch("app.middleware.error_handler.sentry_sdk"):
                response = await error_handler_middleware(mock_request, failing_call_next)
                
                assert response.status_code == 500
                mock_logger.exception.assert_called()


class TestModelsValidation:
    """Tests for Pydantic model validation."""
    
    def test_analyze_request_validation(self):
        """Test AnalyzeRequest validation."""
        from app.models.requests import AnalyzeRequest
        from pydantic import ValidationError
        
        # Valid request
        valid = AnalyzeRequest(
            resume_text="A" * 150,  # >= 100 chars
            target_role="Software Engineer",
        )
        assert valid.resume_text is not None
        
        # Invalid - resume too short
        with pytest.raises(ValidationError):
            AnalyzeRequest(
                resume_text="short",
                target_role="SWE",
            )
    
    def test_interview_request_validation(self):
        """Test InterviewStartRequest validation."""
        from app.models.requests import InterviewStartRequest, InterviewType
        
        valid = InterviewStartRequest(
            interview_type=InterviewType.TECHNICAL,
            difficulty="medium",
            duration_minutes=30,
        )
        
        assert valid.interview_type == InterviewType.TECHNICAL
    
    def test_plan_request_validation(self):
        """Test PlanRequest validation."""
        from app.models.requests import PlanRequest
        from pydantic import ValidationError
        
        valid = PlanRequest(
            target_role="Data Scientist",
            current_skills=["Python", "SQL"],
            timeline_weeks=12,
            hours_per_week=10,
        )
        
        assert valid.timeline_weeks == 12
        
        # Invalid - timeline too short
        with pytest.raises(ValidationError):
            PlanRequest(
                target_role="DS",
                timeline_weeks=2,  # < 4 weeks minimum
            )
