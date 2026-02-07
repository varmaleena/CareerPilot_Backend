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
