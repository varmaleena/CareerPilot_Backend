import hashlib
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
