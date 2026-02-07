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
    
    async def delete(self, key: str) -> None:
        client = await self.connect()
        await client.delete(key)


redis_client = RedisClient()


async def get_redis() -> RedisClient:
    return redis_client
