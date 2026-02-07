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
