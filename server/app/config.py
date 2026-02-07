from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    # App
    APP_ENV: str = "development"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/careerdb"
    DATABASE_POOL_SIZE: int = 10
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # Supabase
    SUPABASE_URL: str = "https://test.supabase.co"
    SUPABASE_KEY: str = "test-key"
    SUPABASE_JWT_SECRET: str = "test-jwt-secret-key-for-development"
    
    # Gemini API Keys
    GEMINI_API_KEYS: str = "test-api-key"  # Comma-separated or list handling
    
    # Security
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    # Monitoring
    SENTRY_DSN: str | None = None
    
    # Rate Limits
    RATE_LIMIT_ANALYZE: int = 5
    RATE_LIMIT_INTERVIEW: int = 100
    RATE_LIMIT_PLAN: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        
    @property
    def gemini_keys_list(self) -> list[str]:
        if "," in self.GEMINI_API_KEYS:
            return [k.strip() for k in self.GEMINI_API_KEYS.split(",")]
        return [self.GEMINI_API_KEYS]

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
