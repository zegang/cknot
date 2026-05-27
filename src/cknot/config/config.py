import os
from typing import List

class Settings:
    PROJECT_NAME: str = "cknot Agentic API"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    CORS_ORIGINS: List[str] = ["*"]

    @property
    def SECRET_KEY(self) -> str:
        return os.getenv("SECRET_KEY", "super-secret-cknot-key")

    @property
    def REDIS_HOST(self) -> str:
        return os.getenv("REDIS_HOST", "localhost")

    @property
    def REDIS_PORT(self) -> int:
        return int(os.getenv("REDIS_PORT", "6379"))

    @property
    def CHECKPOINTER_TYPE(self) -> str:
        return os.getenv("CHECKPOINTER_TYPE", "redis").lower()

    @property
    def DEFAULT_LLM_SERVICE(self) -> str:
        return os.getenv("DEFAULT_LLM_SERVICE_ID", "default-llm")

settings = Settings()