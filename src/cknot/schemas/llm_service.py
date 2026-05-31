from enum import Enum
from pydantic import BaseModel, Field, HttpUrl, PrivateAttr
from typing import Optional, Any

class LLMSelectPolicy(str, Enum):
    """Enum for LLM service selection policies."""
    FIRST = "first"
    RANDOM = "random"
    WEIGHTED = "weighted"
    CAPABILITY = "capability"

class LLMProvider(str, Enum):
    """Enum for supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAM = "ollama"
    VLLM = "vllm"
    SGLANG = "sglang"
    GOOGLE = "google"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"
    AI21 = "ai21"

class LLMService(BaseModel):
    """
    Schema for configuring an LLM service.
    """
    id: str = Field(..., description="Unique identifier for this LLM service (e.g., 'default_vllm', 'openai_gpt4').")
    name: str = Field(..., description="Human-readable name for the LLM service.")
    provider: LLMProvider = Field(..., description="The LLM provider (e.g., 'openai', 'anthropic', 'vllm').")
    model_name: str = Field(..., description="The specific model name (e.g., 'gpt-4o', 'vicuna-7b-v1.5').")
    api_key: Optional[str] = Field(None, description="API key for the LLM service. Will be redacted in logs.")
    base_url: Optional[HttpUrl] = Field(None, description="Base URL for custom API endpoints (e.g., vLLM).")
    is_enabled: bool = Field(True, description="Whether this LLM service is enabled.")
    is_valid: bool = Field(True, description="Whether the LLM service has been validated and is functional.")
    total_input_tokens: int = Field(0, description="Total input tokens consumed by this service in the current session.")
    total_output_tokens: int = Field(0, description="Total output tokens consumed by this service in the current session.")
    # Private attribute to hold the live LLM instance in memory without serializing it to Redis
    _svc_client: Optional[Any] = PrivateAttr(default=None)
    # Add other provider-specific fields as needed

    class Config:
        # Allow extra fields for future flexibility without breaking existing configs
        extra = "allow"
