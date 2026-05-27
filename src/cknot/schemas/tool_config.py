from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any

class ToolConfig(BaseModel):
    """Schema for tool configuration and state."""
    id: str = Field(..., description="Unique identifier for the tool.")
    name: str = Field(..., description="Human-readable name.")
    description: str = Field(..., description="What the tool does.")
    is_enabled: bool = Field(True, description="Whether the tool is active in the graph.")
    usage_count: int = Field(0, description="How many times the tool has been called.")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "web_search",
                    "name": "Web Search",
                    "description": "Performs a search on the internet to retrieve up-to-date information.",
                    "is_enabled": True,
                    "usage_count": 15
                }
            ]
        }
    )