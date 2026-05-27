import os
from typing import Annotated
import logging
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from cknot.utils.redis_client import get_redis_client

logger = logging.getLogger(__name__)

@tool
def web_search(query: str, config: RunnableConfig):
    """Search the web for real-time information."""
    configurable = config.get("configurable", {})
    session_id = configurable.get("session_id", "unknown")
    user_id = configurable.get("user_id", "unknown")

    # 1. Rate Limiting Configuration
    if os.getenv("CHECKPOINTER_TYPE", "redis").lower() == "redis":
        redis_client = get_redis_client()
        limit = 5  # Max 5 requests
        window = 60  # Per 60 seconds
        key = f"rate_limit:web_search:{session_id}"

        # 2. Check and Increment
        current_usage = redis_client.get(key)
        if current_usage and int(current_usage) >= limit:
            return f"Error: Rate limit exceeded for session {session_id}. Please try again later."

        # Atomic increment and TTL set
        pipeline = redis_client.pipeline()
        pipeline.incr(key)
        if not current_usage:
            pipeline.expire(key, window)
        pipeline.execute()
    
    logger.info(f"[AUDIT LOG] User: {user_id} | Session: {session_id} | Action: web_search | Query: {query}")
    # Mocking a search result
    return f"Search results for '{query}': The project 'cknot' is a cutting-edge orchestration framework."
