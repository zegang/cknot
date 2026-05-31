import os
import logging
from typing import Optional, List, Any, Union
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from pydantic import ValidationError
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult
from cknot.utils.redis_client import get_redis_client
from cknot.schemas.llm_service import LLMService, LLMProvider

logger = logging.getLogger(__name__)

class TokenUsageTracker(BaseCallbackHandler):
    """Callback handler to record cumulative token usage in the service config."""
    def __init__(self, config: LLMService):
        self.config = config

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Update cumulative totals when an LLM request finishes."""
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            self.config.total_input_tokens += usage.get("prompt_tokens", 0)
            self.config.total_output_tokens += usage.get("completion_tokens", 0)
        # For models returning usage in metadata (standard for newer LangChain)
        elif hasattr(response, "usage_metadata") and response.usage_metadata:
             self.config.total_input_tokens += response.usage_metadata.get("input_tokens", 0)
             self.config.total_output_tokens += response.usage_metadata.get("output_tokens", 0)

class LLMManager:
    """
    Manages dynamic registration and retrieval of LLM service configurations
    from Redis, and instantiates LLM clients.
    """
    _instances: dict[str, 'LLMManager'] = {}

    def __new__(cls, redis_client: Union[Redis, AsyncRedis] = get_redis_client()):
        # Maintain separate singletons for sync and async clients to avoid IO conflicts
        client_type = "async" if isinstance(redis_client, AsyncRedis) else "sync"
        if client_type not in cls._instances:
            cls._instances[client_type] = super(LLMManager, cls).__new__(cls)
        return cls._instances[client_type]

    def __init__(self, redis_client: Union[Redis, AsyncRedis] = get_redis_client()):
        if getattr(self, "_initialized", False):
            return
        self._redis = redis_client
        self._llm_service_prefix = "llm_manager:"
        # Cache the config objects because they hold the _instance and the cumulative stats
        self._active_services: dict[str, LLMService] = {}
        # Detect if we are using an asynchronous client
        self._is_async = isinstance(redis_client, AsyncRedis)
        self._initialized = True

    def _get_redis_key(self, service_id: str) -> str:
        return f"{self._llm_service_prefix}{service_id}"

    def register_llm_service(self, config: LLMService):
        """Registers or updates an LLM service in Redis."""
        if self._is_async:
            raise RuntimeError("Cannot use sync register_llm_service with an async client. Use aregister_llm_service.")
        key = self._get_redis_key(config.id)
        # Redis hset mapping values must be bytes, strings, ints, or floats.
        # We convert booleans to strings to ensure compatibility with the Redis client.
        data = {
            k: (str(v) if isinstance(v, bool) else v)
            for k, v in config.model_dump(mode='json', exclude_none=True).items()
        }
        self._redis.hset(key, mapping=data)
        self._active_services[config.id] = config
        logger.info(f"LLM service '{config.id}' registered/updated.")

    def get_llm_service(self, service_id: str) -> Optional[LLMService]:
        """Retrieves an LLM service from Redis."""
        if service_id in self._active_services:
            return self._active_services[service_id]

        if self._is_async:
            raise RuntimeError("Cannot use sync get_llm_service with an async client. Use aget_llm_service.")

        key = self._get_redis_key(service_id)
        data = self._redis.hgetall(key)
        if not data:
            logger.warning(f"LLM service config '{service_id}' not found in Redis.")
            return None
        try:
            # key is already a string because decode_responses=True in redis_client
            decoded_data = {k: v for k, v in data.items()}
            return LLMService(**decoded_data)
        except ValidationError as e:
            logger.error(f"Invalid LLM service config for '{service_id}': {e}")
            return None

    def list_llm_services(self) -> List[LLMService]:
        """Lists all registered LLM services."""
        if self._is_async:
            raise RuntimeError("Cannot use sync list_llm_services with an async client. Use alist_llm_services.")
        services = []
        for key in self._redis.scan_iter(f"{self._llm_service_prefix}*"):
            # key is already a string because decode_responses=True in redis_client
            service_id = key.replace(self._llm_service_prefix, '')
            service = self.get_llm_service(service_id)
            if service:
                services.append(service)
        return services

    def get_llm_service_client(self, service_id: str) -> BaseChatModel:
        """Instantiates and returns an LLM client based on the LLM service ID."""
        if not self._active_services[service_id].is_enabled:
            raise ValueError(f"LLM service '{service_id}' is disabled.")

        # Return from cache if already instantiated
        if service_id in self._active_services and self._active_services[service_id]._svc_client:
            return self._active_services[service_id]._svc_client

        service = self.get_llm_service(service_id)
        if not service:
            raise ValueError(f"LLM service '{service_id}' not found or invalid.")

        try:
            # Create a tracker bound to this specific config object
            tracker = TokenUsageTracker(service)

            service._svc_client = ChatOpenAI(
                model=service.model_name,
                api_key=service.api_key,
                base_url=str(service.base_url) if service.base_url else None,
                callbacks=[tracker] # Attach the usage tracker
            )
            self._active_services[service_id] = service
            return self._active_services[service_id]._svc_client
        except Exception as e:
            logger.error(f"Failed to instantiate LLM client for '{service_id}': {e}")
            raise NotImplementedError(f"LLM provider '{service.provider}' is not yet supported.")

    def delete_llm_service(self, service_id: str):
        """Deletes an LLM service from Redis."""
        if self._is_async:
            raise RuntimeError("Cannot use sync delete_llm_service with an async client. Use adelete_llm_service.")
        key = self._get_redis_key(service_id)
        self._active_services.pop(service_id, None)
        if self._redis.delete(key):
            logger.info(f"LLM service '{service_id}' deleted.")
        else:
            logger.warning(f"Attempted to delete non-existent LLM service '{service_id}'.")

    # --- Asynchronous counterparts ---

    async def aregister_llm_service(self, config: LLMService):
        """Registers or updates an LLM service in Redis (Asynchronous)."""
        if not self._is_async:
            self.register_llm_service(config)
            return
        key = self._get_redis_key(config.id)
        self._active_services.pop(config.id, None)
        data = {
            k: (str(v) if isinstance(v, bool) else v)
            for k, v in config.model_dump(mode='json', exclude_none=True).items()
        }
        await self._redis.hset(key, mapping=data)
        logger.info(f"LLM service '{config.id}' registered/updated (async).")

    async def aget_llm_service(self, service_id: str) -> Optional[LLMService]:
        """Retrieves an LLM service from Redis (Asynchronous)."""
        if service_id in self._active_services:
            return self._active_services[service_id]
        if not self._is_async:
            return self.get_llm_service(service_id)

        key = self._get_redis_key(service_id)
        data = await self._redis.hgetall(key)
        if not data:
            return None
        try:
            return LLMService(**data)
        except ValidationError as e:
            logger.error(f"Invalid LLM service config for '{service_id}': {e}")
            return None

    async def alist_llm_services(self) -> List[LLMService]:
        """Lists all registered LLM services (Asynchronous)."""
        if not self._is_async:
            return self.list_llm_services()
        services = []
        async for key in self._redis.scan_iter(f"{self._llm_service_prefix}*"):
            service_id = key.replace(self._llm_service_prefix, '')
            service = await self.aget_llm_service(service_id)
            if service:
                services.append(service)
        return services

    async def adelete_llm_service(self, service_id: str):
        """Deletes an LLM service from Redis (Asynchronous)."""
        if not self._is_async:
            self.delete_llm_service(service_id)
            return
        key = self._get_redis_key(service_id)
        self._active_services.pop(service_id, None)
        if await self._redis.delete(key):
            logger.info(f"LLM service '{service_id}' deleted (async).")

    async def aget_llm_service_client(self, service_id: str) -> BaseChatModel:
        """Async version of getting a client to ensure config is fetched non-blockingly."""
        if service_id in self._active_services and self._active_services[service_id]._svc_client:
            return self._active_services[service_id]._svc_client
        
        service = await self.aget_llm_service(service_id)
        if not service:
            raise ValueError(f"LLM service '{service_id}' not found.")
        
        # Once config is fetched, instantiation is the same as sync
        return self.get_llm_service_client(service_id)

    async def validate_service(self, service_id: str) -> bool:
        """
        Validates the connectivity of an LLM service and updates its is_valid status.
        """
        config = await self.aget_llm_service(service_id)
        if not config:
            logger.error(f"Cannot validate: LLM service '{service_id}' not found.")
            return False

        if not config.is_enabled:
            logger.warning(f"LLM service '{service_id}' is disabled. Skipping validation.")
            return False

        try:
            llm = await self.aget_llm_service_client(service_id)
            # Perform a minimal connectivity check with a timeout
            await llm.ainvoke(
                [HumanMessage(content="connectivity check")], 
                config={"timeout": 10}
            )
            config.is_valid = True
            logger.info(f"LLM service '{service_id}' connectivity check passed.")
        except Exception as e:
            logger.warning(f"Connectivity check failed for LLM service '{service_id}': {e}")
            config.is_valid = False

        # Persist updated status to Redis and invalidate cache to ensure fresh state
        await self.aregister_llm_service(config)
        return config.is_valid

    def load_services_from_file(self, file_path: str):
        """
        Loads and registers multiple LLM services from a JSON or YAML file.
        Expects a list of service configurations.
        """
        if not os.path.exists(file_path):
            logger.error(f"LLM configuration file not found: {file_path}")
            return

        ext = os.path.splitext(file_path)[1].lower()
        try:
            with open(file_path, "r") as f:
                if ext in [".yaml", ".yml"]:
                    import yaml
                    configs_data = yaml.safe_load(f)
                else:
                    import json
                    configs_data = json.load(f)

            if not isinstance(configs_data, list):
                logger.error(f"Invalid format in {file_path}. Expected a list of LLM configurations.")
                return

            for config_dict in configs_data:
                config = LLMService(**config_dict)
                self.register_llm_service(config)
            
            logger.info(f"Successfully registered {len(configs_data)} LLM services from {file_path}")
        except Exception as e:
            logger.error(f"Error loading LLM services from {file_path}: {e}")