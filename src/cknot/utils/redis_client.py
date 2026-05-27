import logging
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from cknot.config.config import settings

logger = logging.getLogger(__name__)

class RedisClientSingleton:
    _instance = None
    _client = None

    def __new__(cls, socket_connect_timeout: int = 5):
        if cls._instance is None:
            cls._instance = super(RedisClientSingleton, cls).__new__(cls)
            cls._client = cls._instance._initialize_client(socket_connect_timeout)
        return cls._instance

    def _initialize_client(self, socket_connect_timeout: int):
        return Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            socket_connect_timeout=socket_connect_timeout,
            decode_responses=True 
        )

    def get_client(self):
        return self._client

class AsyncRedisClientSingleton:
    _instance = None
    _client = None

    def __new__(cls, socket_connect_timeout: int = 5):
        if cls._instance is None:
            cls._instance = super(AsyncRedisClientSingleton, cls).__new__(cls)
            cls._client = cls._instance._initialize_client(socket_connect_timeout)
        return cls._instance

    def _initialize_client(self, socket_connect_timeout: int):
        return AsyncRedis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            socket_connect_timeout=socket_connect_timeout,
            decode_responses=True 
        )

    def get_client(self):
        return self._client

def get_redis_client(socket_connect_timeout: int = 5):
    return RedisClientSingleton(socket_connect_timeout).get_client()

def get_async_redis_client(socket_connect_timeout: int = 5):
    return AsyncRedisClientSingleton(socket_connect_timeout).get_client()