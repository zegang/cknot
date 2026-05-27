import bcrypt
import logging
from typing import List, Optional, Union, Dict
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from cknot.schemas.user import UserRegister, UserResponse, UserUpdate

logger = logging.getLogger(__name__)

class UserManager:
    """
    Manages user profiles, authentication, and persistence in Redis.
    Supports both synchronous and asynchronous Redis clients.
    """
    def __init__(self, redis_client: Union[Redis, AsyncRedis]):
        self._redis = redis_client
        self._prefix = "user_profile:"
        self._is_async = isinstance(redis_client, AsyncRedis)

    def _get_user_key(self, username: str) -> str:
        return f"{self._prefix}{username}"

    def _hash_password(self, password: str) -> str:
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def _check_password(self, password: str, hashed: str) -> bool:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def _parse_user_data(self, data: dict) -> UserResponse:
        """Standardizes user data retrieved from Redis."""
        if "is_admin" in data:
            data["is_admin"] = str(data["is_admin"]) == "True"
        return UserResponse.model_validate(data)

    # --- Synchronous Methods ---

    def register_user(self, user: UserRegister) -> bool:
        if self._is_async:
            raise RuntimeError("Use aregister_user with an async Redis client.")
        key = self._get_user_key(user.username)
        if self._redis.exists(key):
            return False
        self._redis.hset(key, mapping={
            "username": user.username,
            "password": self._hash_password(user.password),
            "email": user.email or "",
            "is_admin": str(user.is_admin)
        })
        return True

    def authenticate(self, username: str, password: str) -> Optional[UserResponse]:
        if self._is_async:
            raise RuntimeError("Use aauthenticate with an async Redis client.")
        data = self._redis.hgetall(self._get_user_key(username))
        if not data:
            return None
        if self._check_password(password, data.get("password", "")):
            return self._parse_user_data(data)
        return None

    def list_users(self) -> List[UserResponse]:
        if self._is_async:
            raise RuntimeError("Use alist_users with an async Redis client.")
        users = []
        for key in self._redis.scan_iter(f"{self._prefix}*"):
            data = self._redis.hgetall(key)
            if data:
                users.append(self._parse_user_data(data))
        return users

    def delete_user(self, username: str) -> bool:
        if self._is_async:
            raise RuntimeError("Use adelete_user with an async Redis client.")
        return bool(self._redis.delete(self._get_user_key(username)))

    def update_user(self, username: str, updates: UserUpdate) -> bool:
        if self._is_async:
            raise RuntimeError("Use aupdate_user with an async Redis client.")
        key = self._get_user_key(username)
        if not self._redis.exists(key):
            return False
        update_data = updates.model_dump(exclude_unset=True)
        if "password" in update_data:
            update_data["password"] = self._hash_password(update_data["password"])
        if "is_admin" in update_data:
            update_data["is_admin"] = str(update_data["is_admin"])
        self._redis.hset(key, mapping=update_data)
        return True

    # --- Asynchronous Methods ---

    async def aregister_user(self, user: UserRegister) -> bool:
        if not self._is_async:
            return self.register_user(user)
        key = self._get_user_key(user.username)
        if await self._redis.exists(key):
            return False
        await self._redis.hset(key, mapping={
            "username": user.username,
            "password": self._hash_password(user.password),
            "email": user.email or "",
            "is_admin": str(user.is_admin)
        })
        return True

    async def aauthenticate(self, username: str, password: str) -> Optional[UserResponse]:
        if not self._is_async:
            return self.authenticate(username, password)
        data = await self._redis.hgetall(self._get_user_key(username))
        if not data:
            return None
        if self._check_password(password, data.get("password", "")):
            return self._parse_user_data(data)
        return None

    async def alist_users(self) -> List[UserResponse]:
        if not self._is_async:
            return self.list_users()
        users = []
        async for key in self._redis.scan_iter(f"{self._prefix}*"):
            data = await self._redis.hgetall(key)
            if data:
                users.append(self._parse_user_data(data))
        return users

    async def aget_user(self, username: str) -> Optional[UserResponse]:
        if not self._is_async:
            data = self._redis.hgetall(self._get_user_key(username))
        else:
            data = await self._redis.hgetall(self._get_user_key(username))
        return self._parse_user_data(data) if data else None

    async def adelete_user(self, username: str) -> bool:
        if not self._is_async:
            return self.delete_user(username)
        return bool(await self._redis.delete(self._get_user_key(username)))

    async def aupdate_user(self, username: str, updates: UserUpdate) -> bool:
        if not self._is_async:
            return self.update_user(username, updates)
        key = self._get_user_key(username)
        if not await self._redis.exists(key):
            return False
        update_data = updates.model_dump(exclude_unset=True)
        if "password" in update_data:
            update_data["password"] = self._hash_password(update_data["password"])
        if "is_admin" in update_data:
            update_data["is_admin"] = str(update_data["is_admin"])
        await self._redis.hset(key, mapping=update_data)
        return True