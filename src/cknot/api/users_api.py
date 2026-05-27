from fastapi import APIRouter, HTTPException, status, Depends, Path
from typing import List, Optional, Annotated
import logging
from cknot.utils.redis_client import get_async_redis_client
from cknot.utils.user_manager import UserManager
from cknot.api.auth import get_user_key, get_current_user, require_admin
from cknot.schemas.user import UserRegister, UserResponse, UserUpdate, UserPasswordChange

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["users"])

@router.post("/register", status_code=status.HTTP_201_CREATED, summary="Register a new user")
async def register(user: UserRegister):
    """Public endpoint to register a new user in the system."""
    mgr = UserManager(get_async_redis_client())
    if not await mgr.aregister_user(user):
        raise HTTPException(status_code=400, detail="Username already registered")
    
    return {"message": "User registered successfully"}

@router.get("/", response_model=List[UserResponse], summary="List all users")
async def list_users(_: Annotated[str, Depends(require_admin)]):
    """Retrieve a list of all registered users."""
    mgr = UserManager(get_async_redis_client())
    return await mgr.alist_users()

@router.get("/{username}", response_model=UserResponse, summary="Get user details")
async def get_user(
    username: str = Path(..., description="The username to retrieve"),
    current_user: Annotated[str, Depends(get_current_user)] = None
):
    """Get details for a specific user."""
    mgr = UserManager(get_async_redis_client())
    user_data = await mgr.aget_user(username)
    
    if not user_data or user_data.username != username:
        raise HTTPException(status_code=404, detail="User not found")
        
    return user_data

@router.patch("/{username}", response_model=UserResponse, summary="Update user")
async def update_user(
    payload: UserUpdate,
    username: str = Path(..., description="The username to update"),
    current_user: Annotated[str, Depends(get_current_user)] = None
):
    """Update user information (password or email). Users can only update themselves."""
    if username != current_user:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You can only update your own profile")

    mgr = UserManager(get_async_redis_client())
    
    if not await mgr.aupdate_user(username, payload):
        raise HTTPException(status_code=404, detail="User not found")

    return await get_user(username, current_user)

@router.post("/{username}/change-password", summary="Change user password")
async def change_password(
    payload: UserPasswordChange,
    username: str = Path(..., description="The username whose password is to be changed"),
    current_user: Annotated[str, Depends(get_current_user)] = None
):
    """Change password after verifying the current one. Users can only change their own password."""
    if username != current_user:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You can only change your own password")

    mgr = UserManager(get_async_redis_client())
    # verify current password
    if not await mgr.aauthenticate(username, payload.old_password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect current password")

    # update with new one
    if not await mgr.aupdate_user(username, UserUpdate(password=payload.new_password)):
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "Password updated successfully"}

@router.delete("/{username}", status_code=status.HTTP_204_NO_CONTENT, summary="Unregister user")
async def unregister(
    username: str = Path(..., description="The username to delete"),
    current_user: Annotated[str, Depends(get_current_user)] = None
):
    """Remove a user from the system. Users can only delete themselves."""
    if username != current_user:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You can only delete your own profile")

    mgr = UserManager(get_async_redis_client())
    if not await mgr.adelete_user(username):
        raise HTTPException(status_code=404, detail="User not found")