from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

class User(BaseModel):
    """Core user structure used across the system."""
    username: str = Field(..., description="Unique username")
    email: Optional[str] = Field(None, description="User email address")
    is_admin: bool = Field(False, description="Administrative status")

    model_config = ConfigDict(from_attributes=True)

class UserRegister(User):
    """Schema for user registration."""
    password: str = Field(..., description="Plain-text password")

class UserResponse(User):
    """Schema for user information response."""
    pass

class UserUpdate(BaseModel):
    """Schema for partial user updates."""
    password: Optional[str] = Field(None, description="New password")
    email: Optional[str] = Field(None, description="New email address")
    is_admin: Optional[bool] = Field(None, description="Administrative status update")

class UserPasswordChange(BaseModel):
    """Schema for password change requests."""
    old_password: str = Field(..., description="Current password")
    new_password: str = Field(..., description="New password")