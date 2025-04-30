from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class UserBase(BaseModel):
    user_id: int
    username: str
    email: EmailStr


class UserCreate(UserBase):
    meta_data: Optional[Dict[str, Any]] = None


class User(UserBase):
    id: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes  = True
        validate_by_name  = True


class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None


class ProjectCreate(ProjectBase):
    config: Optional[Dict[str, Any]] = None


class Project(ProjectBase):
    id: str
    owner_id: str
    config: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes  = True


class UserWithProjects(User):
    projects: List[Project] = []


class ProjectWithOwner(Project):
    owner: User