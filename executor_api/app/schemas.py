from pydantic import BaseModel, EmailStr, Field, UUID4
from typing import Optional, List
from datetime import datetime


class UserBase(BaseModel):
    user_id: int
    username: str
    email: EmailStr


class UserCreate(UserBase):
    pass


class User(UserBase):
    id: UUID4
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True


class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None


class ProjectCreate(ProjectBase):
    pass


class Project(ProjectBase):
    id: UUID4
    owner_id: UUID4
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True


class UserWithProjects(User):
    projects: List[Project] = []


class ProjectWithOwner(Project):
    owner: User
