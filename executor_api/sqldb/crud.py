from sqlalchemy.orm import Session
from . import models, schemas
from fastapi import HTTPException, status


# User CRUD operations
def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.user_id == user_id).first()


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()


def create_user(db: Session, user: schemas.UserCreate):
    # Check if user with same user_id, email or username already exists
    if get_user(db, user.user_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User with user_id {user.user_id} already exists"
        )
    
    if get_user_by_email(db, user.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User with email {user.email} already exists"
        )
    
    if get_user_by_username(db, user.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User with username {user.username} already exists"
        )
    
    # Create dict of user data
    user_data = user.dict()
    
    # Extract metadata if present
    metadata = user_data.pop("metadata", None)
    
    # Create user without metadata first
    db_user = models.User(**user_data)
    
    # Set metadata if provided
    if metadata:
        db_user.meta_data = metadata
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def update_user(db: Session, user_id: int, user_update: schemas.UserCreate):
    db_user = get_user(db, user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update basic fields
    if user_update.username != db_user.username:
        if get_user_by_username(db, user_update.username):
            raise HTTPException(status_code=400, detail="Username already in use")
        db_user.username = user_update.username
    
    if user_update.email != db_user.email:
        if get_user_by_email(db, user_update.email):
            raise HTTPException(status_code=400, detail="Email already in use")
        db_user.email = user_update.email
    
    # Update metadata if provided
    if hasattr(user_update, "metadata") and user_update.metadata is not None:
        db_user.meta_data = user_update.metadata
    
    db.commit()
    db.refresh(db_user)
    return db_user


# Project CRUD operations
def get_project(db: Session, project_id: str):
    return db.query(models.Project).filter(models.Project.id == project_id).first()


def get_projects(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Project).offset(skip).limit(limit).all()


def get_user_projects(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    user = get_user(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with user_id {user_id} not found"
        )
    
    return db.query(models.Project).filter(
        models.Project.owner_id == user.id
    ).offset(skip).limit(limit).all()


def create_project(db: Session, project: schemas.ProjectCreate, user_id: int):
    user = get_user(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with user_id {user_id} not found"
        )
    
    # Create dict of project data
    project_data = project.dict()
    
    # Extract config if present
    config = project_data.pop("config", None)
    
    # Create project with basic data
    db_project = models.Project(**project_data, owner_id=user.id)
    
    # Set config if provided
    if config:
        db_project.config = config
    
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project


def update_project(db: Session, project_id: str, project_update: schemas.ProjectCreate):
    db_project = get_project(db, project_id)
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Update basic fields
    db_project.name = project_update.name
    db_project.description = project_update.description
    
    # Update config if provided
    if hasattr(project_update, "config") and project_update.config is not None:
        db_project.config = project_update.config
    
    db.commit()
    db.refresh(db_project)
    return db_project

