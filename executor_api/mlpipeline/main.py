from fastapi import FastAPI, Depends, HTTPException, status, Query, Body
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from . import models, schemas, crud
from .database import engine, get_db
from .ml_routes import router as ml_router
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api")

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Create FastAPI app
app = FastAPI(
    title="Project Management API with ML Capabilities",
    description="API for managing users and projects with ML experiment support",
    version="1.0.0"
)

# Include ML router
app.include_router(ml_router)


# User endpoints
@app.post("/users/", response_model=schemas.User, status_code=status.HTTP_201_CREATED)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """Create a new user with optional metadata"""
    return crud.create_user(db=db, user=user)


@app.get("/users/", response_model=List[schemas.User])
def read_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get a list of users"""
    users = crud.get_users(db, skip=skip, limit=limit)
    return users


@app.get("/users/{user_id}", response_model=schemas.UserWithProjects)
def read_user(user_id: int, db: Session = Depends(get_db)):
    """Get a user by user_id, including their projects"""
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@app.put("/users/{user_id}", response_model=schemas.User)
def update_user_data(user_id: int, user: schemas.UserCreate, db: Session = Depends(get_db)):
    """Update user data including metadata"""
    return crud.update_user(db=db, user_id=user_id, user_update=user)


@app.patch("/users/{user_id}/metadata", response_model=schemas.User)
def update_user_metadata(
    user_id: int, 
    metadata: Dict[str, Any] = Body(...), 
    db: Session = Depends(get_db)
):
    """Update only user metadata"""
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update metadata
    db_user.meta_data = metadata
    db.commit()
    db.refresh(db_user)
    return db_user


# Project endpoints
@app.post("/users/{user_id}/projects/", response_model=schemas.Project, status_code=status.HTTP_201_CREATED)
def create_project_for_user(
    user_id: int, project: schemas.ProjectCreate, db: Session = Depends(get_db)
):
    """Create a new project for a specific user with optional JSON config"""
    return crud.create_project(db=db, project=project, user_id=user_id)


@app.get("/users/{user_id}/projects/", response_model=List[schemas.Project])
def read_user_projects(
    user_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get all projects for a specific user"""
    projects = crud.get_user_projects(db, user_id=user_id, skip=skip, limit=limit)
    return projects


@app.get("/projects/", response_model=List[schemas.Project])
def read_projects(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get a list of all projects"""
    projects = crud.get_projects(db, skip=skip, limit=limit)
    return projects


@app.get("/projects/{project_id}", response_model=schemas.Project)
def read_project(project_id: str, db: Session = Depends(get_db)):
    """Get a project by id"""
    db_project = crud.get_project(db, project_id=project_id)
    if db_project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return db_project


@app.put("/projects/{project_id}", response_model=schemas.Project)
def update_project_data(project_id: str, project: schemas.ProjectCreate, db: Session = Depends(get_db)):
    """Update project data including config"""
    return crud.update_project(db=db, project_id=project_id, project_update=project)


@app.patch("/projects/{project_id}/config", response_model=schemas.Project)
def update_project_config(
    project_id: str, 
    config: Dict[str, Any] = Body(...), 
    db: Session = Depends(get_db)
):
    """Update only project config JSON"""
    db_project = crud.get_project(db, project_id=project_id)
    if db_project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Update config
    db_project.config = config
    db.commit()
    db.refresh(db_project)
    return db_project