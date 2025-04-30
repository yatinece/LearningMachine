from .logger import setup_logging , get_config
from .config import create_config
import json
# Create config file
create_config()

# Set up logging for the entire application FIRST
logger = setup_logging()
logger.info("Starting application")

conf = get_config()
logger.info("Event Data: %s", json.dumps(conf))

optuna_url = conf["OPTUNA_URL"]
logger.info(f"Starting application optuna_url is at {optuna_url} ")
#uvicorn mlexperiments.main:app --reload --host 0.0.0.0 --port 8000
#http://127.0.0.1:8000/
#http://localhost:8000/docs#/
#http://localhost:8000/redoc


from fastapi import FastAPI, Depends, HTTPException, status, Query, Body
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from . import models, schemas, crud
from .database import engine, get_db
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from . import models, schemas, crud
from .database import engine, get_db

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Create FastAPI app
app = FastAPI(
    title="Project Management API",
    description="API for managing users and projects with JSON storage support",
    version="1.0.0"
)
# ML Experiments Router with Logging and Exception Handling

ml_router = APIRouter(prefix="/ml", tags=["ML Experiments"])

# Create ML experiment endpoints
@ml_router.post("/projects/{project_id}/experiments/", response_model=schemas.MLExperiment, status_code=status.HTTP_201_CREATED)
def create_experiment(
    project_id: str, experiment: schemas.MLExperimentCreate, db: Session = Depends(get_db)
):
    """Create a new ML experiment for a specific project"""
    try:
        logger.info(f"Creating new experiment '{experiment.name}' for project: {project_id}")
        return crud.create_experiment(db=db, experiment=experiment, project_id=project_id)
    except Exception as e:
        logger.error(f"Error creating experiment for project {project_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@ml_router.get("/experiments/", response_model=List[schemas.MLExperiment])
def read_experiments(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get a list of all ML experiments"""
    try:
        logger.info(f"Fetching all experiments with skip={skip}, limit={limit}")
        experiments = crud.get_experiments(db, skip=skip, limit=limit)
        logger.debug(f"Retrieved {len(experiments)} experiments")
        return experiments
    except Exception as e:
        logger.error(f"Error retrieving experiments: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@ml_router.get("/projects/{project_id}/experiments/", response_model=List[schemas.MLExperiment])
def read_project_experiments(
    project_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get all ML experiments for a specific project"""
    try:
        logger.info(f"Fetching experiments for project {project_id} with skip={skip}, limit={limit}")
        experiments = crud.get_project_experiments(db, project_id=project_id, skip=skip, limit=limit)
        logger.debug(f"Retrieved {len(experiments)} experiments for project {project_id}")
        return experiments
    except Exception as e:
        logger.error(f"Error retrieving experiments for project {project_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@ml_router.get("/experiments/{experiment_id}", response_model=schemas.MLExperiment)
def read_experiment(experiment_id: str, db: Session = Depends(get_db)):
    """Get an ML experiment by id"""
    try:
        logger.info(f"Fetching experiment with ID: {experiment_id}")
        db_experiment = crud.get_experiment(db, experiment_id=experiment_id)
        if db_experiment is None:
            logger.warning(f"Experiment not found: {experiment_id}")
            raise HTTPException(status_code=404, detail="Experiment not found")
        return db_experiment
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving experiment {experiment_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@ml_router.put("/experiments/{experiment_id}", response_model=schemas.MLExperiment)
def update_experiment_data(experiment_id: str, experiment: schemas.MLExperimentCreate, db: Session = Depends(get_db)):
    """Update ML experiment data"""
    try:
        logger.info(f"Updating experiment with ID: {experiment_id}")
        updated_experiment = crud.update_experiment(db=db, experiment_id=experiment_id, experiment_update=experiment)
        if updated_experiment is None:
            logger.warning(f"Experiment not found for update: {experiment_id}")
            raise HTTPException(status_code=404, detail="Experiment not found")
        return updated_experiment
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating experiment {experiment_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@ml_router.post("/experiments/{experiment_id}/detect-features", response_model=List[Dict[str, str]])
def detect_features(experiment_id: str, db: Session = Depends(get_db)):
    """Detect and categorize features in the dataset"""
    try:
        logger.info(f"Detecting features for experiment: {experiment_id}")
        features = crud.detect_features_for_experiment(db=db, experiment_id=experiment_id)
        logger.debug(f"Detected {len(features)} features for experiment {experiment_id}")
        return features
    except Exception as e:
        logger.error(f"Error detecting features for experiment {experiment_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@ml_router.post("/experiments/{experiment_id}/train", response_model=Dict[str, Any])
def train_model(experiment_id: str, training_request: schemas.TrainingRequest, db: Session = Depends(get_db)):
    """Train an XGBoost model for the experiment with optimization"""
    try:
        logger.info(f"Training model for experiment: {experiment_id}")
        training_result = crud.train_model_for_experiment(
            db=db, 
            experiment_id=experiment_id, 
            training_request=training_request,
            study_url=optuna_url ,
        )
        logger.info(f"Model training completed for experiment {experiment_id}")
        return training_result
    except Exception as e:
        logger.error(f"Error training model for experiment {experiment_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


# New endpoint for uploading dataset from URL
@ml_router.post("/experiments/{experiment_id}/upload-dataset", response_model=schemas.MLExperiment)
def upload_dataset_from_url(
    experiment_id: str, 
    upload_request: schemas.DatasetUploadRequest, 
    db: Session = Depends(get_db)
):
    """Upload a dataset from a URL to an experiment"""
    try:
        logger.info(f"Uploading dataset from URL for experiment: {experiment_id}")
        
        # Check if experiment exists
        db_experiment = crud.get_experiment(db, experiment_id=experiment_id)
        if db_experiment is None:
            logger.warning(f"Experiment not found: {experiment_id}")
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        # Process the URL dataset
        updated_experiment = crud.process_dataset_from_url(
            db=db,
            experiment_id=experiment_id,
            url=upload_request.url,
            file_format=upload_request.file_format,
            options=upload_request.options
        )
        
        logger.info(f"Dataset successfully uploaded for experiment {experiment_id}")
        return updated_experiment
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading dataset for experiment {experiment_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e

# Include the ML router in the main app
app.include_router(ml_router)

# User endpoints
@app.post("/users/", response_model=schemas.User, status_code=status.HTTP_201_CREATED)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """Create a new user with optional metadata"""
    try:
        logger.info(f"Creating new user with username: {user.username}")
        return crud.create_user(db=db, user=user)
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e

@app.get("/users/", response_model=List[schemas.User])
def read_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get a list of users"""
    try:
        logger.info(f"Fetching users with skip={skip}, limit={limit}")
        users = crud.get_users(db, skip=skip, limit=limit)
        logger.debug(f"Retrieved {len(users)} users")
        return users
    except Exception as e:
        logger.error(f"Error retrieving users: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e



@app.get("/users/{user_id}", response_model=schemas.UserWithProjects)
def read_user(user_id: str, db: Session = Depends(get_db)):
    """Get a user by user_id, including their projects"""
    try:
        logger.info(f"Fetching user with ID: {user_id}")
        db_user = crud.get_user(db, user_id=user_id)
        if db_user is None:
            logger.warning(f"User not found: {user_id}")
            raise HTTPException(status_code=404, detail="User not found")
        return db_user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.put("/users/{user_id}", response_model=schemas.User)
def update_user_data(user_id: str, user: schemas.UserCreate, db: Session = Depends(get_db)):
    """Update user data including metadata"""
    try:
        logger.info(f"Updating user with ID: {user_id}")
        updated_user = crud.update_user(db=db, user_id=user_id, user_update=user)
        if updated_user is None:
            logger.warning(f"User not found for update: {user_id}")
            raise HTTPException(status_code=404, detail="User not found")
        return updated_user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.patch("/users/{user_id}/metadata", response_model=schemas.User)
def update_user_metadata(
    user_id: str, 
    metadata: Dict[str, Any] = Body(...), 
    db: Session = Depends(get_db)
):
    """Update only user metadata"""
    try:
        logger.info(f"Updating metadata for user: {user_id}")
        db_user = crud.get_user(db, user_id=user_id)
        if db_user is None:
            logger.warning(f"User not found for metadata update: {user_id}")
            raise HTTPException(status_code=404, detail="User not found")
        
        # Update metadata
        db_user.metadata = metadata
        db.commit()
        db.refresh(db_user)
        logger.debug(f"Metadata updated successfully for user: {user_id}")
        return db_user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating metadata for user {user_id}: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error") from e


# Project endpoints
@app.post("/users/{user_id}/projects/", response_model=schemas.Project, status_code=status.HTTP_201_CREATED)
def create_project_for_user(
    user_id: str, project: schemas.ProjectCreate, db: Session = Depends(get_db)
):
    """Create a new project for a specific user with optional JSON config"""
    try:
        logger.info(f"Creating new project '{project.name}' for user: {user_id}")
        return crud.create_project(db=db, project=project, user_id=user_id)
    except Exception as e:
        logger.error(f"Error creating project for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/users/{user_id}/projects/", response_model=List[schemas.Project])
def read_user_projects(
    user_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get all projects for a specific user"""
    try:
        logger.info(f"Fetching projects for user {user_id} with skip={skip}, limit={limit}")
        projects = crud.get_user_projects(db, user_id=user_id, skip=skip, limit=limit)
        logger.debug(f"Retrieved {len(projects)} projects for user {user_id}")
        return projects
    except Exception as e:
        logger.error(f"Error retrieving projects for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/projects/", response_model=List[schemas.Project])
def read_projects(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get a list of all projects"""
    try:
        logger.info(f"Fetching all projects with skip={skip}, limit={limit}")
        projects = crud.get_projects(db, skip=skip, limit=limit)
        logger.debug(f"Retrieved {len(projects)} projects")
        return projects
    except Exception as e:
        logger.error(f"Error retrieving projects: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/projects/{project_id}", response_model=schemas.Project)
def read_project(project_id: str, db: Session = Depends(get_db)):
    """Get a project by id"""
    try:
        logger.info(f"Fetching project with ID: {project_id}")
        db_project = crud.get_project(db, project_id=project_id)
        if db_project is None:
            logger.warning(f"Project not found: {project_id}")
            raise HTTPException(status_code=404, detail="Project not found")
        return db_project
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving project {project_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.put("/projects/{project_id}", response_model=schemas.Project)
def update_project_data(project_id: str, project: schemas.ProjectCreate, db: Session = Depends(get_db)):
    """Update project data including config"""
    try:
        logger.info(f"Updating project with ID: {project_id}")
        updated_project = crud.update_project(db=db, project_id=project_id, project_update=project)
        if updated_project is None:
            logger.warning(f"Project not found for update: {project_id}")
            raise HTTPException(status_code=404, detail="Project not found")
        return updated_project
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating project {project_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e

@app.patch("/projects/{project_id}/config", response_model=schemas.Project)
def update_project_config(
    project_id: str, 
    config: Dict[str, Any] = Body(...), 
    db: Session = Depends(get_db)
):
    """Update only project config JSON"""
    try:
        logger.info(f"Updating config for project: {project_id}")
        db_project = crud.get_project(db, project_id=project_id)
        if db_project is None:
            logger.warning(f"Project not found for config update: {project_id}")
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Update config
        db_project.config = config
        db.commit()
        db.refresh(db_project)
        logger.debug(f"Config updated successfully for project: {project_id}")
        return db_project
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating config for project {project_id}: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error") from e