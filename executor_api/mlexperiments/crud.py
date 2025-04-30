import logging
logger = logging.getLogger(__name__)
from sqlalchemy.orm import Session
from . import models, schemas
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from .ml_services import DataLoader, FeatureProcessor, ModelTrainer
import json
import requests
import pandas as pd
import io
import os
import json
# User CRUD operations
def get_user(db: Session, user_id: str):
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


def update_user(db: Session, user_id: str, user_update: schemas.UserCreate):
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


def get_user_projects(db: Session, user_id: str, skip: int = 0, limit: int = 100):
    user = get_user(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with user_id {user_id} not found"
        )
    
    return db.query(models.Project).filter(
        models.Project.owner_id == user.id
    ).offset(skip).limit(limit).all()


def create_project(db: Session, project: schemas.ProjectCreate, user_id: str):
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
    db_project = models.Project(**project_data, owner_id=user.user_id)
    
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



# ML Experiment CRUD operations
def get_experiment(db: Session, experiment_id: str):
    return db.query(models.MLExperiment).filter(models.MLExperiment.id == experiment_id).first()

def get_experiments(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.MLExperiment).offset(skip).limit(limit).all()

def get_project_experiments(db: Session, project_id: str, skip: int = 0, limit: int = 100):
    return db.query(models.MLExperiment).filter(
        models.MLExperiment.project_id == project_id
    ).offset(skip).limit(limit).all()

def create_experiment(db: Session, experiment: schemas.MLExperimentCreate, project_id: str):
    # Check if project exists
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id {project_id} not found"
        )
    
    # Create dict of experiment data
    experiment_data = experiment.dict()
    
    # Extract configs if present
    features_config = experiment_data.pop("features_config", None)
    model_configuration = experiment_data.pop("model_configuration", None)
    hyperparams = experiment_data.pop("hyperparams", None)
    
    # Create experiment with basic data
    db_experiment = models.MLExperiment(**experiment_data, project_id=project_id)
    
    # Set configs if provided
    if features_config:
        db_experiment.features_config = features_config
    
    if model_configuration:
        db_experiment.model_configuration = model_configuration
        
    if hyperparams:
        db_experiment.hyperparams = hyperparams
    
    db.add(db_experiment)
    db.commit()
    db.refresh(db_experiment)
    return db_experiment

def update_experiment(db: Session, experiment_id: str, experiment_update: schemas.MLExperimentCreate):
    db_experiment = get_experiment(db, experiment_id)
    if not db_experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Update basic fields
    db_experiment.name = experiment_update.name
    db_experiment.description = experiment_update.description
    db_experiment.data_source = experiment_update.data_source
    db_experiment.data_format = experiment_update.data_format
    db_experiment.target_column = experiment_update.target_column
    db_experiment.task_type = experiment_update.task_type
    
    # Update configs if provided
    if hasattr(experiment_update, "features_config") and experiment_update.features_config is not None:
        db_experiment.features_config = experiment_update.features_config
    
    if hasattr(experiment_update, "model_configuration") and experiment_update.model_configuration is not None:
        db_experiment.model_configuration = experiment_update.model_configuration
        
    if hasattr(experiment_update, "hyperparams") and experiment_update.hyperparams is not None:
        db_experiment.hyperparams = experiment_update.hyperparams
    
    db.commit()
    db.refresh(db_experiment)
    return db_experiment

def update_experiment_results(db: Session, experiment_id: str, results: dict):
    db_experiment = get_experiment(db, experiment_id)
    if not db_experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Update results
    db_experiment.results = results
    
    db.commit()
    db.refresh(db_experiment)
    return db_experiment

def detect_features_for_experiment(db: Session, experiment_id: str):
    """
    Load data and detect feature types for an experiment
    """
    # Get experiment
    db_experiment = get_experiment(db, experiment_id)
    if not db_experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    try:
        # Load data
        df = DataLoader.load_data(db_experiment.data_source, db_experiment.data_format)
        
        # Detect feature types
        feature_types = FeatureProcessor.detect_feature_types(df, db_experiment.target_column)
        
        # Convert to list of FeatureConfig
        features_config = []
        
        # Add target column first if it exists
        if db_experiment.target_column and db_experiment.target_column in feature_types["target"]:
            features_config.append({
                "name": db_experiment.target_column,
                "feature_type": "target",
                "encoding": "none"
            })
        
        # Add other features
        for feature_type, columns in feature_types.items():
            if feature_type == "target":
                continue  # Already handled
                
            for column in columns:
                encoding = "none"
                if feature_type == "categorical":
                    encoding = "one_hot" if df[column].nunique() < 10 else "label"
                    
                features_config.append({
                    "name": column,
                    "feature_type": feature_type,
                    "encoding": encoding
                })
        
        # Update experiment with detected features
        db_experiment.features_config = features_config
        db.commit()
        db.refresh(db_experiment)
        
        return features_config
        
    except Exception as e:
        logger.error(f"Error detecting features: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error detecting features: {str(e)}")


def train_model_for_experiment(db: Session, experiment_id: str, training_request: schemas.TrainingRequest , study_url=str ):
    """
    Train a model for the experiment with optional hyperparameter optimization
    """
    # Get experiment
    db_experiment = get_experiment(db, experiment_id)
    if not db_experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    try:
        # Load data
        df = DataLoader.load_data(db_experiment.data_source, db_experiment.data_format)
        
        # Check if target column is set
        if not db_experiment.target_column or db_experiment.target_column not in df.columns:
            raise HTTPException(status_code=400, detail="Target column not set or not found in data")
        
        # Process features according to config
        features_config = db_experiment.features_config
        if not features_config:
            # Auto-detect if not configured
            features_config = detect_features_for_experiment(db, experiment_id)
        
        # Apply feature transformations
        transformed_df, encoders = FeatureProcessor.encode_features(df, features_config)
        feature_types = FeatureProcessor.detect_feature_types(transformed_df, db_experiment.target_column)
        id_columns = feature_types.get("id", [])
        logger.info(f"id_columns assigned: {id_columns}")
        logger.info(f"Sample of transformed_df:\n{transformed_df.head(5).to_string(index=False)}")

        # Prepare training data
        X_train, X_test, y_train, y_test = ModelTrainer.prepare_training_data(
            transformed_df, 
            db_experiment.target_column,
            id_columns
        )
        
        # Optimize hyperparameters if requested
        best_params, optimization_results, all_trials = ModelTrainer.optimize_hyperparams(
            X_train, 
            y_train, 
            X_test, 
            y_test, 
            db_experiment.task_type,
            training_request.optimization_metric,
            training_request.n_trials,
            training_request.use_gpu,
            experiment_id,
            study_url,
        )
        
        # Train final model with best params
        model, metrics = ModelTrainer.train_xgboost(
            X_train, 
            y_train, 
            X_test, 
            y_test, 
            db_experiment.task_type,
            best_params,
            training_request.use_gpu
        )
        
        # Save results
        results = {
            "metrics": metrics,
            "optimization_results": optimization_results,
            "feature_importance": {
                name: float(score) for name, score in 
                zip(transformed_df.drop(columns=[db_experiment.target_column]).columns, 
                    model.get_score(importance_type='gain').values())
            },
           "all_trials" : all_trials,
        }
        logger.info(f"All optimization trials: {json.dumps(all_trials, indent=2)}")
        # Update experiment with results and hyperparameters
        db_experiment.results = results
        db_experiment.hyperparams = best_params
        db.commit()
        db.refresh(db_experiment)
        
        return results
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

def process_dataset_from_url(
    db: Session, experiment_id: str, url: str, file_format: str = "csv", options:dict = None):
    """
    Download and process a dataset from a URL for an experiment
    
    Args:
        db: Database session
        experiment_id: ID of the experiment
        url: URL to download the dataset from
        file_format: Format of the file (csv, excel, json)
        options: Additional options for reading the dataset
    
    Returns:
        Updated experiment object
    """
    # Retrieve the experiment
    experiment = get_experiment(db, experiment_id=experiment_id)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_id} not found")
    
    # Set default options if none provided
    if options is None:
        options = {}
    
    try:
        # Download the file
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Read the file based on format
        file_content = io.BytesIO(response.content)
        
        if file_format.lower() == "csv":
            df = pd.read_csv(file_content, **options)
        elif file_format.lower() in ["xls", "xlsx", "excel"]:
            df = pd.read_excel(file_content, **options) 
        elif file_format.lower() == "json":
            df = pd.read_json(file_content, **options)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        # Store dataset info in the experiment
        experiment.dataset_info = {
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "source_url": url,
            "file_format": file_format,
            "options": options
        }
        
        # Save the dataset to a file in the data directory
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        
        dataset_path = os.path.join(data_dir, f"experiment_{experiment_id}_dataset.parquet")
        df.to_parquet(dataset_path)
        
        # Update the experiment with the dataset path
        experiment.dataset_path = dataset_path
        experiment.status = "dataset_loaded"
        
        # Save changes to the database
        db.commit()
        db.refresh(experiment)
        
        return experiment
        
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error downloading dataset: {str(e)}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing dataset: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error processing dataset: {str(e)}")