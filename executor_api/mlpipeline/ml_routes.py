# ml_routes.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File, Form, Body
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional, List
import os
import uuid
import json
import pandas as pd
import logging

from .database import get_db
from . import crud, schemas, models
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineering
from .ml_training import XGBoostTrainer
from .ml_schemas import (
    ExperimentCreate, ExperimentResults, ExperimentStatus,
    DataSourceType, ModelType, FeatureConfig, TrainingConfig
)

# Configure logging
logger = logging.getLogger("ml_routes")

# Create router
router = APIRouter(
    prefix="/experiments",
    tags=["experiments"],
    responses={404: {"description": "Not found"}},
)

# Create directory for model storage
os.makedirs("models", exist_ok=True)
os.makedirs("experiment_results", exist_ok=True)

# Helper function to run experiment in background
async def run_experiment(
    experiment_id: str,
    project_id: str,
    data_source,
    feature_config,
    training_config,
    db: Session
):
    try:
        # Update experiment status
        update_experiment_status(db, experiment_id, "loading_data")
        
        # Load data based on source type
        if data_source["source_type"] == DataSourceType.UPLOAD:
            # Should have been handled in the endpoint directly
            raise ValueError("Upload source type should be handled in the endpoint")
            
        elif data_source["source_type"] == DataSourceType.FILE_PATH:
            df = DataLoader.load_from_path(data_source["file_path_config"]["file_path"])
            
        elif data_source["source_type"] == DataSourceType.S3:
            s3_config = data_source["s3_config"]
            df = DataLoader.load_from_s3(
                s3_config["bucket_name"],
                s3_config["object_key"],
                s3_config.get("aws_access_key_id"),
                s3_config.get("aws_secret_access_key")
            )
        
        # Update status
        update_experiment_status(db, experiment_id, "preparing_features")
        
        # Prepare features
        X_transformed, y, encoders, updated_feature_config = FeatureEngineering.prepare_training_data(
            df, 
            feature_config["target_feature"],
            feature_config
        )
        
        # Update status
        update_experiment_status(db, experiment_id, "training_model")
        
        # Initialize trainer
        is_classification = training_config["model_type"] == ModelType.CLASSIFICATION
        trainer = XGBoostTrainer(is_classification=is_classification)
        
        # Train/optimize model
        if training_config.get("optimize", True):
            model, metrics = trainer.optimize(
                X_transformed, y,
                n_trials=training_config.get("num_trials", 20),
                test_size=training_config.get("test_size", 0.2),
                random_state=training_config.get("random_state", 42)
            )
        else:
            model, metrics = trainer.train(
                X_transformed, y,
                test_size=training_config.get("test_size", 0.2),
                random_state=training_config.get("random_state", 42)
            )
        
        # Save model
        model_path = f"models/{experiment_id}.model"
        trainer.save_model(model_path)
        
        # Save results
        results_path = f"experiment_results/{experiment_id}.json"
        trainer.save_experiment_results(results_path)
        
        # Get feature importance
        feature_importance = trainer.get_feature_importance()
        
        # Update experiment metadata in database
        experiment_results = {
            "metrics": metrics,
            "feature_importance": feature_importance,
            "model_path": model_path,
            "results_path": results_path,
            "feature_config": updated_feature_config
        }
        
        # Update project in database
        update_experiment_results(db, experiment_id, project_id, experiment_results)
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        # Update experiment status to failed
        update_experiment_status(db, experiment_id, "failed", str(e))


def update_experiment_status(db: Session, experiment_id: str, status: str, message: str = None):
    """Update experiment status in project config"""
    try:
        # Get project by experiment ID from metadata
        projects = crud.get_projects(db)
        for project in projects:
            config = project.config
            if config and "experiments" in config:
                for exp in config["experiments"]:
                    if exp.get("experiment_id") == experiment_id:
                        exp["status"] = status
                        if message:
                            exp["message"] = message
                        
                        # Update project
                        project.config = config
                        db.commit()
                        logger.info(f"Updated experiment {experiment_id} status to {status}")
                        return True
        
        logger.warning(f"Could not find experiment {experiment_id} to update status")
        return False
    except Exception as e:
        logger.error(f"Error updating experiment status: {str(e)}")
        return False


def update_experiment_results(db: Session, experiment_id: str, project_id: str, results: Dict[str, Any]):
    """Update experiment results in project config"""
    try:
        # Get project
        project = crud.get_project(db, project_id)
        if not project:
            logger.warning(f"Project {project_id} not found")
            return False
            
        config = project.config or {}
        if "experiments" not in config:
            config["experiments"] = []
            
        # Find experiment or add new entry
        experiment_found = False
        for exp in config["experiments"]:
            if exp.get("experiment_id") == experiment_id:
                exp.update({
                    "status": "completed",
                    "results": results["metrics"],
                    "feature_importance": results["feature_importance"],
                    "model_path": results["model_path"],
                    "results_path": results["results_path"]
                })
                experiment_found = True
                break
                
        if not experiment_found:
            config["experiments"].append({
                "experiment_id": experiment_id,
                "status": "completed",
                "results": results["metrics"],
                "feature_importance": results["feature_importance"],
                "model_path": results["model_path"],
                "results_path": results["results_path"]
            })
            
        # Update project
        project.config = config
        db.commit()
        logger.info(f"Updated experiment {experiment_id} results")
        return True
        
    except Exception as e:
        logger.error(f"Error updating experiment results: {str(e)}")
        return False


@router.post("/{project_id}/create", response_model=ExperimentStatus)
async def create_experiment(
    project_id: str,
    experiment: ExperimentCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create a new ML experiment for a project"""
    # Check if project exists
    db_project = crud.get_project(db, project_id)
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Generate experiment ID
    experiment_id = str(uuid.uuid4())
    
    # Initialize project config if needed
    if not db_project.config:
        db_project.config = {}
    
    # Add experiment to project config
    if "experiments" not in db_project.config:
        db_project.config["experiments"] = []
        
    # Add experiment metadata
    db_project.config["experiments"].append({
        "experiment_id": experiment_id,
        "name": experiment.name,
        "description": experiment.description,
        "status": "created",
        "created_at": str(pd.Timestamp.now())
    })
    
    # Save to database
    db.commit()
    
    # Run experiment in background
    background_tasks.add_task(
        run_experiment,
        experiment_id=experiment_id,
        project_id=project_id,
        data_source=experiment.data_source.dict(),
        feature_config=experiment.feature_config.dict() if experiment.feature_config else None,
        training_config=experiment.training_config.dict(),
        db=db
    )
    
    return ExperimentStatus(
        experiment_id=experiment_id,
        project_id=project_id,
        status="created"
    )


@router.post("/{project_id}/upload", response_model=ExperimentStatus)
async def create_experiment_with_upload(
    project_id: str,
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    target_feature: str = Form(...),
    model_type: ModelType = Form(ModelType.CLASSIFICATION),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """Create a new ML experiment with an uploaded file"""
    # Check if project exists
    db_project = crud.get_project(db, project_id)
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    try:
        # Load data
        df = await DataLoader.load_file(file)
        
        # Generate experiment ID
        experiment_id = str(uuid.uuid4())
        
        # Save file temporarily
        temp_file_path = f"uploads/{experiment_id}_{file.filename}"
        os.makedirs("uploads", exist_ok=True)
        
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())
        
        # Detect features
        feature_types = FeatureEngineering.detect_feature_types(df)
        # Initialize project config if needed
        if not db_project.config:
            db_project.config = {}
        
        # Add experiment to project config
        if "experiments" not in db_project.config:
            db_project.config["experiments"] = []
            
        # Create feature config from detected features
        feature_config = {
            "numerical_features": feature_types["numerical_features"],
            "categorical_features": feature_types["categorical_features"],
            "id_features": feature_types["id_features"],
            "target_feature": target_feature,
            "onehot_encode": feature_types["categorical_features"],  # Default to one-hot encoding
            "label_encode": []
        }
        
        # Add experiment metadata
        db_project.config["experiments"].append({
            "experiment_id": experiment_id,
            "name": name,
            "description": description,
            "status": "processing_upload",
            "file_path": temp_file_path,
            "feature_config": feature_config,
            "created_at": str(pd.Timestamp.now())
        })
        
        # Save to database
        db.commit()
        
        # Run experiment in background
        training_config = {
            "model_type": model_type,
            "test_size": 0.2,
            "random_state": 42,
            "optimize": True,
            "num_trials": 20
        }
        
        data_source = {
            "source_type": DataSourceType.FILE_PATH,
            "file_path_config": {"file_path": temp_file_path}
        }
        
        background_tasks.add_task(
            run_experiment,
            experiment_id=experiment_id,
            project_id=project_id,
            data_source=data_source,
            feature_config=feature_config,
            training_config=training_config,
            db=db
        )
        
        return ExperimentStatus(
            experiment_id=experiment_id,
            project_id=project_id,
            status="processing_upload"
        )
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing upload: {str(e)}"
        )


@router.get("/{project_id}/list", response_model=List[Dict[str, Any]])
def list_project_experiments(project_id: str, db: Session = Depends(get_db)):
    """List all experiments for a project"""
    # Check if project exists
    db_project = crud.get_project(db, project_id)
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Get experiments from project config
    config = db_project.config or {}
    experiments = config.get("experiments", [])
    
    return experiments


@router.get("/{project_id}/experiment/{experiment_id}", response_model=Dict[str, Any])
def get_experiment_details(
    project_id: str,
    experiment_id: str,
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific experiment"""
    # Check if project exists
    db_project = crud.get_project(db, project_id)
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Find experiment in project config
    config = db_project.config or {}
    experiments = config.get("experiments", [])
    
    for experiment in experiments:
        if experiment.get("experiment_id") == experiment_id:
            return experiment
    
    raise HTTPException(status_code=404, detail="Experiment not found")


@router.post("/{project_id}/experiment/{experiment_id}/update-features", response_model=ExperimentStatus)
async def update_experiment_features(
    project_id: str,
    experiment_id: str,
    feature_config: FeatureConfig = Body(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """Update feature configuration and re-run an experiment"""
    # Check if project exists
    db_project = crud.get_project(db, project_id)
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Find experiment in project config
    config = db_project.config or {}
    experiments = config.get("experiments", [])
    
    experiment_found = False
    file_path = None
    training_config = None
    
    for exp in experiments:
        if exp.get("experiment_id") == experiment_id:
            experiment_found = True
            # Check if experiment is in a state that can be updated
            if exp.get("status") in ["failed", "completed"]:
                # Update feature config
                exp["feature_config"] = feature_config.dict()
                exp["status"] = "updating_features"
                file_path = exp.get("file_path")
                training_config = exp.get("training_config", {
                    "model_type": ModelType.CLASSIFICATION,
                    "test_size": 0.2,
                    "random_state": 42,
                    "optimize": True,
                    "num_trials": 20
                })
                break
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Experiment is currently in state '{exp.get('status')}' and cannot be updated"
                )
    
    if not experiment_found:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Make sure we have a file path
    if not file_path:
        raise HTTPException(
            status_code=400,
            detail="Cannot update experiment with missing file path"
        )
    
    # Save to database
    db.commit()
    
    # Re-run experiment with updated feature config
    data_source = {
        "source_type": DataSourceType.FILE_PATH,
        "file_path_config": {"file_path": file_path}
    }
    
    background_tasks.add_task(
        run_experiment,
        experiment_id=experiment_id,
        project_id=project_id,
        data_source=data_source,
        feature_config=feature_config.dict(),
        training_config=training_config,
        db=db
    )
    
    return ExperimentStatus(
        experiment_id=experiment_id,
        project_id=project_id,
        status="updating_features"
    )


@router.post("/{project_id}/experiment/{experiment_id}/predict", response_model=Dict[str, Any])
async def make_prediction(
    project_id: str,
    experiment_id: str,
    data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """Make a prediction using a trained experiment model"""
    import xgboost as xgb
    import numpy as np
    import pandas as pd
    
    # Check if project exists
    db_project = crud.get_project(db, project_id)
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Find experiment in project config
    config = db_project.config or {}
    experiments = config.get("experiments", [])
    
    experiment = None
    for exp in experiments:
        if exp.get("experiment_id") == experiment_id:
            experiment = exp
            break
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Check if experiment is completed
    if experiment.get("status") != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Experiment is in state '{experiment.get('status')}' and cannot be used for prediction"
        )
    
    # Check if we have a model path
    model_path = experiment.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(
            status_code=400,
            detail="Model file not found"
        )
    
    try:
        # Convert input data to DataFrame
        if isinstance(data, dict) and "data" in data:
            input_data = pd.DataFrame(data["data"])
        elif isinstance(data, dict) and all(isinstance(v, (list, int, float, str)) for v in data.values()):
            # Single row as dict
            input_data = pd.DataFrame([data])
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid input data format. Expected a dictionary with 'data' key containing rows of data"
            )
        
        # Load feature config
        feature_config = experiment.get("feature_config", {})
        
        # Process features using the same transformations as during training
        # Note: This is simplified - in a real application, you'd need to apply the exact same transformations
        # as during training, including loading encoders
        
        # Load the model
        model = xgb.Booster()
        model.load_model(model_path)
        
        # Prepare data for prediction
        # Simplified approach - assumes input data already has the same columns as training data
        dmatrix = xgb.DMatrix(input_data)
        
        # Make prediction
        predictions = model.predict(dmatrix)
        
        # Format response based on model type
        model_type = experiment.get("training_config", {}).get("model_type", "classification")
        if model_type == "classification":
            class_predictions = (predictions > 0.5).astype(int)
            result = {
                "probabilities": predictions.tolist(),
                "predictions": class_predictions.tolist()
            }
        else:
            result = {
                "predictions": predictions.tolist()
            }
        
        return {
            "experiment_id": experiment_id,
            "project_id": project_id,
            "results": result
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )