import logging
logger = logging.getLogger(__name__)
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class UserBase(BaseModel):
    user_id: str
    username: str
    email: EmailStr


class UserCreate(UserBase):
    meta_data: Optional[Dict[str, Any]] = None


class User(UserBase):
    id: str
    meta_data: Optional[Dict[str, Any]] = None
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


# Add to schemas.py

from enum import Enum

class DataFormat(str, Enum):
    CSV = "csv"
    PARQUET = "parquet"
    CEPH = "ceph"
    JSON = "json"

class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

class FeatureType(str, Enum):
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    TARGET = "target"
    ID = "id"
    
class EncodingType(str, Enum):
    ONE_HOT = "one_hot"
    LABEL = "label"
    NONE = "none"

class FeatureConfig(BaseModel):
    name: str
    feature_type: FeatureType
    encoding: Optional[EncodingType] = EncodingType.NONE
    
class MLExperimentBase(BaseModel):
    name: str
    description: Optional[str] = None
    data_source: str
    data_format: DataFormat
    target_column: Optional[str] = None
    task_type: TaskType

class MLExperimentCreate(MLExperimentBase):
    features_config: Optional[List[FeatureConfig]] = None #Optional[Dict[str, Any]] = None  # Changed from List to Dict Optional[List[Dict[str, Any]]] = None
    model_configuration : Optional[Dict[str, Any]] = None
    hyperparams: Optional[Dict[str, Any]] = None

class MLExperiment(MLExperimentBase):
    id: str
    project_id: str
    features_config: Optional[List[FeatureConfig]] = None ###Optional[Dict[str, Any]] = None  # Changed from List to Dict Optional[List[Dict[str, Any]]] = None
    results: Optional[Dict[str, Any]] = None
    model_configuration : Optional[Dict[str, Any]] = None
    hyperparams: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class TrainingRequest(BaseModel):
    experiment_id: str
    optimization_metric: str = "accuracy"  # For classification default
    n_trials: int = 20  # Default number of Optuna trials
    use_gpu: bool = False


class DatasetUploadRequest(BaseModel):
    url: str
    file_format: str = "csv"  # Default to CSV format
    options: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "url": "https://example.com/dataset.csv",
                "file_format": "csv",
                "options": {
                    "separator": ",",
                    "header": True,
                    "index_col": 0
                }
            }
        }