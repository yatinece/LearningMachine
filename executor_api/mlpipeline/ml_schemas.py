# ml_schemas.py
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
from enum import Enum


class DataSourceType(str, Enum):
    UPLOAD = "upload"
    FILE_PATH = "file_path"
    S3 = "s3"


class S3Config(BaseModel):
    bucket_name: str
    object_key: str
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None


class FilePathConfig(BaseModel):
    file_path: str


class DataSource(BaseModel):
    source_type: DataSourceType
    s3_config: Optional[S3Config] = None
    file_path_config: Optional[FilePathConfig] = None


class ModelType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class FeatureConfig(BaseModel):
    numerical_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    id_features: Optional[List[str]] = None
    target_feature: str
    onehot_encode: Optional[List[str]] = None
    label_encode: Optional[List[str]] = None


class TrainingConfig(BaseModel):
    model_type: ModelType
    test_size: float = 0.2
    random_state: int = 42
    optimize: bool = True
    num_trials: int = 20


class ExperimentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    data_source: DataSource
    feature_config: Optional[FeatureConfig] = None
    training_config: TrainingConfig


class ExperimentResults(BaseModel):
    experiment_id: str
    project_id: str
    metrics: Dict[str, Any]
    feature_importance: Dict[str, float]
    model_path: str
    status: str = "completed"


class ExperimentStatus(BaseModel):
    experiment_id: str
    project_id: str
    status: str
    message: Optional[str] = None