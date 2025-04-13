# learnai_ready/core/schemas.py

from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field


class DataSourceType(str, Enum):
    CEPH = "ceph"
    LOCAL = "local"


class CephConfig(BaseModel):
    bucket_name: str = Field(default="ml-data-bucket")
    endpoint_url: str = Field(default="https://ceph.example.com")
    access_key: str = Field(default="${CEPH_ACCESS_KEY}")
    secret_key: str = Field(default="${CEPH_SECRET_KEY}")
    region: str = Field(default="default")


class LocalConfig(BaseModel):
    base_path: str = Field(default="/data/ml-datasets")


class ReadOptions(BaseModel):
    encoding: str = Field(default="utf-8")
    delimiter: str = Field(default=",")
    header: bool = Field(default=True)
    na_values: List[str] = Field(default=["NA", "null", ""])


class DataSourceConfig(BaseModel):
    type: DataSourceType
    ceph: Optional[CephConfig] = None
    local: Optional[LocalConfig] = None
    file_name: str = Field(default="customer_churn_data.csv")
    file_format: str = Field(default="csv")
    read_options: ReadOptions = Field(default_factory=ReadOptions)


class CategoricalFeature(BaseModel):
    name: str
    encoding: str = Field(default="one_hot")


class FeatureSelection(BaseModel):
    enabled: bool = Field(default=True)
    method: str = Field(default="rfe")
    n_features: int = Field(default=10)


class FeatureEngineering(BaseModel):
    scaling: str = Field(default="standard")
    handling_missing: str = Field(default="median")
    feature_selection: FeatureSelection = Field(default_factory=FeatureSelection)


class ModelConfig(BaseModel):
    target_variable: str = Field(default="churn")
    features: List[str] = Field(default=["tenure", "monthly_charges", "total_charges", "contract_type", "payment_method", "internet_service"])
    categorical_features: List[CategoricalFeature] = Field(default_factory=list)
    feature_engineering: FeatureEngineering = Field(default_factory=FeatureEngineering)


class XGBoostBaseConfig(BaseModel):
    n_estimators: int = Field(default=100)
    learning_rate: float = Field(default=0.1)
    max_depth: int = Field(default=6)
    subsample: float = Field(default=0.8)
    colsample_bytree: float = Field(default=0.8)
    random_state: int = Field(default=42)


class XGBoostConfig(BaseModel):
    objective: str = Field(default="binary:logistic")
    eval_metric: str = Field(default="auc")
    booster: str = Field(default="gbtree")
    base_config: XGBoostBaseConfig = Field(default_factory=XGBoostBaseConfig)


class CatBoostBaseConfig(BaseModel):
    iterations: int = Field(default=500)
    learning_rate: float = Field(default=0.1)
    depth: int = Field(default=6)
    random_seed: int = Field(default=42)
    verbose: int = Field(default=100)


class CatBoostConfig(BaseModel):
    loss_function: str = Field(default="Logloss")
    eval_metric: str = Field(default="AUC")
    base_config: CatBoostBaseConfig = Field(default_factory=CatBoostBaseConfig)


class ModelSelection(BaseModel):
    type: str = Field(default="xgboost")
    xgboost: Optional[XGBoostConfig] = None
    catboost: Optional[CatBoostConfig] = None


class HyperparameterTuning(BaseModel):
    enabled: bool = Field(default=True)
    method: str = Field(default="bayesian")
    cv_folds: int = Field(default=5)
    scoring: str = Field(default="roc_auc")
    n_trials: int = Field(default=50)
    xgboost_params: Dict[str, List[Any]] = Field(default_factory=lambda: {
        'n_estimators': [50, 100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6, 8, 10],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4]
    })
    catboost_params: Dict[str, List[Any]] = Field(default_factory=lambda: {
        'iterations': [100, 250, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'depth': [4, 6, 8, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'border_count': [32, 64, 128, 254],
        'bagging_temperature': [0, 1, 10]
    })


class TrainTestSplit(BaseModel):
    test_size: float = Field(default=0.2)
    validation_size: float = Field(default=0.15)
    stratify: bool = Field(default=True)
    random_state: int = Field(default=42)


class EarlyStopping(BaseModel):
    enabled: bool = Field(default=True)
    patience: int = Field(default=20)
    monitor: str = Field(default="val_auc")
    mode: str = Field(default="max")


class Callback(BaseModel):
    type: str
    monitor: Optional[str] = None
    mode: Optional[str] = None
    save_best_only: Optional[bool] = None
    factor: Optional[float] = None
    patience: Optional[int] = None


class TrainingConfig(BaseModel):
    train_test_split: TrainTestSplit = Field(default_factory=TrainTestSplit)
    early_stopping: EarlyStopping = Field(default_factory=EarlyStopping)
    callbacks: List[Callback] = Field(default_factory=lambda: [
        {
            'type': "model_checkpoint",
            'monitor': "val_auc",
            'mode': "max",
            'save_best_only': True
        },
        {
            'type': "learning_rate_scheduler",
            'monitor': "val_loss",
            'mode': "min",
            'factor': 0.5,
            'patience': 10
        }
    ])


class MLflowConfig(BaseModel):
    tracking_uri: str = Field(default="http://mlflow-server:5000")
    experiment_name: str = Field(default="churn-prediction")
    run_name: str = Field(default="xgboost-run-${RUN_ID}")


class ExperimentTracking(BaseModel):
    enabled: bool = Field(default=True)
    backend: str = Field(default="mlflow")
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    metrics: List[str] = Field(default=["accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix", "feature_importance"])
    artifacts: List[str] = Field(default=["model_file", "feature_importance_plot", "confusion_matrix_plot", "roc_curve_plot", 
                       "precision_recall_curve", "calibration_curve", "hyperparameter_search_results"])


class PredictionOutput(BaseModel):
    enabled: bool = Field(default=True)
    output_type: str = Field(default="local")
    path: str = Field(default="${OUTPUT_PATH}/predictions_${RUN_ID}.csv")
    include_probabilities: bool = Field(default=True)
    include_feature_importance: bool = Field(default=True)


class Reports(BaseModel):
    format: str = Field(default="html")
    output_path: str = Field(default="/reports/experiment_report_${RUN_ID}.html")
    include_sections: List[str] = Field(default=["experiment_overview", "data_profile", "model_performance", 
                         "feature_importance", "hyperparameter_tuning_results", 
                         "confusion_matrix", "roc_curves", "distribution_plots"])


class OutputConfig(BaseModel):
    save_model: bool = Field(default=True)
    model_format: str = Field(default="pickle")
    model_path: str = Field(default="/models/${MODEL_TYPE}_${RUN_ID}.pkl")
    prediction_output: PredictionOutput = Field(default_factory=PredictionOutput)
    reports: Reports = Field(default_factory=Reports)


class MLExperimentConfig(BaseModel):
    data_source: DataSourceConfig
    model_settings: ModelConfig
    model: ModelSelection
    hyperparameter_tuning: HyperparameterTuning
    training: TrainingConfig
    experiment_tracking: ExperimentTracking
    output: OutputConfig