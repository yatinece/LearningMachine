# Configuration Guide

This guide explains how to configure and use the LearnAI Ready framework for your machine learning experiments.

## Table of Contents
1. [Configuration Overview](#configuration-overview)
2. [Data Source Configuration](#data-source-configuration)
3. [Model Settings](#model-settings)
4. [Model Selection](#model-selection)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Training Configuration](#training-configuration)
7. [Experiment Tracking](#experiment-tracking)
8. [Output Configuration](#output-configuration)

## Configuration Overview

The configuration system is built using Pydantic models, ensuring type safety and validation. A complete configuration consists of seven main sections:

```yaml
data_source:
  # Data source configuration
model_settings:
  # Model settings configuration
model:
  # Model selection and configuration
hyperparameter_tuning:
  # Hyperparameter tuning settings
training:
  # Training process configuration
experiment_tracking:
  # Experiment tracking settings
output:
  # Output configuration
```

## Data Source Configuration

Configure where your data is stored and how to access it.

### Local Storage
```yaml
data_source:
  type: "local"
  local:
    base_path: "/path/to/your/data"
  file_name: "dataset.csv"
  file_format: "csv"
  read_options:
    encoding: "utf-8"
    delimiter: ","
    header: true
```

### Ceph Storage
```yaml
data_source:
  type: "ceph"
  ceph:
    bucket_name: "ml-data-bucket"
    endpoint_url: "https://ceph.example.com"
    access_key: "${CEPH_ACCESS_KEY}"
    secret_key: "${CEPH_SECRET_KEY}"
    region: "default"
  file_name: "dataset.csv"
  file_format: "csv"
```

## Model Settings

Define your target variable and features.

```yaml
model_settings:
  target_variable: "churn"
  features:
    - "tenure"
    - "monthly_charges"
    - "total_charges"
    - "contract_type"
    - "payment_method"
    - "internet_service"
  categorical_features:
    - name: "contract_type"
      encoding: "one_hot"
    - name: "payment_method"
      encoding: "one_hot"
  feature_engineering:
    scaling: "standard"
    handling_missing: "median"
    feature_selection:
      enabled: true
      method: "rfe"
      n_features: 10
```

## Model Selection

Choose and configure your ML model.

### XGBoost
```yaml
model:
  type: "xgboost"
  xgboost:
    objective: "binary:logistic"
    eval_metric: "auc"
    booster: "gbtree"
    base_config:
      n_estimators: 100
      learning_rate: 0.1
      max_depth: 6
      subsample: 0.8
      colsample_bytree: 0.8
      random_state: 42
```

### CatBoost
```yaml
model:
  type: "catboost"
  catboost:
    loss_function: "Logloss"
    eval_metric: "AUC"
    base_config:
      iterations: 500
      learning_rate: 0.1
      depth: 6
      random_seed: 42
      verbose: 100
```

## Hyperparameter Tuning

Configure hyperparameter optimization.

```yaml
hyperparameter_tuning:
  enabled: true
  method: "bayesian"
  cv_folds: 5
  scoring: "roc_auc"
  n_trials: 50
  xgboost_params:
    n_estimators: [50, 100, 200, 300, 500]
    learning_rate: [0.01, 0.05, 0.1, 0.2]
    max_depth: [3, 4, 5, 6, 8, 10]
    subsample: [0.6, 0.7, 0.8, 0.9, 1.0]
    colsample_bytree: [0.6, 0.7, 0.8, 0.9, 1.0]
    min_child_weight: [1, 3, 5, 7]
    gamma: [0, 0.1, 0.2, 0.3, 0.4]
```

## Training Configuration

Configure the training process.

```yaml
training:
  train_test_split:
    test_size: 0.2
    validation_size: 0.15
    stratify: true
    random_state: 42
  early_stopping:
    enabled: true
    patience: 20
    monitor: "val_auc"
    mode: "max"
  callbacks:
    - type: "model_checkpoint"
      monitor: "val_auc"
      mode: "max"
      save_best_only: true
    - type: "learning_rate_scheduler"
      monitor: "val_loss"
      mode: "min"
      factor: 0.5
      patience: 10
```

## Experiment Tracking

Configure experiment tracking with MLflow.

```yaml
experiment_tracking:
  enabled: true
  backend: "mlflow"
  mlflow:
    tracking_uri: "http://mlflow-server:5000"
    experiment_name: "churn-prediction"
    run_name: "xgboost-run-${RUN_ID}"
  metrics:
    - "accuracy"
    - "precision"
    - "recall"
    - "f1"
    - "roc_auc"
    - "confusion_matrix"
    - "feature_importance"
  artifacts:
    - "model_file"
    - "feature_importance_plot"
    - "confusion_matrix_plot"
    - "roc_curve_plot"
```

## Output Configuration

Configure output formats and locations.

```yaml
output:
  save_model: true
  model_format: "pickle"
  model_path: "/models/${MODEL_TYPE}_${RUN_ID}.pkl"
  prediction_output:
    enabled: true
    output_type: "local"
    path: "${OUTPUT_PATH}/predictions_${RUN_ID}.csv"
    include_probabilities: true
    include_feature_importance: true
  reports:
    format: "html"
    output_path: "/reports/experiment_report_${RUN_ID}.html"
    include_sections:
      - "experiment_overview"
      - "data_profile"
      - "model_performance"
      - "feature_importance"
      - "hyperparameter_analysis"
```

## Using the Configuration

### Via API
```python
import requests

# Generate configuration
response = requests.post(
    "http://localhost:8000/api/config/generate",
    json={
        "model_settings": {
            "target_variable": "churn",
            "features": ["feature1", "feature2"]
        }
    }
)
config = response.json()

# Execute experiment
response = requests.post(
    "http://localhost:8000/api/execute",
    json={
        "config_file": config["file_path"],
        "tasks": ["data_preparation", "training"]
    }
)
```

### Via Python
```python
from learnai_ready.core.services import ConfigService

# Initialize service
config_service = ConfigService()

# Create configuration
config = config_service.create_full_config()

# Modify specific settings
config.model_settings.target_variable = "churn"
config.model_settings.features = ["feature1", "feature2"]

# Save configuration
yaml_str = config_service.generate_yaml(config.dict())
file_path = config_service.save_yaml(config.dict()) 