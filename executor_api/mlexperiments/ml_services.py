# ml_services.py
import logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import optuna
from fastapi import HTTPException
import json
import boto3
from io import StringIO, BytesIO



class DataLoader:
    """Service for loading data from various sources"""
    
    @staticmethod
    def load_data(data_source: str, data_format: str) -> pd.DataFrame:
        """
        Load data from the specified source and format
        
        Args:
            data_source: Path or URL to the data
            data_format: Format of the data (csv, parquet, ceph)
            
        Returns:
            DataFrame containing the loaded data
        """
        logger.info(f"Loading data from {data_source} in {data_format} format")
        
        try:
            if data_format.lower() == "csv":
                # Handle local file or URL
                if data_source.startswith(("http://", "https://", "s3://")):
                    return pd.read_csv(data_source)
                elif os.path.exists(data_source):
                    return pd.read_csv(data_source)
                else:
                    raise HTTPException(status_code=404, detail=f"CSV file not found: {data_source}")
                    
            elif data_format.lower() == "parquet":
                # Handle local file or URL
                if data_source.startswith(("http://", "https://", "s3://")):
                    return pd.read_parquet(data_source)
                elif os.path.exists(data_source):
                    return pd.read_parquet(data_source)
                else:
                    raise HTTPException(status_code=404, detail=f"Parquet file not found: {data_source}")
                    
            elif data_format.lower() == "ceph":
                # Example for S3/Ceph compatible storage
                # This assumes data_source format like "bucket_name/object_key"
                try:
                    parts = data_source.split("/", 1)
                    if len(parts) != 2:
                        raise ValueError("Invalid Ceph source format. Expected 'bucket/object_key'")
                    
                    bucket_name, object_key = parts
                    
                    # Initialize S3 client (configure for your Ceph endpoint)
                    s3_client = boto3.client(
                        's3',
                        endpoint_url=os.environ.get('CEPH_ENDPOINT', 'http://localhost:8000'),
                        aws_access_key_id=os.environ.get('CEPH_ACCESS_KEY', 'minioadmin'),
                        aws_secret_access_key=os.environ.get('CEPH_SECRET_KEY', 'minioadmin')
                    )
                    
                    response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
                    
                    # Determine file type based on extension
                    if object_key.endswith('.csv'):
                        return pd.read_csv(BytesIO(response['Body'].read()))
                    elif object_key.endswith('.parquet'):
                        return pd.read_parquet(BytesIO(response['Body'].read()))
                    else:
                        raise HTTPException(status_code=400, detail=f"Unsupported file format for Ceph object: {object_key}")
                        
                except Exception as e:
                    logger.error(f"Error loading data from Ceph: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Error loading data from Ceph: {str(e)}")
                    
            elif data_format.lower() == "json":
                # Handle local file or URL
                if data_source.startswith(("http://", "https://", "s3://")):
                    return pd.read_json(data_source)
                elif os.path.exists(data_source):
                    return pd.read_json(data_source)
                else:
                    raise HTTPException(status_code=404, detail=f"JSON file not found: {data_source}")
                    
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported data format: {data_format}")
                
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")


class FeatureProcessor:
    """Service for feature engineering and processing"""
    
    @staticmethod
    def detect_feature_types(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Automatically detect feature types in the dataframe
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column if known
            
        Returns:
            Dictionary mapping feature types to lists of column names
        """
        logger.info("Detecting feature types")
        
        feature_types = {
            "numerical": [],
            "categorical": [],
            "id": [],
            "target": []
        }
        
        # If target column specified, add it
        if target_column and target_column in df.columns:
            feature_types["target"].append(target_column)
        
        # Process each column
        for col in df.columns:
            # Skip target column
            if col == target_column:
                continue
                
            # Check if column name suggests it's an ID
            if col.lower() in ["id", "uuid", "guid"] or "id" in col.lower():
                feature_types["id"].append(col)
                continue
                
            # Check data type
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if it's likely a categorical variable encoded as numeric
                if df[col].nunique() < 15 and df[col].nunique() / len(df) < 0.05:
                    feature_types["categorical"].append(col)
                else:
                    feature_types["numerical"].append(col)
            else:
                # Non-numeric columns are probably categorical
                feature_types["categorical"].append(col)
                
        logger.info(f"Feature type detection complete. Found: {json.dumps(feature_types)}")
        return feature_types
    
    @staticmethod
    def encode_features(
        df: pd.DataFrame, 
        feature_config: List[Dict[str, str]]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Encode features according to the specified configuration
        
        Args:
            df: Input DataFrame
            feature_config: List of feature configurations with name, type, and encoding
            
        Returns:
            Transformed DataFrame and encoders dictionary for later use
        """
        logger.info("Encoding features")
        
        # Create a copy of the dataframe to avoid modifying the original
        transformed_df = df.copy()
        encoders = {}
        
        for feature in feature_config:
            name = feature.get("name")
            feature_type = feature.get("feature_type")
            encoding = feature.get("encoding", "none")
            
            if name not in df.columns:
                logger.warning(f"Column {name} not found in dataframe, skipping")
                continue
                
            if feature_type == "categorical" and encoding != "none":
                if encoding == "one_hot":
                    # One-hot encode
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded = encoder.fit_transform(df[[name]])
                    
                    # Create new column names
                    new_cols = [f"{name}_{val}" for val in encoder.categories_[0]]
                    
                    # Add encoded columns to dataframe
                    for i, col in enumerate(new_cols):
                        transformed_df[col] = encoded[:, i]
                        
                    # Drop original column
                    transformed_df = transformed_df.drop(columns=[name])
                    
                    # Save encoder
                    encoders[name] = {"type": "one_hot", "encoder": encoder}
                    
                elif encoding == "label":
                    # Label encode
                    encoder = LabelEncoder()
                    transformed_df[name] = encoder.fit_transform(df[name])
                    
                    # Save encoder
                    encoders[name] = {"type": "label", "encoder": encoder}
            # Handle target feature encoding (label encode if string/categorical)
            elif feature_type == "target":
                if pd.api.types.is_object_dtype(df[name]) or pd.api.types.is_categorical_dtype(df[name]):
                    encoder = LabelEncoder()
                    transformed_df[name] = encoder.fit_transform(df[name])
                    encoders[name] = {"type": "label", "encoder": encoder}
                    logger.info(f"Encoded target column {name} with label encoder")        
        logger.info(f"Feature encoding complete. Transformed {len(encoders)} features")
        return transformed_df, encoders


class ModelTrainer:
    """Service for training XGBoost models with Optuna optimization"""
    
    @staticmethod
    def prepare_training_data(
        df: pd.DataFrame, 
        target_column: str,
        id_columns: List[str] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training by splitting into train/test sets
        
        Args:
            df: Processed DataFrame
            target_column: Name of the target column
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test arrays
        """
        logger.info(f"Preparing training data with target: {target_column}")
        
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column {target_column} not found in data")
        # Create a list of columns to drop (target + id columns)
        columns_to_drop = [target_column]
        
        # Add ID columns to drop list if provided
        if id_columns:
            logger.info(f"Excluding ID columns from training: {id_columns}")
            columns_to_drop.extend([col for col in id_columns if col in df.columns])
        
        # Filter out any object/string columns that might cause issues with XGBoost
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        if object_cols:
            logger.info(f"Found object dtype columns: {object_cols}")
            # Add object columns to drop list if they're not already included
            for col in object_cols:
                if col not in columns_to_drop and col != target_column:
                    logger.info(f"Adding object column to exclude list: {col}")
                    columns_to_drop.append(col)



        # Split data into features and target
        X = df.drop(columns=columns_to_drop)
        y = df[target_column]
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Data prepared. Training set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def train_xgboost(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        task_type: str,
        params: Dict[str, Any],
        use_gpu: bool = False
    ) -> Tuple[xgb.Booster, Dict[str, float]]:
        """
        Train an XGBoost model with the given parameters
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            task_type: 'classification' or 'regression'
            params: XGBoost parameters
            use_gpu: Whether to use GPU training
            
        Returns:
            Trained model and performance metrics
        """
        logger.info(f"Training XGBoost model for {task_type} task")
            # Handle DataFrame inputs and remove any object columns
        if isinstance(X_train, pd.DataFrame):
            object_cols = X_train.select_dtypes(include=['object']).columns
            if len(object_cols) > 0:
                logger.info(f"Removing object columns before XGBoost training: {list(object_cols)}")
                X_train = X_train.drop(columns=object_cols)
                X_test = X_test.drop(columns=object_cols)
        # Set tree method based on GPU availability
        if use_gpu:
            logger.info("Using GPU acceleration")
            params['tree_method'] = 'gpu_hist'
            params['gpu_id'] = 0
        else:
            logger.info("Using CPU training")
            params['tree_method'] = 'hist'
        
        # Prepare DMatrix objects
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Set objective based on task type
        if task_type == "classification":
            if len(np.unique(y_train)) > 2:
                params['objective'] = 'multi:softprob'
                params['num_class'] = len(np.unique(y_train))
            else:
                params['objective'] = 'binary:logistic'
        else:  # regression
            params['objective'] = 'reg:squarederror'
        
        # Train the model
        num_rounds = params.pop('num_boost_round', 100)
        
        # Set up early stopping
        evallist = [(dtrain, 'train'), (dtest, 'eval')]
        model = xgb.train(
            params, 
            dtrain, 
            num_rounds, 
            evallist, 
            early_stopping_rounds=20,
            verbose_eval=False
        )
        
        # Evaluate the model
        metrics = {}
        
        preds = model.predict(dtest)
        
        if task_type == "classification":
            if params.get('objective') == 'multi:softprob':
                # For multiclass, convert probabilities to class predictions
                preds = np.argmax(preds, axis=1)
            else:
                # For binary, convert probabilities to class predictions using 0.5 threshold
                preds = (preds > 0.5).astype(int)
                
            # Calculate accuracy
            metrics['accuracy'] = np.mean(preds == y_test)
            
            # Calculate F1 score (if binary or multiclass)
            from sklearn.metrics import f1_score
            metrics['f1_score'] = f1_score(y_test, preds, average='weighted')
        else:
            # For regression, calculate RMSE and MAE
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            metrics['rmse'] = float(np.sqrt(mean_squared_error(y_test, preds)))
            metrics['mae'] = float(mean_absolute_error(y_test, preds))
        
        logger.info(f"Model training complete. Metrics: {metrics}")
        return model, metrics
    
    @staticmethod
    def set_params(
        task_type: str,
        use_gpu: str,
        num_class: int = 2,
        param: dict = []) -> Dict[str, Any]:
                # Set objective based on task type
        if task_type == "classification":
            if num_class > 2:
                param['objective'] = 'multi:softprob'
                param['num_class'] = num_class
            else:
                param['objective'] = 'binary:logistic'
        else:  # regression
            param['objective'] = 'reg:squarederror'
        
        if task_type == "classification":
            pruning_metric = "logloss"
        else:
            pruning_metric = "rmse"
        if use_gpu:
            param['tree_method'] = 'gpu_hist'
            param['gpu_id'] = 0
        else:
            param['tree_method'] = 'hist'

        return param

    @staticmethod
    def optimize_hyperparams(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        task_type: str,
        optimization_metric: str,
        n_trials: int = 20,
        use_gpu: bool = False,
        experiment_id: str = "study_optuna",
        study_url: str = "sqlite:///optuna_studies.db" , 
    ) -> Tuple[Dict[str, Any], Dict[str, Any] , Dict[str, Any]]:
        """
        Optimize XGBoost hyperparameters using Optuna
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            task_type: 'classification' or 'regression'
            optimization_metric: Metric to optimize for
            n_trials: Number of Optuna trials
            use_gpu: Whether to use GPU training
            
        Returns:
            Best parameters and optimization results
        """
        logger.info(f"Starting hyperparameter optimization with Optuna for {task_type} task")
        
        # Prepare DMatrix objects
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, label=y_test , enable_categorical=True)
        
        # Function to evaluate parameters
        def objective(trial):
            # Define the search space
            param = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 20),
                'num_boost_round': trial.suggest_int('num_boost_round', 50, 300)
            }
            
            # Set GPU/CPU specific parameters

            param = ModelTrainer.set_params(task_type,use_gpu,len(np.unique(y_train)), param)
            # if use_gpu:
            #     param['tree_method'] = 'gpu_hist'
            #     param['gpu_id'] = 0
            # else:
            #     param['tree_method'] = 'hist'
            
            # # Set objective based on task type
            # if task_type == "classification":
            #     if len(np.unique(y_train)) > 2:
            #         param['objective'] = 'multi:softprob'
            #         param['num_class'] = len(np.unique(y_train))
            #     else:
            #         param['objective'] = 'binary:logistic'
            # else:  # regression
            #     param['objective'] = 'reg:squarederror'
            
            if task_type == "classification":
                pruning_metric = "logloss"
            else:
                pruning_metric = "rmse"
            # Train model with early stopping
            pruning_callback = optuna.integration.XGBoostPruningCallback(
                trial, "eval-" + pruning_metric ## pruning_metric replace optimization_metric
            )
            
            evallist = [(dtrain, 'train'), (dtest, 'eval')]
            model = xgb.train(
                param, 
                dtrain, 
                param['num_boost_round'], 
                evallist, 
                callbacks=[pruning_callback],
                early_stopping_rounds=20,
                verbose_eval=False,
                
            )
            
            # Evaluate model based on the optimization metric
            preds = model.predict(dtest)
            
            if task_type == "classification":
                if param.get('objective') == 'multi:softprob':
                    # For multiclass, convert probabilities to class predictions
                    preds = np.argmax(preds, axis=1)
                else:
                    # For binary, convert probabilities to class predictions
                    preds = (preds > 0.5).astype(int)
                
                if optimization_metric == "accuracy":
                    score = np.mean(preds == y_test)
                elif optimization_metric == "f1":
                    from sklearn.metrics import f1_score
                    score = f1_score(y_test, preds, average='weighted')
                else:
                    # Default to accuracy
                    score = np.mean(preds == y_test)
            else:
                # For regression
                if optimization_metric == "rmse":
                    from sklearn.metrics import mean_squared_error
                    score = -np.sqrt(mean_squared_error(y_test, preds))  # Negative because Optuna minimizes
                elif optimization_metric == "mae":
                    from sklearn.metrics import mean_absolute_error
                    score = -mean_absolute_error(y_test, preds)  # Negative because Optuna minimizes
                else:
                    # Default to RMSE
                    from sklearn.metrics import mean_squared_error
                    score = -np.sqrt(mean_squared_error(y_test, preds))
            
            return score
        
        # Create Optuna study
        study = optuna.create_study( study_name=experiment_id,
             storage=study_url,
            direction="maximize" if task_type == "classification" else "minimize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            load_if_exists=True,
        )
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        
        # Include objective and tree_method in best params
        best_params = ModelTrainer.set_params(task_type,use_gpu,len(np.unique(y_train)), best_params)
        # best_params['objective'] = param['objective']
        # best_params['tree_method'] = param['tree_method']
        # if len(np.unique(y_train)) > 2: best_params['num_class'] = len(np.unique(y_train))



        df = study.trials_dataframe(
            attrs=("number", "value", "params", "state", "datetime_start", "datetime_complete")
        )
        # Convert timestamp columns to string format
        for col in df.select_dtypes(include=['datetime64']).columns:
            df[col] = df[col].astype(str)

        all_trials = df.to_dict(orient="records")
        # if task_type == "classification":
        #     if len(np.unique(y_train)) > 2:
        #         best_params['objective'] = 'multi:softprob'
        #         best_params['num_class'] = len(np.unique(y_train))
        #     else:
        #         best_params['objective'] = 'binary:logistic'
        # else:
        #     best_params['objective'] = 'reg:squarederror'
        
        # if use_gpu:
        #     best_params['tree_method'] = 'gpu_hist'
        # else:
        #     best_params['tree_method'] = 'hist'
        
        # Get optimization results
        optimization_results = {
            "best_value": study.best_value,
            "best_iteration": study.best_trial.number,
            "n_trials": n_trials
        }
        
        logger.info(f"Hyperparameter optimization complete. Best parameters: {best_params}")
        return best_params, optimization_results, all_trials