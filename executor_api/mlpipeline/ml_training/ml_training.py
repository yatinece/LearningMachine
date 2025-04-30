# ml_training.py
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np
import optuna
import logging
import json
import joblib
import os

logger = logging.getLogger("ml_training")

class XGBoostTrainer:
    """Handles XGBoost model training with GPU/CPU auto-detection"""
    
    def __init__(self, is_classification=True):
        self.is_classification = is_classification
        self.best_model = None
        self.feature_importance = None
        self.metrics = {}
        
        # Check if GPU is available
        try:
            gpu_count = xgb.config.get_config()['n_gpus']
            self.use_gpu = gpu_count > 0
            logger.info(f"XGBoost detected {gpu_count} GPUs")
        except:
            self.use_gpu = False
            logger.info("No GPUs detected, using CPU")
    
    def _get_base_params(self):
        """Get base parameters for XGBoost model based on task and hardware"""
        if self.is_classification:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'hist'
            }
        else:
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'tree_method': 'hist'
            }
            
        # Use GPU if available
        if self.use_gpu:
            params['tree_method'] = 'gpu_hist'
            params['gpu_id'] = 0
            
        return params
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """Train an XGBoost model with default parameters"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Set parameters
        params = self._get_base_params()
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Train model
        logger.info("Training XGBoost model with default parameters")
        self.best_model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train'), (dtest, 'eval')],
            early_stopping_rounds=20,
            verbose_eval=10
        )
        
        # Get predictions
        predictions = self.best_model.predict(dtest)
        
        # Calculate metrics
        self._calculate_metrics(y_test, predictions)
        
        # Feature importance
        self.feature_importance = self.best_model.get_score(importance_type='gain')
        
        return self.best_model, self.metrics
    
    def optimize(self, X, y, n_trials=20, test_size=0.2, random_state=42):
        """Optimize XGBoost parameters using Optuna"""
        # Split data
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        
        # Define objective function for Optuna
        def objective(trial):
            # Base parameters
            params = self._get_base_params()
            
            # Parameters to optimize
            params.update({
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 2),
                'alpha': trial.suggest_float('alpha', 0, 5),
                'lambda': trial.suggest_float('lambda', 0, 5)
            })
            
            # Train with early stopping
            pruning_callback = optuna.integration.XGBoostPruningCallback(
                trial, "eval-" + params['eval_metric']
            )
            
            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=500,
                evals=[(dvalid, "eval")],
                early_stopping_rounds=30,
                verbose_eval=False,
                callbacks=[pruning_callback]
            )
            
            # Get validation score
            preds = bst.predict(dvalid)
            if self.is_classification:
                score = roc_auc_score(y_valid, preds)
            else:
                score = -mean_squared_error(y_valid, preds, squared=False)  # Negative RMSE for maximization
                
            return score
        
        # Create study
        logger.info(f"Starting Optuna optimization with {n_trials} trials")
        if self.is_classification:
            study = optuna.create_study(direction='maximize')
        else:
            study = optuna.create_study(direction='maximize')  # Using negative RMSE
            
        # Run optimization
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_params.update(self._get_base_params())
        logger.info(f"Best parameters: {best_params}")
        
        # Train final model with best parameters
        dtrain_full = xgb.DMatrix(X, label=y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        self.best_model = xgb.train(
            best_params,
            dtrain_full,
            num_boost_round=study.best_trial.user_attrs.get('n_estimators', 500)
        )
        
        # Get predictions and metrics
        predictions = self.best_model.predict(dtest)
        self._calculate_metrics(y_test, predictions)
        
        # Feature importance
        self.feature_importance = self.best_model.get_score(importance_type='gain')
        
        # Add optimization results to metrics
        self.metrics['optimization_study'] = {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': n_trials
        }
        
        return self.best_model, self.metrics
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate and store metrics based on task type"""
        if self.is_classification:
            y_pred_binary = (y_pred > 0.5).astype(int)
            self.metrics = {
                'accuracy': accuracy_score(y_true, y_pred_binary),
                'f1_score': f1_score(y_true, y_pred_binary, average='weighted'),
                'auc_roc': roc_auc_score(y_true, y_pred)
            }
            logger.info(f"Classification metrics: Accuracy={self.metrics['accuracy']:.4f}, "
                       f"F1={self.metrics['f1_score']:.4f}, AUC={self.metrics['auc_roc']:.4f}")
        else:
            self.metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': mean_squared_error(y_true, y_pred, squared=False),
                'mae': np.mean(np.abs(y_true - y_pred))
            }
            logger.info(f"Regression metrics: MSE={self.metrics['mse']:.4f}, "
                       f"RMSE={self.metrics['rmse']:.4f}, MAE={self.metrics['mae']:.4f}")
    
    def save_model(self, filepath):
        """Save the trained model to a file"""
        if self.best_model:
            self.best_model.save_model(filepath)
            logger.info(f"Model saved to {filepath}")
        else:
            logger.error("No model has been trained yet")
    
    def get_feature_importance(self):
        """Get feature importance as a sorted dictionary"""
        if not self.feature_importance:
            logger.warning("No feature importance available. Train a model first.")
            return {}
            
        return dict(sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def save_experiment_results(self, filepath):
        """Save experiment results including metrics and feature importance"""
        if not self.metrics:
            logger.warning("No metrics available. Train a model first.")
            return
            
        results = {
            "metrics": self.metrics,
            "feature_importance": self.get_feature_importance()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Experiment results saved to {filepath}")