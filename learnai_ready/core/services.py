# learnai_ready/core/services.py

import yaml
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
from learnai_ready.core.schemas import (
    MLExperimentConfig, DataSourceConfig, ModelConfig,
    ModelSelection, HyperparameterTuning, TrainingConfig,
    ExperimentTracking, OutputConfig
)


class ConfigService:
    """
    Service to handle configuration generation and management
    """
    
    def __init__(self, base_path: str = "./configs"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def _get_run_id(self) -> str:
        """Generate a unique run ID based on timestamp"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def build_config(self, 
                     data_source: Optional[Dict[str, Any]] = None,
                     model_settings: Optional[Dict[str, Any]] = None,
                     model: Optional[Dict[str, Any]] = None,
                     hyperparameter_tuning: Optional[Dict[str, Any]] = None,
                     training: Optional[Dict[str, Any]] = None,
                     experiment_tracking: Optional[Dict[str, Any]] = None,
                     output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Build a configuration dictionary with provided parameters
        """
        # Start with a full default configuration
        config = self.create_full_config().dict()
        
        # Update with provided values
        if data_source:
            config["data_source"].update(data_source)
        
        if model_settings:
            config["model_settings"].update(model_settings)
        
        if model:
            config["model"].update(model)
        
        if hyperparameter_tuning:
            config["hyperparameter_tuning"].update(hyperparameter_tuning)
        
        if training:
            config["training"].update(training)
        
        if experiment_tracking:
            config["experiment_tracking"].update(experiment_tracking)
        
        if output:
            config["output"].update(output)
        
        return config
    
    def create_full_config(self) -> MLExperimentConfig:
        """
        Create a full default configuration
        """
        return MLExperimentConfig(
            data_source=DataSourceConfig(type="ceph"),
            model_settings=ModelConfig(),
            model=ModelSelection(type="xgboost"),
            hyperparameter_tuning=HyperparameterTuning(),
            training=TrainingConfig(),
            experiment_tracking=ExperimentTracking(),
            output=OutputConfig()
        )
    
    def generate_yaml(self, config: Dict[str, Any]) -> str:
        """
        Generate YAML string from configuration dictionary
        """
        return yaml.dump(config, sort_keys=False, default_flow_style=False)
    
    def save_yaml(self, config: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save configuration as YAML file and return the filepath
        """
        if not filename:
            run_id = self._get_run_id()
            filename = f"ml_experiment_config_{run_id}.yaml"
        
        filepath = os.path.join(self.base_path, filename)
        
        with open(filepath, "w") as f:
            yaml.dump(config, f, sort_keys=False, default_flow_style=False)
        
        return filepath
    
    def load_yaml(self, filepath: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        """
        with open(filepath, "r") as f:
            return yaml.safe_load(f)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate if the configuration is properly structured
        """
        try:
            # Try creating a pydantic model with the config to validate
            MLExperimentConfig(**config)
            return True
        except Exception:
            return False
    
    def get_template(self, section: str) -> Dict[str, Any]:
        """
        Get a template configuration for a specific section
        """
        full_config = self.create_full_config().dict()
        
        if section in full_config:
            return {section: full_config[section]}
        else:
            raise ValueError(f"Section '{section}' not found in configuration schema")
    
    def merge_configs(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries
        """
        merged_config = {}
        
        for config in configs:
            merged_config.update(config)
        
        return merged_config