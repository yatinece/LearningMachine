# learnai_ready/core/executor.py

import os
import subprocess
import yaml
import logging
from typing import Dict, Any, List, Optional
import importlib
import inspect

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TaskExecutor:
    """
    Executes ML pipeline tasks based on YAML configuration
    """
    
    def __init__(self, config_path: str, output_dir: str = "./output"):
        self.config_path = config_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_available_tasks(self) -> List[str]:
        """Get list of available tasks based on config sections"""
        tasks = []
        
        if 'data_source' in self.config:
            tasks.append('data_preparation')
        
        if 'model_config' in self.config:
            tasks.append('feature_engineering')
        
        if 'model' in self.config:
            tasks.append('model_selection')
        
        if 'hyperparameter_tuning' in self.config and self.config['hyperparameter_tuning'].get('enabled', False):
            tasks.append('hyperparameter_tuning')
        
        if 'training' in self.config:
            tasks.append('training')
        
        if 'experiment_tracking' in self.config and self.config['experiment_tracking'].get('enabled', False):
            tasks.append('experiment_tracking')
        
        if 'output' in self.config:
            tasks.append('generate_outputs')
        
        return tasks
    
    def run_task(self, task_name: str) -> bool:
        """Run a specific task"""
        task_map = {
            'data_preparation': self._run_data_preparation,
            'feature_engineering': self._run_feature_engineering,
            'model_selection': self._run_model_selection,
            'hyperparameter_tuning': self._run_hyperparameter_tuning,
            'training': self._run_training,
            'experiment_tracking': self._run_experiment_tracking,
            'generate_outputs': self._run_generate_outputs
        }
        
        if task_name in task_map:
            logger.info(f"Running task: {task_name}")
            return task_map[task_name]()
        else:
            logger.error(f"Task '{task_name}' not found")
            return False
    
    def _run_data_preparation(self) -> bool:
        """Run data preparation task"""
        if 'data_source' not in self.config:
            logger.error("data_source section missing in config")
            return False
        
        logger.info(f"Running data preparation with source type: {self.config['data_source']['type']}")
        # Here we'd implement the actual data preparation logic
        # For now, just log and return success
        return True
    
    def _run_feature_engineering(self) -> bool:
        """Run feature engineering task"""
        if 'model_config' not in self.config:
            logger.error("model_config section missing in config")
            return False
        
        logger.info(f"Running feature engineering on target: {self.config['model_config']['target_variable']}")
        # Here we'd implement the feature engineering logic
        return True
    
    def _run_model_selection(self) -> bool:
        """Run model selection task"""
        if 'model' not in self.config:
            logger.error("model section missing in config")
            return False
        
        logger.info(f"Setting up model: {self.config['model']['type']}")
        # Here we'd implement the model selection logic
        return True
    
    def _run_hyperparameter_tuning(self) -> bool:
        """Run hyperparameter tuning task"""
        if 'hyperparameter_tuning' not in self.config:
            logger.error("hyperparameter_tuning section missing in config")
            return False
        
        if not self.config['hyperparameter_tuning'].get('enabled', False):
            logger.info("Hyperparameter tuning is disabled in config")
            return True
        
        logger.info(f"Running hyperparameter tuning with method: {self.config['hyperparameter_tuning']['method']}")
        # Here we'd implement the hyperparameter tuning logic
        return True
    
    def _run_training(self) -> bool:
        """Run model training task"""
        if 'training' not in self.config:
            logger.error("training section missing in config")
            return False
        
        logger.info("Training model with configured parameters")
        # Here we'd implement the model training logic
        return True
    
    def _run_experiment_tracking(self) -> bool:
        """Run experiment tracking task"""
        if 'experiment_tracking' not in self.config:
            logger.error("experiment_tracking section missing in config")
            return False
        
        if not self.config['experiment_tracking'].get('enabled', False):
            logger.info("Experiment tracking is disabled in config")
            return True
        
        logger.info(f"Setting up experiment tracking with backend: {self.config['experiment_tracking']['backend']}")
        # Here we'd implement the experiment tracking logic
        return True
    
    def _run_generate_outputs(self) -> bool:
        """Run output generation task"""
        if 'output' not in self.config:
            logger.error("output section missing in config")
            return False
        
        logger.info("Generating outputs and reports")
        # Here we'd implement the output generation logic
        return True
    
    def run_pipeline(self, tasks: Optional[List[str]] = None) -> bool:
        """
        Run the entire pipeline or specific tasks
        
        Args:
            tasks: List of task names to run. If None, run all available tasks.
        
        Returns:
            bool: True if all tasks completed successfully, False otherwise
        """
        if tasks is None:
            tasks = self._get_available_tasks()
        
        logger.info(f"Starting pipeline execution with tasks: {tasks}")
        
        success = True
        for task in tasks:
            task_success = self.run_task(task)
            if not task_success:
                logger.error(f"Task '{task}' failed")
                success = False
                break
        
        if success:
            logger.info("Pipeline execution completed successfully")
        else:
            logger.error("Pipeline execution failed")
        
        return success