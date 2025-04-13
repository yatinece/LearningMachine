# learnai_ready/api/app.py

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.responses import FileResponse
from typing import Dict, Any, List, Optional
import os
from datetime import datetime
import yaml
import shutil
from pydantic import BaseModel

from learnai_ready.core.services import ConfigService
from learnai_ready.core.executor import TaskExecutor

app = FastAPI(
    title="LearnAI: Ready",
    description="API for ML experiment configuration and execution",
    version="1.0.0"
)

# Initialize services
config_service = ConfigService()

# Base directory for storing files
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
CONFIG_DIR = os.path.join(BASE_DIR, "configs")
os.makedirs(CONFIG_DIR, exist_ok=True)




class ConfigRequest(BaseModel):
    data_source: Optional[Dict[str, Any]] = None
    model_settings: Optional[Dict[str, Any]] = None
    model: Optional[Dict[str, Any]] = None
    hyperparameter_tuning: Optional[Dict[str, Any]] = None
    training: Optional[Dict[str, Any]] = None
    experiment_tracking: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None

class ExecuteRequest(BaseModel):
    config_file: str
    tasks: Optional[List[str]] = None


class ConfigResponse(BaseModel):
    config_id: str
    yaml_content: str
    file_path: str


@app.get("/")
async def root():
    return {"message": "Welcome to LearnAI: Ready API"}

@app.post("/api/config/generate", response_model=ConfigResponse)
async def generate_config(config_data: ConfigRequest):
    """Generate YAML configuration from provided data"""
    try:
        config = config_service.build_config(
            data_source=config_data.data_source,
            model_settings=config_data.model_settings,
            model=config_data.model,
            hyperparameter_tuning=config_data.hyperparameter_tuning,
            training=config_data.training,
            experiment_tracking=config_data.experiment_tracking,
            output=config_data.output
        )
        
        yaml_str = config_service.generate_yaml(config)
        config_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"config_{config_id}.yaml"
        file_path = config_service.save_yaml(config, file_name)
        
        return ConfigResponse(
            config_id=config_id,
            yaml_content=yaml_str,
            file_path=file_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate configuration: {str(e)}")



@app.get("/api/config/templates/{section}")
async def get_template(section: str):
    """Get a template configuration for a specific section"""
    try:
        template = config_service.get_template(section)
        return template
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get template: {str(e)}")


@app.post("/api/config/validate")
async def validate_config(config: Dict[str, Any]):
    """Validate a configuration"""
    is_valid = config_service.validate_config(config)
    return {"valid": is_valid}


@app.post("/api/config/upload")
async def upload_config(file: UploadFile = File(...)):
    """Upload a YAML configuration file"""
    try:
        # Generate unique filename
        config_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"config_{config_id}.yaml"
        file_path = os.path.join(CONFIG_DIR, file_name)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load and validate the config
        try:
            config = config_service.load_yaml(file_path)
            is_valid = config_service.validate_config(config)
            
            if not is_valid:
                os.remove(file_path)
                raise HTTPException(status_code=400, detail="Invalid configuration format")
            
            return {
                "config_id": config_id,
                "file_path": file_path,
                "valid": True
            }
        except Exception as e:
            # Clean up file if validation fails
            if os.path.exists(file_path):
                os.remove(file_path)
            raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload configuration: {str(e)}")


@app.get("/api/config/list")
async def list_configs():
    """List all available configuration files"""
    try:
        configs = []
        for filename in os.listdir(CONFIG_DIR):
            if filename.endswith(".yaml"):
                filepath = os.path.join(CONFIG_DIR, filename)
                configs.append({
                    "filename": filename,
                    "path": filepath,
                    "created": datetime.fromtimestamp(os.path.getctime(filepath)).isoformat()
                })
        
        return {"configs": configs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list configurations: {str(e)}")


@app.get("/api/config/download/{config_id}")
async def download_config(config_id: str):
    """Download a configuration file"""
    try:
        filepath = os.path.join(CONFIG_DIR, f"config_{config_id}.yaml")
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Configuration file not found")
        
        return FileResponse(filepath, filename=f"config_{config_id}.yaml")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download configuration: {str(e)}")


@app.post("/api/execute")
async def execute_tasks(request: ExecuteRequest):
    """Execute ML pipeline tasks using a configuration file"""
    try:
        # Find the configuration file
        config_path = os.path.join(CONFIG_DIR, request.config_file)
        
        if not os.path.exists(config_path):
            raise HTTPException(status_code=404, detail="Configuration file not found")
        
        # Initialize task executor
        executor = TaskExecutor(config_path)
        
        # Run specified tasks or all tasks
        success = executor.run_pipeline(request.tasks)
        
        return {
            "success": success,
            "config_file": request.config_file,
            "tasks": request.tasks if request.tasks else "all available tasks"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute tasks: {str(e)}")


@app.get("/api/available-tasks/{config_id}")
async def get_available_tasks(config_id: str):
    """Get available tasks for a specific configuration"""
    try:
        # Find the configuration file
        config_file = f"config_{config_id}.yaml"
        config_path = os.path.join(CONFIG_DIR, config_file)
        
        if not os.path.exists(config_path):
            raise HTTPException(status_code=404, detail="Configuration file not found")
        
        # Initialize task executor to get tasks
        executor = TaskExecutor(config_path)
        tasks = executor._get_available_tasks()
        
        return {
            "config_id": config_id,
            "available_tasks": tasks
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get available tasks: {str(e)}")