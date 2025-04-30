import logging
logger = logging.getLogger(__name__)
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
import json
from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    # Store JSON data as Text in SQLite
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship - one user can have many projects
    projects = relationship("Project", back_populates="owner", cascade="all, delete-orphan")
    
    @property
    def meta_data(self):
        """Convert stored JSON string to Python dict"""
        if self.metadata_json:
            return json.loads(self.metadata_json) if self.metadata_json else {}
        return {}
    
    @meta_data.setter
    def meta_data(self, value):
        """Convert Python dict to JSON string for storage"""
        if value is not None:
            self.metadata_json = json.dumps(value)
        else:
            self.metadata_json = None


class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, index=True)
    description = Column(String, nullable=True)
    owner_id = Column(String, ForeignKey("users.id"), nullable=False)
    # Store JSON data as Text in SQLite
    config_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship - each project belongs to one user
    owner = relationship("User", back_populates="projects")
    
    @property
    def config(self):
        """Convert stored JSON string to Python dict"""
        if self.config_json:
            return json.loads(self.config_json)
        return {}
    
    @config.setter
    def config(self, value):
        """Convert Python dict to JSON string for storage"""
        if value is not None:
            self.config_json = json.dumps(value)
        else:
            self.config_json = None

# Add to models.py

class MLExperiment(Base):
    __tablename__ = "ml_experiments"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    data_source = Column(String, nullable=False)  # Path or URL to data
    data_format = Column(String, nullable=False)  # CSV, Parquet, etc.
    target_column = Column(String, nullable=True)
    task_type = Column(String, nullable=False)  # Classification or Regression
    # Store features configuration as JSON
    features_config_json = Column(JSON, nullable=True)
    # Store experiment results as JSON
    results_json = Column(JSON, nullable=True)
    # Store model config as JSON
    model_config_json = Column(JSON, nullable=True)
    # Store hyperparameters as JSON
    hyperparams_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship - each experiment belongs to one project
    project = relationship("Project", backref="experiments")
    
    @property
    def features_config(self) -> list[dict]:
        # Always return a list, so Pydantic sees a proper List
        return self.features_config_json or []

    @features_config.setter
    def features_config(self, value: list[dict]):
        self.features_config_json = value
            
    @property
    def results(self):
        """Convert stored JSON string to Python dict"""
        if self.results_json:
            return json.loads(self.results_json)
        return {}
    
    @results.setter
    def results(self, value):
        """Convert Python dict to JSON string for storage"""
        if value is not None:
            self.results_json = json.dumps(value)
        else:
            self.results_json = None
            
    @property
    def model_configuration(self):
        """Convert stored JSON string to Python dict"""
        if self.model_config_json:
            return json.loads(self.model_config_json)
        return {}
    
    @model_configuration.setter
    def model_configuration(self, value):
        """Convert Python dict to JSON string for storage"""
        if value is not None:
            self.model_config_json = json.dumps(value)
        else:
            self.model_config_json = None
            
    @property
    def hyperparams(self):
        """Convert stored JSON string to Python dict"""
        if self.hyperparams_json:
            return json.loads(self.hyperparams_json)
        return {}
    
    @hyperparams.setter
    def hyperparams(self, value):
        """Convert Python dict to JSON string for storage"""
        if value is not None:
            self.hyperparams_json = json.dumps(value)
        else:
            self.hyperparams_json = None