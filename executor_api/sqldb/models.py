from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
import json
from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, unique=True, index=True, nullable=False)
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
            return json.loads(self.metadata_json)
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

