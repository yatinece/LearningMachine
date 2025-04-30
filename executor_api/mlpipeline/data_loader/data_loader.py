# data_loader.py
import pandas as pd
import os
import io
import boto3
from fastapi import UploadFile, File, HTTPException
import logging

# Configure logging

logger = logging.getLogger("data_loader")

class DataLoader:
    """Handles loading data from various sources and formats"""
    
    @staticmethod
    async def load_file(file: UploadFile = File(...)):
        """Load data from an uploaded file (CSV, Excel, Parquet)"""
        try:
            content = await file.read()
            file_ext = os.path.splitext(file.filename)[1].lower()
            
            if file_ext == '.csv':
                return pd.read_csv(io.BytesIO(content))
            elif file_ext == '.parquet':
                return pd.read_parquet(io.BytesIO(content))
            elif file_ext in ['.xls', '.xlsx']:
                return pd.read_excel(io.BytesIO(content))
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_ext}")
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
    @staticmethod
    def load_from_path(file_path: str):
        """Load data from a file path on the server"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.csv':
                return pd.read_csv(file_path)
            elif file_ext == '.parquet':
                return pd.read_parquet(file_path)
            elif file_ext in ['.xls', '.xlsx']:
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
        except Exception as e:
            logger.error(f"Error loading file from path: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading file: {str(e)}")
    
    @staticmethod
    def load_from_s3(bucket_name: str, object_key: str, aws_access_key_id=None, aws_secret_access_key=None):
        """Load data from an S3 bucket"""
        try:
            # Create S3 client
            session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
            s3_client = session.client('s3')
            
            # Get file object
            response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
            file_content = response['Body'].read()
            
            # Determine file type
            file_ext = os.path.splitext(object_key)[1].lower()
            
            if file_ext == '.csv':
                return pd.read_csv(io.BytesIO(file_content))
            elif file_ext == '.parquet':
                return pd.read_parquet(io.BytesIO(file_content))
            elif file_ext in ['.xls', '.xlsx']:
                return pd.read_excel(io.BytesIO(file_content))
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
        except Exception as e:
            logger.error(f"Error loading file from S3: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading from S3: {str(e)}")