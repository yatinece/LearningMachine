# feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import logging

logger = logging.getLogger("feature_engineering")

class FeatureEngineering:
    """Handles feature detection and transformation for ML datasets"""
    
    @staticmethod
    def detect_feature_types(df: pd.DataFrame):
        """Automatically detect categorical, numerical, and potential ID features"""
        numerical_features = []
        categorical_features = []
        id_features = []
        
        # Examine each column
        for col in df.columns:
            # Check if it looks like an ID column
            if col.lower().endswith('id') or col.lower().startswith('id_'):
                id_features.append(col)
                continue
                
            # Check data type
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if it's actually categorical (few unique values)
                if df[col].nunique() < min(20, df.shape[0] * 0.05):
                    categorical_features.append(col)
                else:
                    numerical_features.append(col)
            else:
                # Text/string columns are treated as categorical
                categorical_features.append(col)
        
        logger.info(f"Detected {len(numerical_features)} numerical features, "
                   f"{len(categorical_features)} categorical features, "
                   f"{len(id_features)} ID features")
        
        return {
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "id_features": id_features
        }
    
    @staticmethod
    def transform_features(df: pd.DataFrame, feature_config=None):
        """Transform features based on configuration or auto-detection"""
        # If no config provided, auto-detect feature types
        if not feature_config:
            feature_config = FeatureEngineering.detect_feature_types(df)
            
        result_df = df.copy()
        encoders = {}
        
        # Apply transformations
        # Label encoding for categorical features
        for cat_feature in feature_config.get("label_encode", []):
            if cat_feature in result_df.columns:
                logger.info(f"Label encoding feature: {cat_feature}")
                le = LabelEncoder()
                result_df[f"{cat_feature}_encoded"] = le.fit_transform(result_df[cat_feature].astype(str))
                encoders[f"{cat_feature}_le"] = le
        
        # One-hot encoding for categorical features
        for cat_feature in feature_config.get("onehot_encode", []):
            if cat_feature in result_df.columns:
                logger.info(f"One-hot encoding feature: {cat_feature}")
                ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_features = ohe.fit_transform(result_df[[cat_feature]])
                
                # Create new DataFrame with encoded features
                feature_names = [f"{cat_feature}_{val}" for val in ohe.categories_[0]]
                encoded_df = pd.DataFrame(encoded_features, columns=feature_names)
                
                # Concatenate with original DataFrame
                result_df = pd.concat([result_df, encoded_df], axis=1)
                encoders[f"{cat_feature}_ohe"] = ohe
        
        return result_df, encoders
    
    @staticmethod
    def prepare_training_data(df: pd.DataFrame, target_column, feature_config=None):
        """Prepare data for training by applying transformations and splitting features/target"""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
            
        # Auto-detect feature types if not provided
        if not feature_config:
            detected_features = FeatureEngineering.detect_feature_types(df)
            feature_config = {
                "numerical_features": detected_features["numerical_features"],
                "categorical_features": detected_features["categorical_features"],
                "id_features": detected_features["id_features"],
                "onehot_encode": detected_features["categorical_features"],
                "label_encode": []  # Default is to one-hot encode
            }
        
        # Extract target and drop from features
        y = df[target_column].copy()
        X = df.drop(columns=[target_column])
        
        # Apply transformations
        X_transformed, encoders = FeatureEngineering.transform_features(X, feature_config)
        
        # Drop ID features from training data
        if "id_features" in feature_config:
            X_transformed = X_transformed.drop(columns=feature_config["id_features"], errors='ignore')
        
        return X_transformed, y, encoders, feature_config