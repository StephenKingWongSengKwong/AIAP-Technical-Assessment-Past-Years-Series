"""
Data preprocessing and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, List
import joblib


class Preprocessor:
    def __init__(self, config: dict):
        """Initialize preprocessor with configuration.

        Args:
            config (dict): Preprocessing configuration
        """
        self.config = config
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        
    def preprocess(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Compatibility wrapper: fit if preprocessors are not yet fitted, else transform.

        For first-time use it will fit scalers/encoders on the provided data. On subsequent
        calls it will use the fitted transformers.
        """
        # If no scalers/encoders fitted, call fit
        if not self.scalers and not self.encoders:
            return self.fit(data)
        else:
            df = data.copy()
            df = self._handle_missing_values(df)
            target = df[self.config['target_column']] if self.config.get('target_column') in df.columns else None
            features = df.drop(columns=[self.config['target_column']]) if target is not None else df
            features = self._process_features(features)
            return features, target
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to config.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        return df
    
    def _process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process features according to their types.

        Args:
            df (pd.DataFrame): Input features

        Returns:
            pd.DataFrame: Processed features
        """
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = df.select_dtypes(include=['object']).columns
        
        # Scale numeric features
        for col in numeric_features:
            if col not in self.scalers:
                # If scaler not fitted, it's an error for transform phase
                raise RuntimeError(f"Scaler for column '{col}' not found. Call fit() before transform().")
            else:
                df[col] = self.scalers[col].transform(df[[col]])
        
        # Encode categorical features
        for col in categorical_features:
            if col not in self.encoders:
                # If encoder not fitted, it's an error for transform phase
                raise RuntimeError(f"Encoder for column '{col}' not found. Call fit() before transform().")
            else:
                df[col] = self.encoders[col].transform(df[col])
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features based on EDA insights.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Dataframe with additional features
        """
        # Add feature engineering based on EDA insights
        # To be implemented based on EDA findings
        return df

    def fit(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Fit preprocessors on training data and return processed X,y.

        This will fit scalers and encoders and store them in the object for later transform.
        """
        df = data.copy()
        df = self._handle_missing_values(df)

        # Extract target
        target = df[self.config['target_column']]
        features = df.drop(columns=[self.config['target_column']])

        # Fit and transform features
        features = self._fit_process_features(features)
        return features, target

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using already-fitted preprocessors."""
        df = data.copy()
        df = self._handle_missing_values(df)
        features = df.drop(columns=[self.config['target_column']]) if self.config.get('target_column') in df.columns else df
        return self._process_features(features)

    def _fit_process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit scalers/encoders on the dataframe and return transformed dataframe."""
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = df.select_dtypes(include=['object']).columns

        # Fit numeric scalers
        for col in numeric_features:
            self.scalers[col] = StandardScaler()
            df[col] = self.scalers[col].fit_transform(df[[col]])

        # Fit categorical encoders
        for col in categorical_features:
            self.encoders[col] = LabelEncoder()
            df[col] = self.encoders[col].fit_transform(df[col])

        return df

    def save(self, path: str) -> None:
        """Persist the preprocessor (scalers/encoders and config) to disk."""
        joblib.dump({'config': self.config, 'scalers': self.scalers, 'encoders': self.encoders}, path)

    @classmethod
    def load(cls, path: str):
        """Load a persisted preprocessor from disk and return a Preprocessor instance."""
        obj = joblib.load(path)
        p = cls(obj.get('config', {}))
        p.scalers = obj.get('scalers', {})
        p.encoders = obj.get('encoders', {})
        return p