"""
Data preprocessing and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, List


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
        """Preprocess the data according to configuration.

        Args:
            data (pd.DataFrame): Raw input data

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Processed features and target
        """
        df = data.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Extract target
        target = df[self.config['target_column']]
        features = df.drop(columns=[self.config['target_column']])
        
        # Process features
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
                self.scalers[col] = StandardScaler()
                df[col] = self.scalers[col].fit_transform(df[[col]])
            else:
                df[col] = self.scalers[col].transform(df[[col]])
        
        # Encode categorical features
        for col in categorical_features:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col])
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