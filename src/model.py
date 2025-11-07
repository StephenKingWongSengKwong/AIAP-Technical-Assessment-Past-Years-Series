"""
Model training and evaluation
"""

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np
from typing import Dict, Any
import joblib


class ModelTrainer:
    def __init__(self, config: dict):
        """Initialize model trainer with configuration.

        Args:
            config (dict): Model configuration
        """
        self.config = config
        self.model = None
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train the model and return performance metrics.

        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Target variable

        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        # Initialize model based on config
        self.model = self._get_model()
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model, 
            X, 
            y, 
            cv=self.config.get('cv_folds', 5),
            scoring=self.config.get('scoring', 'accuracy')
        )
        
        # Train final model
        self.model.fit(X, y)
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model on test data.

        Args:
            X (pd.DataFrame): Test features
            y (pd.Series): True labels

        Returns:
            Dict[str, float]: Performance metrics
        """
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_prob)
        }
    
    def save_model(self, path: str) -> None:
        """Save trained model to disk.

        Args:
            path (str): Path to save model
        """
        joblib.dump(self.model, path)
    
    def load_model(self, path: str) -> None:
        """Load trained model from disk.

        Args:
            path (str): Path to load model from
        """
        self.model = joblib.load(path)
    
    def _get_model(self):
        """Initialize model based on configuration."""
        model_type = self.config['model_type'].lower()
        
        if model_type == 'randomforest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**self.config.get('model_params', {}))
        
        elif model_type == 'xgboost':
            import xgboost as xgb
            return xgb.XGBClassifier(**self.config.get('model_params', {}))
        
        elif model_type == 'lightgbm':
            import lightgbm as lgb
            return lgb.LGBMClassifier(**self.config.get('model_params', {}))
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")