"""
Main pipeline execution script
"""

import argparse
import os
from src.config import load_config, get_env_config, merge_configs
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.model import ModelTrainer
from sklearn.model_selection import train_test_split
import logging
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(config_path: str):
    """Main pipeline execution function.

    Args:
        config_path (str): Path to configuration file
    """
    # Load configuration
    yaml_config = load_config(config_path)
    env_config = get_env_config()
    config = merge_configs(yaml_config, env_config)
    
    logger.info("Configuration loaded")
    
    # Initialize components
    data_loader = DataLoader(config['database_path'])
    preprocessor = Preprocessor(config)
    model_trainer = ModelTrainer(config)
    
    # Load data
    logger.info("Loading data...")
    with data_loader as loader:
        data = loader.load_data(config['data_query'])
    
    # Preprocess data
    logger.info("Preprocessing data...")
    X, y = preprocessor.preprocess(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['test_size'],
        random_state=config['random_state']
    )
    
    # Train model
    logger.info(f"Training {config['model_type']} model...")
    cv_metrics = model_trainer.train(X_train, y_train)
    logger.info(f"Cross-validation metrics: {cv_metrics}")
    
    # Evaluate model
    logger.info("Evaluating model...")
    test_metrics = model_trainer.evaluate(X_test, y_test)
    logger.info(f"Test metrics: {test_metrics}")
    
    # Save model
    if config.get('save_model', True):
        model_path = os.path.join(config['model_dir'], f"{config['model_type']}_model.joblib")
        model_trainer.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

    # Persist metrics (CV + test) to metrics file
    try:
        metrics_record = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'model_type': config['model_type'],
            'model_params': config.get('model_params', {}),
            'cv_metrics': cv_metrics,
            'test_metrics': test_metrics,
            'data_query': config.get('data_query')
        }

        # ensure model_dir exists
        os.makedirs(config.get('model_dir', 'models'), exist_ok=True)
        metrics_path = os.path.join(config.get('model_dir', 'models'), 'metrics.json')

        # append the new record to a JSON array in metrics file
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as mf:
                try:
                    metrics_list = json.load(mf)
                except Exception:
                    metrics_list = []
        else:
            metrics_list = []

        metrics_list.append(metrics_record)

        with open(metrics_path, 'w', encoding='utf-8') as mf:
            json.dump(metrics_list, mf, indent=2)

        logger.info(f"Metrics saved to {metrics_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hotel No-Show Prediction Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/params.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    main(args.config)