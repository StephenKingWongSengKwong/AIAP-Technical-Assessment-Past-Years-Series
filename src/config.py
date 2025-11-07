"""
Pipeline configuration management
"""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path (str): Path to config file

    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_env_config() -> Dict[str, Any]:
    """Get configuration from environment variables.

    Returns:
        Dict[str, Any]: Configuration from environment variables
    """
    config = {}
    
    # Model configuration
    config['model_type'] = os.getenv('MODEL_TYPE', 'randomforest')
    
    # Data processing
    config['target_column'] = os.getenv('TARGET_COLUMN', 'no_show')
    
    # Training parameters
    config['test_size'] = float(os.getenv('TEST_SIZE', '0.2'))
    config['random_state'] = int(os.getenv('RANDOM_STATE', '42'))
    
    return config


def merge_configs(yaml_config: Dict[str, Any], env_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge YAML and environment configurations.

    Args:
        yaml_config (Dict[str, Any]): Configuration from YAML
        env_config (Dict[str, Any]): Configuration from environment

    Returns:
        Dict[str, Any]: Merged configuration
    """
    # Environment variables take precedence
    return {**yaml_config, **env_config}