"""
Simple inference utility: loads saved preprocessor and model, fetches a small sample from the DB,
transforms it and prints predictions and probabilities.

Usage:
    python src/infer.py --config config/params.yaml --n 3
"""
import argparse
import os
import joblib
from src.config import load_config, get_env_config, merge_configs
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor


def main(config_path: str, n: int):
    yaml_config = load_config(config_path)
    env_config = get_env_config()
    config = merge_configs(yaml_config, env_config)

    model_dir = config.get('model_dir', 'models')
    model_path = os.path.join(model_dir, f"{config['model_type']}_model.joblib")
    preproc_path = os.path.join(model_dir, 'preprocessor.joblib')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")
    if not os.path.exists(preproc_path):
        raise FileNotFoundError(f"Preprocessor not found at {preproc_path}. Run training first.")

    model = joblib.load(model_path)
    preproc = Preprocessor.load(preproc_path)

    with DataLoader(config['database_path']) as dl:
        # fetch a small sample
        query = config.get('data_query', 'SELECT * FROM bookings') + f" LIMIT {n}"
        df = dl.load_data(query)

    # Transform sample
    X = preproc.transform(df)

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None

    print("Sample rows (raw):")
    print(df.head())
    print("Predictions:", preds)
    if probs is not None:
        print("Probabilities:", probs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/params.yaml')
    parser.add_argument('--n', type=int, default=3, help='Number of sample rows to fetch')
    args = parser.parse_args()
    main(args.config, args.n)
