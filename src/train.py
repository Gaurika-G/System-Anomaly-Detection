import pandas as pd
import numpy as np
import pickle
import os
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from src.logger import get_logger

logger = get_logger("train")


def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_processed_data(config):
    train_path = config["data"]["processed_train"]
    test_path = config["data"]["processed_test"]

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Processed data not found at '{train_path}'.\n"
            f"Please run pipeline.py first."
        )

    logger.info(f"Loading processed training data from: {train_path}")
    train_df = pd.read_parquet(train_path)

    logger.info(f"Loading processed test data from: {test_path}")
    test_df = pd.read_parquet(test_path)

    logger.info(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")
    return train_df, test_df


def split_features_labels(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def get_normal_traffic_only(X_train, y_train):
    normal_mask = y_train == 0
    X_normal = X_train[normal_mask]
    logger.info(
        f"Filtered to normal traffic only: {len(X_normal):,} samples "
        f"(excluded {(~normal_mask).sum():,} attack samples from training)"
    )
    return X_normal


def tune_contamination(X_normal, X_train, y_train, config):
    contamination_values = config["model"]["contamination_search"]
    logger.info(f"Searching for best contamination over: {contamination_values}")

    best_contamination = contamination_values[0]
    best_f1 = -1

    for c in contamination_values:
        model = IsolationForest(
            n_estimators=100,
            contamination=c,
            random_state=config["model"]["random_state"],
            n_jobs=-1
        )
        model.fit(X_normal)

        raw_preds = model.predict(X_train)
        preds = np.where(raw_preds == -1, 1, 0)

        f1 = f1_score(y_train, preds)
        logger.info(f"  contamination={c:.2f} -> F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_contamination = c

    logger.info(f"Best contamination: {best_contamination} with F1={best_f1:.4f}")
    return best_contamination


def train_model(X_normal, config, contamination):
    logger.info("Training final Isolation Forest model...")
    logger.info(f"  n_estimators : {config['model']['n_estimators']}")
    logger.info(f"  contamination: {contamination}")
    logger.info(f"  max_samples  : {config['model']['max_samples']}")
    logger.info(f"  random_state : {config['model']['random_state']}")

    model = IsolationForest(
        n_estimators=config["model"]["n_estimators"],
        contamination=contamination,
        max_samples=config["model"]["max_samples"],
        random_state=config["model"]["random_state"],
        n_jobs=-1
    )
    model.fit(X_normal)

    logger.info(f"Training complete on {len(X_normal):,} normal traffic samples")
    return model


def save_model(model, config):
    model_path = config["data"]["model_path"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    size_kb = os.path.getsize(model_path) / 1024
    logger.info(f"Model saved -> {model_path} ({size_kb:.1f} KB)")


def run_training():
    logger.info("=" * 60)
    logger.info("PHASE 2 -- MODEL TRAINING")
    logger.info("=" * 60)

    config = load_config()
    target = config["preprocessing"]["target_column"]

    logger.info("STEP 1: Loading Processed Data")
    train_df, test_df = load_processed_data(config)

    logger.info("STEP 2: Splitting Features and Labels")
    X_train, y_train = split_features_labels(train_df, target)
    X_test, y_test = split_features_labels(test_df, target)
    logger.info(f"Training features: {X_train.shape[1]} columns")

    logger.info("STEP 3: Isolating Normal Traffic for Training")
    X_normal = get_normal_traffic_only(X_train, y_train)

    logger.info("STEP 4: Tuning Contamination Hyperparameter")
    best_contamination = tune_contamination(X_normal, X_train, y_train, config)

    logger.info("STEP 5: Training Final Model")
    model = train_model(X_normal, config, best_contamination)

    logger.info("STEP 6: Saving Model")
    save_model(model, config)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)

    return model, X_train, y_train, X_test, y_test