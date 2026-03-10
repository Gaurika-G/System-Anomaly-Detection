import pandas as pd
import numpy as np
import pickle
import yaml
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.logger import get_logger

logger = get_logger("preprocessing")


def load_config() -> dict:
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def drop_unnecessary_columns(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    cols_to_drop = [c for c in config["preprocessing"]["drop_columns"] if c in df.columns]
    if cols_to_drop:
        logger.info(f"Dropping columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    return df


def handle_missing_values(df: pd.DataFrame, config: dict, medians: dict = None) -> tuple[pd.DataFrame, dict]:
    strategy = config["preprocessing"]["missing_value_strategy"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target = config["preprocessing"]["target_column"]
    if target in numeric_cols:
        numeric_cols.remove(target)

    if medians is None:
        medians = {}
        for col in numeric_cols:
            medians[col] = df[col].median()

    missing_before = df.isnull().sum().sum()
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(medians.get(col, 0))

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].fillna("unknown")

    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values: {missing_before} before → {missing_after} after imputation")
    return df, medians


def encode_categorical_columns(
    df: pd.DataFrame,
    config: dict,
    encoders: dict = None
) -> tuple[pd.DataFrame, dict]:
    cat_cols = [c for c in config["features"]["categorical_columns"] if c in df.columns]
    is_training = encoders is None

    if is_training:
        encoders = {}
        logger.info(f"Fitting label encoders on: {cat_cols}")
    else:
        logger.info(f"Applying existing label encoders to: {cat_cols}")

    for col in cat_cols:
        if is_training:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            known_classes = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known_classes else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    return df, encoders


def scale_numeric_columns(
    df: pd.DataFrame,
    config: dict,
    scaler: StandardScaler = None
) -> tuple[pd.DataFrame, StandardScaler]:
    target = config["preprocessing"]["target_column"]
    num_cols = [
        c for c in config["features"]["numeric_columns"]
        if c in df.columns and c != target
    ]

    is_training = scaler is None

    if is_training:
        logger.info(f"Fitting StandardScaler on {len(num_cols)} numeric columns")
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        logger.info(f"Applying existing StandardScaler to {len(num_cols)} numeric columns")
        df[num_cols] = scaler.transform(df[num_cols])

    return df, scaler


def preprocess(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config()
    encoder_path = config["data"]["encoder_path"]
    os.makedirs(os.path.dirname(encoder_path), exist_ok=True)

    logger.info("Starting preprocessing on training data...")
    train_df = drop_unnecessary_columns(train_df.copy(), config)
    test_df = drop_unnecessary_columns(test_df.copy(), config)

    train_df, medians = handle_missing_values(train_df, config)
    test_df, _ = handle_missing_values(test_df, config, medians=medians)

    train_df, encoders = encode_categorical_columns(train_df, config)
    test_df, _ = encode_categorical_columns(test_df, config, encoders=encoders)

    train_df, scaler = scale_numeric_columns(train_df, config)
    test_df, _ = scale_numeric_columns(test_df, config, scaler=scaler)

    artifacts = {"encoders": encoders, "scaler": scaler, "medians": medians}
    with open(encoder_path, "wb") as f:
        pickle.dump(artifacts, f)
    logger.info(f"Saved preprocessing artifacts to: {encoder_path}")

    logger.info(f"Preprocessing complete — train: {train_df.shape}, test: {test_df.shape}")
    return train_df, test_df
