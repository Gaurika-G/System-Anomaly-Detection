import os
import pandas as pd
import yaml
from src.logger import get_logger

logger = get_logger("ingestion")


def load_config() -> dict:
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def validate_dataframe(df: pd.DataFrame, name: str) -> None:
    if df.empty:
        raise ValueError(f"{name} is empty after loading.")

    logger.info(f"{name} — shape: {df.shape}")
    logger.info(f"{name} — columns: {list(df.columns)}")

    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        logger.warning(f"{name} — missing values found:\n{missing_cols}")
    else:
        logger.info(f"{name} — no missing values found")

    if "label" in df.columns:
        label_counts = df["label"].value_counts()
        logger.info(f"{name} — label distribution:\n{label_counts}")


def ingest_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config()
    train_path = config["data"]["raw_train"]
    test_path = config["data"]["raw_test"]

    logger.info("Starting data ingestion...")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training file not found at '{train_path}'.\n"
            f"Please download the UNSW-NB15 dataset from Kaggle and place it in the data/ folder.\n"
            f"Run: kagglehub.dataset_download('alextamboli/unsw-nb15')"
        )

    logger.info(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    validate_dataframe(train_df, "Training set")

    if os.path.exists(test_path):
        logger.info(f"Loading test data from: {test_path}")
        test_df = pd.read_csv(test_path)
        validate_dataframe(test_df, "Test set")
    else:
        logger.warning(f"Test file not found at '{test_path}'. Splitting train set 80/20.")
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df["label"])
        logger.info(f"Created train split: {train_df.shape}, test split: {test_df.shape}")

    return train_df, test_df
