import os
import yaml
import pandas as pd
from src.logger import get_logger
from src.ingestion import ingest_data
from src.preprocessing import preprocess
from src.feature_engineering import engineer_features

logger = get_logger("pipeline")


def load_config() -> dict:
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def save_processed_data(train_df: pd.DataFrame, test_df: pd.DataFrame, config: dict) -> None:
    train_out = config["data"]["processed_train"]
    test_out = config["data"]["processed_test"]

    os.makedirs(os.path.dirname(train_out), exist_ok=True)

    train_df.to_parquet(train_out, index=False)
    test_df.to_parquet(test_out, index=False)

    logger.info(f"Saved processed training data to: {train_out}")
    logger.info(f"Saved processed test data to:     {test_out}")


def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("=" * 60)
    logger.info("NETWORK TRAFFIC ANOMALY DETECTION — DATA PIPELINE")
    logger.info("=" * 60)

    config = load_config()

    logger.info("STEP 1: Data Ingestion")
    train_df, test_df = ingest_data()

    logger.info("STEP 2: Preprocessing")
    train_df, test_df = preprocess(train_df, test_df)

    logger.info("STEP 3: Feature Engineering")
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    logger.info("STEP 4: Saving Processed Data")
    save_processed_data(train_df, test_df, config)

    target = config["preprocessing"]["target_column"]
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Training samples : {len(train_df)}")
    logger.info(f"  Test samples     : {len(test_df)}")
    logger.info(f"  Feature columns  : {len([c for c in train_df.columns if c != target])}")
    logger.info(f"  Normal (train)   : {(train_df[target] == 0).sum()}")
    logger.info(f"  Anomalies (train): {(train_df[target] == 1).sum()}")
    logger.info("=" * 60)

    return train_df, test_df


if __name__ == "__main__":
    run_pipeline()
