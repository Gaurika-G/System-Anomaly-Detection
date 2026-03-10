import pandas as pd
import numpy as np
import yaml
from src.logger import get_logger

logger = get_logger("feature_engineering")


def load_config() -> dict:
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def add_byte_ratio(df: pd.DataFrame) -> pd.DataFrame:
    if "sbytes" in df.columns and "dbytes" in df.columns:
        total = df["sbytes"] + df["dbytes"]
        df["byte_ratio"] = df["sbytes"] / total.replace(0, np.nan)
        df["byte_ratio"] = df["byte_ratio"].fillna(0.5)
        logger.info("Added feature: byte_ratio (src bytes / total bytes)")
    return df


def add_packet_ratio(df: pd.DataFrame) -> pd.DataFrame:
    if "spkts" in df.columns and "dpkts" in df.columns:
        total = df["spkts"] + df["dpkts"]
        df["packet_ratio"] = df["spkts"] / total.replace(0, np.nan)
        df["packet_ratio"] = df["packet_ratio"].fillna(0.5)
        logger.info("Added feature: packet_ratio (src packets / total packets)")
    return df


def add_bytes_per_second(df: pd.DataFrame) -> pd.DataFrame:
    if "sbytes" in df.columns and "dur" in df.columns:
        df["src_bytes_per_sec"] = df["sbytes"] / df["dur"].replace(0, np.nan)
        df["src_bytes_per_sec"] = df["src_bytes_per_sec"].fillna(0)
        logger.info("Added feature: src_bytes_per_sec")

    if "dbytes" in df.columns and "dur" in df.columns:
        df["dst_bytes_per_sec"] = df["dbytes"] / df["dur"].replace(0, np.nan)
        df["dst_bytes_per_sec"] = df["dst_bytes_per_sec"].fillna(0)
        logger.info("Added feature: dst_bytes_per_sec")

    return df


def add_jitter_ratio(df: pd.DataFrame) -> pd.DataFrame:
    if "sjit" in df.columns and "djit" in df.columns:
        df["jitter_ratio"] = df["sjit"] / (df["djit"] + 1e-9)
        df["jitter_ratio"] = df["jitter_ratio"].clip(upper=1000)
        logger.info("Added feature: jitter_ratio (src jitter / dst jitter)")
    return df


def add_ttl_difference(df: pd.DataFrame) -> pd.DataFrame:
    if "sttl" in df.columns and "dttl" in df.columns:
        df["ttl_diff"] = (df["sttl"] - df["dttl"]).abs()
        logger.info("Added feature: ttl_diff (absolute TTL difference)")
    return df


def add_connection_symmetry(df: pd.DataFrame) -> pd.DataFrame:
    if "spkts" in df.columns and "dpkts" in df.columns:
        diff = (df["spkts"] - df["dpkts"]).abs()
        total = (df["spkts"] + df["dpkts"]).replace(0, np.nan)
        df["conn_asymmetry"] = diff / total
        df["conn_asymmetry"] = df["conn_asymmetry"].fillna(0)
        logger.info("Added feature: conn_asymmetry (how one-sided the connection is)")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Starting feature engineering — input shape: {df.shape}")

    df = add_byte_ratio(df)
    df = add_packet_ratio(df)
    df = add_bytes_per_second(df)
    df = add_jitter_ratio(df)
    df = add_ttl_difference(df)
    df = add_connection_symmetry(df)

    new_cols = ["byte_ratio", "packet_ratio", "src_bytes_per_sec",
                "dst_bytes_per_sec", "jitter_ratio", "ttl_diff", "conn_asymmetry"]
    added = [c for c in new_cols if c in df.columns]
    logger.info(f"Feature engineering complete — added {len(added)} features, output shape: {df.shape}")

    return df
