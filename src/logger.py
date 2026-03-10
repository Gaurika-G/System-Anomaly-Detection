import logging
import os
import yaml
from datetime import datetime


def get_logger(name: str) -> logging.Logger:
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    log_cfg = config.get("logging", {})
    log_level = getattr(logging, log_cfg.get("level", "INFO"))
    log_file = log_cfg.get("log_file", "logs/pipeline.log")

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
