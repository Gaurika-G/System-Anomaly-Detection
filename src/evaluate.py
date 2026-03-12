import pandas as pd
import numpy as np
import pickle
import os
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from src.logger import get_logger

logger = get_logger("evaluate")


def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_model(config):
    model_path = config["data"]["model_path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. Please run train_model.py first."
        )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Loaded model from: {model_path}")
    return model


def get_predictions(model, X):
    raw_preds = model.predict(X)
    binary_preds = np.where(raw_preds == -1, 1, 0)

    raw_scores = model.score_samples(X)
    anomaly_scores = -raw_scores

    return binary_preds, anomaly_scores


def print_evaluation_report(y_true, y_pred, anomaly_scores, split_name):
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATION RESULTS -- {split_name}")
    logger.info(f"{'='*60}")

    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred)
    roc_auc   = roc_auc_score(y_true, anomaly_scores)

    logger.info(f"  Precision : {precision:.4f}  (of flagged flows, how many were real attacks?)")
    logger.info(f"  Recall    : {recall:.4f}  (of all attacks, how many did we catch?)")
    logger.info(f"  F1 Score  : {f1:.4f}  (balance of precision and recall)")
    logger.info(f"  ROC-AUC   : {roc_auc:.4f}  (overall detection ability, 1.0 = perfect)")

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    logger.info(f"\n  Confusion Matrix:")
    logger.info(f"                       Predicted Normal   Predicted Attack")
    logger.info(f"  Actual Normal  :      {tn:>12,}        {fp:>12,}   <- False Alarms")
    logger.info(f"  Actual Attack  :      {fn:>12,}        {tp:>12,}   <- Caught Attacks")

    logger.info(f"\n  In plain English:")
    logger.info(f"  Caught  : {tp:,} real attacks out of {tp+fn:,} total")
    logger.info(f"  Missed  : {fn:,} attacks slipped through")
    logger.info(f"  Alarms  : {fp:,} false alarms on normal traffic")

    logger.info(f"\n  Full Classification Report:")
    report = classification_report(y_true, y_pred, target_names=["Normal", "Attack"])
    for line in report.split("\n"):
        logger.info(f"    {line}")

    return {
        "split": split_name,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def save_results(results, config):
    results_path = config["data"]["results_path"]
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    pd.DataFrame(results).to_csv(results_path, index=False)
    logger.info(f"Results saved to: {results_path}")


def run_evaluation(model=None, X_train=None, y_train=None, X_test=None, y_test=None):
    logger.info("=" * 60)
    logger.info("PHASE 2 -- MODEL EVALUATION")
    logger.info("=" * 60)

    config = load_config()

    if model is None:
        model = load_model(config)

    if X_test is None:
        from src.train import load_processed_data, split_features_labels
        target = config["preprocessing"]["target_column"]
        train_df, test_df = load_processed_data(config)
        X_train, y_train = split_features_labels(train_df, target)
        X_test, y_test = split_features_labels(test_df, target)

    all_results = []

    logger.info("Evaluating on TRAINING SET...")
    train_preds, train_scores = get_predictions(model, X_train)
    train_results = print_evaluation_report(y_train, train_preds, train_scores, "Training Set")
    all_results.append(train_results)

    logger.info("\nEvaluating on TEST SET...")
    test_preds, test_scores = get_predictions(model, X_test)
    test_results = print_evaluation_report(y_test, test_preds, test_scores, "Test Set")
    all_results.append(test_results)

    save_results(all_results, config)

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"  Test F1      : {test_results['f1']}")
    logger.info(f"  Test ROC-AUC : {test_results['roc_auc']}")
    logger.info("=" * 60)

    return all_results