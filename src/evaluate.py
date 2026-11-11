"""
evaluate.py  [Step 6: EVALUATE]
===============================
Purpose   Score the fitted pipeline on data it has never seen
Input     Fitted pipeline + X_eval, y_eval (validation split; test split once, at the end)
Output    Dictionary of plain numbers, e.g. {"pr_auc": 0.61, "roc_auc": 0.84}
Called by main.py          Feeds  stage 3: the quality gate and experiment tracking

Why it matters
- Never score on training data: it hides overfitting
- Metric choice follows the business goal and problem type (set in config)
- Plain numbers are easy to log, compare between runs and turn into a go/no-go decision

Tests     tests/test_evaluate.py
"""
import logging
import math

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

PROBLEM_TYPES = ("classification", "regression")


def _check_inputs(X_eval: pd.DataFrame, y_eval: pd.Series,  # pylint: disable=invalid-name
                  problem_type: str, threshold: float, shortlist_share: float) -> None:
    """Raise ValueError if the problem type, the data or the settings are unusable."""
    if problem_type not in PROBLEM_TYPES:
        raise ValueError(f"problem_type must be one of {list(PROBLEM_TYPES)}, got {problem_type!r}")
    if not isinstance(X_eval, pd.DataFrame) or X_eval.empty:
        raise ValueError("X_eval must be a non-empty DataFrame")
    if len(y_eval) == 0:
        raise ValueError("y_eval is empty")
    if len(X_eval) != len(y_eval):
        raise ValueError(f"X_eval has {len(X_eval)} rows but y_eval has {len(y_eval)}")
    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
    if not 0 < shortlist_share <= 1:
        raise ValueError(f"shortlist_share must be in (0, 1], got {shortlist_share}")


def _precision_top_share(y_true: np.ndarray, proba: np.ndarray, shortlist_share: float) -> float:
    """Share of real positives among the top `shortlist_share` rows ranked by probability."""
    n_top = max(1, math.ceil(len(y_true) * shortlist_share))
    top_rows = np.argsort(-proba, kind="stable")[:n_top]
    return float(y_true[top_rows].mean())


def _classification_metrics(model: Pipeline, X_eval: pd.DataFrame,  # pylint: disable=invalid-name
                            y_true: np.ndarray, threshold: float,
                            shortlist_share: float) -> dict[str, float]:
    """ROC AUC, threshold metrics and shortlist precision for a binary classifier."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("model has no predict_proba: classification metrics need probabilities")
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_eval has a single class: ROC AUC needs both classes")
    proba = model.predict_proba(X_eval)[:, 1]
    predicted = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "precision_top_share": _precision_top_share(y_true, proba, shortlist_share),
    }


def _regression_metrics(model: Pipeline, X_eval: pd.DataFrame,  # pylint: disable=invalid-name
                        y_true: np.ndarray) -> dict[str, float]:
    """RMSE, MAE and R2 for a regressor."""
    predicted = model.predict(X_eval)
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, predicted))),
        "mae": float(mean_absolute_error(y_true, predicted)),
        "r2": float(r2_score(y_true, predicted)),
    }


# Signature fixed by the course contract: X_eval name and six arguments.
# pylint: disable-next=invalid-name,too-many-arguments
def evaluate_model(model: Pipeline, X_eval: pd.DataFrame, y_eval: pd.Series, problem_type: str, *,
                   threshold: float = 0.5, shortlist_share: float = 0.1) -> dict[str, float]:
    """Score a fitted pipeline on unseen data and return plain float metrics.

    Args:
        model: Fitted pipeline from train.train_model.
        X_eval: Features of the evaluation split (never the training split).
        y_eval: True target, same length as X_eval.
        problem_type: "classification" or "regression".
        threshold: Probability at or above which a row is predicted as 1 (classification only).
        shortlist_share: Share of rows, ranked by probability, kept in the shortlist
            (classification only).

    Returns:
        Classification: roc_auc, accuracy, precision, recall, f1, precision_top_share.
        Regression: rmse, mae, r2.

    Raises:
        ValueError: Unknown problem type, empty or mismatched data, a single class, or a threshold
            or shortlist_share out of range.
        TypeError: Classification model without predict_proba.
    """
    _check_inputs(X_eval, y_eval, problem_type, threshold, shortlist_share)
    y_true = np.asarray(y_eval)
    if problem_type == "regression":
        metrics = _regression_metrics(model, X_eval, y_true)
    else:
        metrics = _classification_metrics(model, X_eval, y_true, threshold, shortlist_share)
    logger.info("Evaluated %s rows (%s): %s", len(y_true), problem_type, metrics)
    return metrics
