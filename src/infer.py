"""
infer.py  [Step 7: INFER]
=========================
Purpose   Turn new, unseen records into predictions with the saved pipeline
Input     Fitted pipeline + new data (DataFrame, same feature columns, no target)
Output    DataFrame with a fixed schema: "prediction" (+ optional "probability"),
          same index as input
Called by main.py (batch, writes reports/predictions.csv) and api.py (stage 3, live requests)

Why it matters
- No training here: inference is reused unchanged by the API, so batch and live results match
- Fixed output columns: downstream teams can rely on them
- Keep the input index (and ID) so predictions join back to source records

Tests     tests/test_infer.py
"""
import logging

import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _check_inputs(model: Pipeline, X_infer: pd.DataFrame,  # pylint: disable=invalid-name
                  threshold: float) -> list | None:
    """Raise ValueError on unusable input; return the columns the model was fitted on, if known."""
    if not isinstance(X_infer, pd.DataFrame) or X_infer.empty:
        raise ValueError("X_infer must be a non-empty DataFrame")
    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return None
    missing = [column for column in expected if column not in X_infer.columns]
    if missing:
        raise ValueError(f"X_infer is missing columns the model needs: {missing}")
    return list(expected)


def _predict_with_proba(model: Pipeline, features: pd.DataFrame, include_proba: bool,
                        threshold: float) -> pd.DataFrame:
    """Turn class-1 probabilities into 0/1 predictions, keeping the probability if asked."""
    proba = model.predict_proba(features)[:, 1]
    output = pd.DataFrame({"prediction": (proba >= threshold).astype(int)}, index=features.index)
    if include_proba:
        output["probability"] = proba
    return output


def run_inference(model: Pipeline, X_infer: pd.DataFrame,  # pylint: disable=invalid-name
                  include_proba: bool = False, *, threshold: float = 0.5) -> pd.DataFrame:
    """Predict on new rows with a fitted pipeline, keeping the input's index.

    Args:
        model: Fitted pipeline from train.train_model (or loaded from models/model.joblib).
        X_infer: New rows with the feature columns used in training;
            extra columns (e.g. an ID) are ignored.
        include_proba: Also return the probability of class 1 (classifiers only).
        threshold: Probability at or above which a row is predicted as 1 (classifiers only).

    Returns:
        DataFrame with the same index as X_infer and column "prediction",
        plus "probability" when include_proba is True.

    Raises:
        ValueError: Empty or non-DataFrame input, missing feature columns,
            or a threshold out of range.
        TypeError: include_proba is True but the model has no predict_proba.
    """
    expected = _check_inputs(model, X_infer, threshold)
    features = X_infer[expected] if expected is not None else X_infer
    if hasattr(model, "predict_proba"):
        output = _predict_with_proba(model, features, include_proba, threshold)
    elif include_proba:
        raise TypeError("include_proba=True needs a model with predict_proba")
    else:
        output = pd.DataFrame({"prediction": model.predict(features)}, index=features.index)
    logger.info("Predicted %s rows (probability included: %s)", len(output), include_proba)
    return output
