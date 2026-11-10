"""
train.py  [Step 5: TRAIN]
=========================
Purpose   Bundle the feature recipe + the algorithm into ONE pipeline and fit it
Input     X_train, y_train + unfitted recipe from features.py + model settings
Output    One fitted pipeline (preprocessing + model), saved by main.py as models/model.joblib
Called by main.py          Next   evaluate.py

Why it matters
- .fit() runs on the train split only, so validation and test metrics stay honest
- One artifact for preprocessing + model: what you test is exactly what you deploy
- Model choice and hyperparameters come from config: swap models without rewriting code
- Fixed random seed: same data, same result

Tests     tests/test_train.py
"""
import logging

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

MODELS = {
    "classification": {"logistic_regression": LogisticRegression,
                       "random_forest": RandomForestClassifier},
    "regression": {"linear_regression": LinearRegression,
                   "random_forest": RandomForestRegressor},
}
DEFAULT_MODEL = {"classification": "logistic_regression", "regression": "linear_regression"}


def _check_data(X_train: pd.DataFrame, y_train: pd.Series,  # pylint: disable=invalid-name
                problem_type: str) -> None:
    """Raise ValueError if the training data is empty, mismatched or has a single class."""
    if problem_type not in MODELS:
        raise ValueError(f"problem_type must be one of {sorted(MODELS)}, got {problem_type!r}")
    if not isinstance(X_train, pd.DataFrame) or X_train.empty:
        raise ValueError("X_train must be a non-empty DataFrame")
    if len(y_train) == 0:
        raise ValueError("y_train is empty")
    if len(X_train) != len(y_train):
        raise ValueError(f"X_train has {len(X_train)} rows but y_train has {len(y_train)}")
    if problem_type == "classification" and pd.Series(y_train).nunique() < 2:
        raise ValueError("y_train has a single class: classification needs at least two")


def _build_model(problem_type: str, model_params: dict | None, random_state: int):
    """Create the unfitted estimator named in model_params, seeded when it takes a seed."""
    model_params = model_params or {}
    model_type = model_params.get("type", DEFAULT_MODEL[problem_type])
    if model_type not in MODELS[problem_type]:
        raise ValueError(f"Unknown model type {model_type!r} for {problem_type}; "
                         f"choose from {sorted(MODELS[problem_type])}")
    params = dict(model_params.get("params") or {})
    estimator_class = MODELS[problem_type][model_type]
    if "random_state" in estimator_class().get_params():
        params.setdefault("random_state", random_state)
    try:
        return model_type, estimator_class(**params)
    except TypeError as error:
        raise ValueError(f"Bad params for {model_type}: {error}") from error


# Signature fixed by the course contract: X_train name and six arguments.
# pylint: disable-next=invalid-name,too-many-arguments,too-many-positional-arguments
def train_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer,
                problem_type: str, model_params: dict | None = None,
                random_state: int = 42) -> Pipeline:
    """Build the model from settings and fit preprocessing + model on the training split.

    Args:
        X_train: Training features.
        y_train: Training target, same length as X_train.
        preprocessor: Unfitted recipe from features.get_feature_preprocessor (copied, not changed).
        problem_type: "classification" or "regression".
        model_params: {"type": <model name>, "params": {...}}. None = the default model
            (logistic_regression for classification, linear_regression for regression).
        random_state: Seed used when params do not set one.

    Returns:
        A fitted Pipeline with steps "preprocess" and "model".

    Raises:
        ValueError: Unknown problem or model type, bad params, empty or mismatched data,
            or a single class in a classification target.
    """
    _check_data(X_train, y_train, problem_type)
    model_type, estimator = _build_model(problem_type, model_params, random_state)
    pipeline = Pipeline([("preprocess", clone(preprocessor)), ("model", estimator)])
    logger.info("Training %s on %s rows, %s columns", model_type, len(X_train), X_train.shape[1])
    pipeline.fit(X_train, y_train)
    logger.info("Training done")
    return pipeline
