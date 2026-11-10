"""Tests for src/evaluate.py: scoring a fitted pipeline on unseen data."""
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import evaluate

CLASSIFICATION_KEYS = {"roc_auc", "accuracy", "precision", "recall", "f1", "precision_top_share"}


class FixedProbaModel:  # pylint: disable=too-few-public-methods
    """Fake model that returns probabilities we choose, so results are exact."""

    def __init__(self, proba):
        self.proba = np.asarray(proba, dtype=float)

    def predict_proba(self, X):  # pylint: disable=invalid-name,unused-argument
        return np.column_stack([1 - self.proba, self.proba])


class NoProbaModel:  # pylint: disable=too-few-public-methods
    """Fake model with predict but no predict_proba."""

    def predict(self, X):  # pylint: disable=invalid-name
        return np.zeros(len(X))


@pytest.fixture
def data() -> tuple[pd.DataFrame, pd.Series]:
    """Twenty rows where the target follows x."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=20), "z": rng.normal(size=20)})
    y = pd.Series((X["x"] > 0).astype(int))
    return X, y


@pytest.fixture
def fitted(data) -> Pipeline:
    """A small logistic pipeline fitted on the fixture data."""
    X, y = data
    return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())]).fit(X, y)


def test_unit_classification_returns_all_keys_as_floats(fitted, data):
    X, y = data
    metrics = evaluate.evaluate_model(fitted, X, y, "classification")
    assert set(metrics) == CLASSIFICATION_KEYS
    assert all(isinstance(value, float) for value in metrics.values())
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_unit_precision_top_share_is_exact():
    # 10 rows; top two probabilities are row 0 (positive) and row 1 (negative): 1 hit out of 2
    proba = [0.95, 0.90, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.05]
    y = pd.Series([1, 0, 0, 0, 1, 0, 0, 1, 0, 0])
    X = pd.DataFrame({"x": range(10)})
    model = FixedProbaModel(proba)
    metrics = evaluate.evaluate_model(model, X, y, "classification", shortlist_share=0.2)
    assert metrics["precision_top_share"] == pytest.approx(0.5)


def test_unit_tiny_share_keeps_at_least_one_row():
    proba = [0.9, 0.1, 0.2, 0.3]
    y = pd.Series([1, 0, 1, 0])
    X = pd.DataFrame({"x": range(4)})
    model = FixedProbaModel(proba)
    metrics = evaluate.evaluate_model(model, X, y, "classification", shortlist_share=0.01)
    assert metrics["precision_top_share"] == pytest.approx(1.0)


def test_unit_threshold_changes_predictions():
    proba = [0.9, 0.6, 0.4, 0.1]
    y = pd.Series([1, 1, 0, 0])
    X = pd.DataFrame({"x": range(4)})
    model = FixedProbaModel(proba)
    strict = evaluate.evaluate_model(model, X, y, "classification", threshold=0.8)
    loose = evaluate.evaluate_model(model, X, y, "classification", threshold=0.5)
    assert strict["recall"] == pytest.approx(0.5)
    assert loose["recall"] == pytest.approx(1.0)
    assert strict["roc_auc"] == loose["roc_auc"] == pytest.approx(1.0)


def test_unit_regression_returns_rmse_mae_r2(data):
    X, _ = data
    y = 2 * X["x"] + 1
    model = LinearRegression().fit(X, y)
    metrics = evaluate.evaluate_model(model, X, y, "regression")
    assert set(metrics) == {"rmse", "mae", "r2"}
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["r2"] == pytest.approx(1.0)


def test_unit_unknown_problem_type_raises(fitted, data):
    X, y = data
    with pytest.raises(ValueError, match="problem_type"):
        evaluate.evaluate_model(fitted, X, y, "clustering")


def test_unit_empty_X_raises(fitted):
    with pytest.raises(ValueError, match="non-empty DataFrame"):
        evaluate.evaluate_model(fitted, pd.DataFrame(), pd.Series([1, 0]), "classification")


def test_unit_not_a_dataframe_raises(fitted, data):
    X, y = data
    with pytest.raises(ValueError, match="non-empty DataFrame"):
        evaluate.evaluate_model(fitted, X.to_numpy(), y, "classification")


def test_unit_empty_y_raises(fitted, data):
    X, _ = data
    with pytest.raises(ValueError, match="y_eval is empty"):
        evaluate.evaluate_model(fitted, X, pd.Series([], dtype=int), "classification")


def test_unit_mismatched_lengths_raise(fitted, data):
    X, y = data
    with pytest.raises(ValueError, match="rows"):
        evaluate.evaluate_model(fitted, X, y.iloc[:5], "classification")


@pytest.mark.parametrize("threshold", [0.0, 1.0, 1.5])
def test_unit_bad_threshold_raises(fitted, data, threshold):
    X, y = data
    with pytest.raises(ValueError, match="threshold"):
        evaluate.evaluate_model(fitted, X, y, "classification", threshold=threshold)


@pytest.mark.parametrize("share", [0.0, -0.1, 1.5])
def test_unit_bad_shortlist_share_raises(fitted, data, share):
    X, y = data
    with pytest.raises(ValueError, match="shortlist_share"):
        evaluate.evaluate_model(fitted, X, y, "classification", shortlist_share=share)


def test_unit_model_without_predict_proba_raises_type_error(data):
    X, y = data
    with pytest.raises(TypeError, match="predict_proba"):
        evaluate.evaluate_model(NoProbaModel(), X, y, "classification")


def test_unit_single_class_raises(fitted, data):
    X, _ = data
    with pytest.raises(ValueError, match="single class"):
        evaluate.evaluate_model(fitted, X, pd.Series([1] * len(X)), "classification")


def test_integration_train_output_can_be_evaluated(data):
    from src import train  # pylint: disable=import-outside-toplevel

    X, y = data
    recipe = ColumnTransformer([("num", StandardScaler(), ["x", "z"])])
    model = train.train_model(X, y, recipe, "classification")
    metrics = evaluate.evaluate_model(model, X, y, "classification",
                                      threshold=0.29, shortlist_share=0.10)
    assert set(metrics) == CLASSIFICATION_KEYS
    assert metrics["roc_auc"] > 0.9
