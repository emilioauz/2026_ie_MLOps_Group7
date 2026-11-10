"""Tests for src/infer.py: predictions on new rows with a fitted pipeline."""
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import infer


class FixedProbaModel:  # pylint: disable=too-few-public-methods
    """Fake classifier that returns probabilities we choose, so results are exact."""

    feature_names_in_ = np.array(["x"])

    def __init__(self, proba):
        self.proba = np.asarray(proba, dtype=float)

    def predict_proba(self, X):  # pylint: disable=invalid-name,unused-argument
        return np.column_stack([1 - self.proba, self.proba])


@pytest.fixture
def data() -> tuple[pd.DataFrame, pd.Series]:
    """Twenty rows where the target follows x, with a non-default index."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=20), "z": rng.normal(size=20)}, index=range(100, 120))
    y = pd.Series((X["x"] > 0).astype(int), index=X.index)
    return X, y


@pytest.fixture
def fitted(data) -> Pipeline:
    """A small logistic pipeline fitted on the fixture data."""
    X, y = data
    return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())]).fit(X, y)


def test_unit_prediction_only_keeps_index(fitted, data):
    X, _ = data
    result = infer.run_inference(fitted, X)
    assert list(result.columns) == ["prediction"]
    assert result.index.equals(X.index)
    assert set(result["prediction"].unique()) <= {0, 1}


def test_unit_include_proba_adds_probability(fitted, data):
    X, _ = data
    result = infer.run_inference(fitted, X, include_proba=True)
    assert list(result.columns) == ["prediction", "probability"]
    assert result["probability"].between(0, 1).all()


def test_unit_threshold_changes_predictions():
    X = pd.DataFrame({"x": [1, 2, 3, 4]})
    model = FixedProbaModel([0.9, 0.6, 0.4, 0.1])
    strict = infer.run_inference(model, X, threshold=0.8)
    loose = infer.run_inference(model, X, threshold=0.3)
    assert strict["prediction"].tolist() == [1, 0, 0, 0]
    assert loose["prediction"].tolist() == [1, 1, 1, 0]


def test_unit_extra_columns_are_ignored(fitted, data):
    X, _ = data
    with_id = X.assign(ID=range(len(X)))[["ID", "z", "x"]]
    result = infer.run_inference(fitted, with_id, include_proba=True)
    expected = infer.run_inference(fitted, X, include_proba=True)
    pd.testing.assert_frame_equal(result, expected)


def test_unit_regression_model_predicts_values(data):
    X, _ = data
    y = 2 * X["x"] + 1
    model = LinearRegression().fit(X, y)
    result = infer.run_inference(model, X)
    assert result["prediction"].to_numpy() == pytest.approx(y.to_numpy())


def test_unit_model_without_feature_names_still_predicts():
    class NoNamesModel:  # pylint: disable=too-few-public-methods
        def predict(self, X):  # pylint: disable=invalid-name
            return np.full(len(X), 7.0)

    result = infer.run_inference(NoNamesModel(), pd.DataFrame({"a": [1, 2]}))
    assert result["prediction"].tolist() == [7.0, 7.0]


def test_unit_empty_input_raises(fitted):
    with pytest.raises(ValueError, match="non-empty DataFrame"):
        infer.run_inference(fitted, pd.DataFrame())


def test_unit_not_a_dataframe_raises(fitted, data):
    X, _ = data
    with pytest.raises(ValueError, match="non-empty DataFrame"):
        infer.run_inference(fitted, X.to_numpy())


@pytest.mark.parametrize("threshold", [0.0, 1.0, 1.5])
def test_unit_bad_threshold_raises(fitted, data, threshold):
    X, _ = data
    with pytest.raises(ValueError, match="threshold"):
        infer.run_inference(fitted, X, threshold=threshold)


def test_unit_missing_columns_raise(fitted, data):
    X, _ = data
    with pytest.raises(ValueError, match="missing columns"):
        infer.run_inference(fitted, X.drop(columns=["z"]))


def test_unit_include_proba_without_predict_proba_raises(data):
    X, _ = data
    model = LinearRegression().fit(X, X["x"])
    with pytest.raises(TypeError, match="predict_proba"):
        infer.run_inference(model, X, include_proba=True)


def test_integration_train_output_can_predict(data):
    from src import train  # pylint: disable=import-outside-toplevel

    X, y = data
    recipe = ColumnTransformer([("num", StandardScaler(), ["x", "z"])])
    model = train.train_model(X, y, recipe, "classification")
    new_rows = X.iloc[:5]
    result = infer.run_inference(model, new_rows, include_proba=True, threshold=0.29)
    assert list(result.columns) == ["prediction", "probability"]
    assert result.index.equals(new_rows.index)
    assert result["prediction"].tolist() == (result["probability"] >= 0.29).astype(int).tolist()
