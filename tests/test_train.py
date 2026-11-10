"""Tests for src/train.py: building and fitting the model pipeline."""
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from src import train

RF = {"type": "random_forest", "params": {"n_estimators": 5, "max_depth": 3}}


@pytest.fixture
def data() -> tuple[pd.DataFrame, pd.Series]:
    """Twenty rows where the target follows x."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=20), "z": rng.normal(size=20)})
    y = pd.Series((X["x"] > 0).astype(int))
    return X, y


@pytest.fixture
def recipe() -> ColumnTransformer:
    """A tiny unfitted recipe that scales both columns."""
    return ColumnTransformer([("num", StandardScaler(), ["x", "z"])])


def test_unit_default_classification_is_fitted_logistic(data, recipe):
    X, y = data
    model = train.train_model(X, y, recipe, "classification")
    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["model"], LogisticRegression)
    check_is_fitted(model)
    assert model.predict_proba(X).shape == (20, 2)


def test_unit_random_forest_uses_params_and_seed(data, recipe):
    X, y = data
    model = train.train_model(X, y, recipe, "classification", RF, random_state=7)
    forest = model.named_steps["model"]
    assert isinstance(forest, RandomForestClassifier)
    assert (forest.n_estimators, forest.max_depth, forest.random_state) == (5, 3, 7)


def test_unit_seed_in_params_wins(data, recipe):
    X, y = data
    params = {"type": "random_forest", "params": {"n_estimators": 5, "random_state": 1}}
    model = train.train_model(X, y, recipe, "classification", params, random_state=7)
    assert model.named_steps["model"].random_state == 1


def test_unit_same_seed_same_result(data, recipe):
    X, y = data
    first = train.train_model(X, y, recipe, "classification", RF).predict_proba(X)
    second = train.train_model(X, y, recipe, "classification", RF).predict_proba(X)
    np.testing.assert_array_equal(first, second)


def test_unit_regression_models(data, recipe):
    X, _ = data
    y = X["x"] * 2
    assert isinstance(train.train_model(X, y, recipe, "regression").named_steps["model"],
                      LinearRegression)
    forest = train.train_model(X, y, recipe, "regression", RF).named_steps["model"]
    assert isinstance(forest, RandomForestRegressor)


def test_unit_preprocessor_passed_in_stays_unfitted(data, recipe):
    X, y = data
    train.train_model(X, y, recipe, "classification")
    with pytest.raises(NotFittedError):
        check_is_fitted(recipe)


def test_unit_unknown_problem_type_raises(data, recipe):
    X, y = data
    with pytest.raises(ValueError, match="problem_type"):
        train.train_model(X, y, recipe, "clustering")


def test_unit_unknown_model_type_raises(data, recipe):
    X, y = data
    with pytest.raises(ValueError, match="Unknown model type"):
        train.train_model(X, y, recipe, "classification", {"type": "linear_regression"})


def test_unit_bad_params_raise(data, recipe):
    X, y = data
    with pytest.raises(ValueError, match="Bad params"):
        train.train_model(X, y, recipe, "classification",
                          {"type": "random_forest", "params": {"not_a_param": 1}})


def test_unit_empty_x_raises(data, recipe):
    _, y = data
    with pytest.raises(ValueError, match="X_train"):
        train.train_model(pd.DataFrame(), y, recipe, "classification")


def test_unit_empty_y_raises(data, recipe):
    X, _ = data
    with pytest.raises(ValueError, match="y_train is empty"):
        train.train_model(X, pd.Series(dtype=int), recipe, "classification")


def test_unit_mismatched_lengths_raise(data, recipe):
    X, y = data
    with pytest.raises(ValueError, match="rows"):
        train.train_model(X, y.iloc[:10], recipe, "classification")


def test_unit_single_class_raises(data, recipe):
    X, _ = data
    with pytest.raises(ValueError, match="single class"):
        train.train_model(X, pd.Series([1] * 20), recipe, "classification")


def test_integration_features_recipe_trains_and_predicts():
    from src import features  # pylint: disable=import-outside-toplevel

    rng = np.random.default_rng(1)
    X = pd.DataFrame({
        "goal": rng.uniform(100, 10000, size=30),
        "days": rng.integers(10, 60, size=30).astype(float),
        "cat": rng.choice(["a", "b", "c"], size=30),
    })
    X.loc[0, "goal"] = np.nan
    y = pd.Series((X["days"] > 35).astype(int))
    recipe = features.get_feature_preprocessor(["goal", "days"], ["cat"], log_columns=["goal"],
                                               rare_category_minimum=5)
    model = train.train_model(X, y, recipe, "classification", RF)
    new = pd.DataFrame({"goal": [500.0], "days": [20.0], "cat": ["unseen"]})
    proba = model.predict_proba(new)
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0
