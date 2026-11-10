"""Tests for src/features.py: the unfitted feature recipe."""
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from src import features


@pytest.fixture
def df() -> pd.DataFrame:
    """Twelve rows with numbers, a missing value and a rare category."""
    return pd.DataFrame({
        "goal": [100.0, 2000.0, 500.0, np.nan, 50.0, 800.0, 300.0, 1200.0, 90.0, 400.0, 700.0, 60.0],
        "days": [30, 45, 10, 60, 30, 20, 15, 30, 40, 35, 25, 50],
        "cat": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b", "rare", None],
    })


def test_unit_returns_unfitted_column_transformer():
    recipe = features.get_feature_preprocessor(["goal", "days"], ["cat"], log_columns=["goal"])
    assert isinstance(recipe, ColumnTransformer)
    with pytest.raises(NotFittedError):
        check_is_fitted(recipe)


def test_unit_log_columns_are_split_from_plain_numeric():
    recipe = features.get_feature_preprocessor(["goal", "days"], ["cat"], log_columns=["goal"])
    columns = {name: cols for name, _, cols in recipe.transformers}
    assert columns == {"log": ["goal"], "num": ["days"], "cat": ["cat"]}
    assert "log" in recipe.transformers[0][1].named_steps


def test_unit_no_log_and_no_categorical():
    recipe = features.get_feature_preprocessor(["goal", "days"], [])
    assert [name for name, _, _ in recipe.transformers] == ["num"]


def test_unit_only_categorical():
    recipe = features.get_feature_preprocessor([], ["cat"], rare_category_minimum=3)
    assert [name for name, _, _ in recipe.transformers] == ["cat"]
    assert recipe.transformers[0][1].named_steps["onehot"].min_frequency == 3


def test_unit_transform_imputes_logs_and_groups_rare(df):
    recipe = features.get_feature_preprocessor(["goal", "days"], ["cat"], log_columns=["goal"],
                                               rare_category_minimum=3)
    out = recipe.fit_transform(df)
    out = out.toarray() if hasattr(out, "toarray") else out
    names = list(recipe.get_feature_names_out())
    assert not np.isnan(out).any()
    assert "missingindicator_goal" in names
    assert "cat_infrequent_sklearn" in names          # 'rare' and 'Unknown' are grouped
    assert {"cat_a", "cat_b"} <= set(names)


def test_unit_empty_lists_raise():
    with pytest.raises(ValueError, match="both empty"):
        features.get_feature_preprocessor([], [])


def test_unit_overlapping_lists_raise():
    with pytest.raises(ValueError, match="both numeric and categorical"):
        features.get_feature_preprocessor(["goal", "cat"], ["cat"])


def test_unit_log_column_not_numeric_raises():
    with pytest.raises(ValueError, match="log_columns"):
        features.get_feature_preprocessor(["days"], ["cat"], log_columns=["goal"])


def test_unit_bad_rare_minimum_raises():
    with pytest.raises(ValueError, match="rare_category_minimum"):
        features.get_feature_preprocessor(["goal"], ["cat"], rare_category_minimum=0)


def test_integration_recipe_handles_unseen_category_after_fit(df):
    recipe = features.get_feature_preprocessor(["goal", "days"], ["cat"], log_columns=["goal"],
                                               rare_category_minimum=3)
    recipe.fit(df)
    new = pd.DataFrame({"goal": [250.0], "days": [30], "cat": ["never_seen"]})
    out = recipe.transform(new)
    out = out.toarray() if hasattr(out, "toarray") else out
    assert out.shape == (1, len(recipe.get_feature_names_out()))
