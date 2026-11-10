"""
features.py  [Step 4: FEATURE RECIPE]
=====================================
Purpose   Define HOW raw columns become model inputs (impute, log, scale, one-hot encode)
Input     Lists of column names per transformation (from config)
Output    An UNFITTED scikit-learn ColumnTransformer (a recipe, not transformed data)
Called by main.py
Used by   train.py (fits it) and infer.py (applies it, inside the saved model)

Why it matters
- Never call .fit() here. Fitting happens in train.py on the train split only: no data leakage
- One recipe for training AND inference, so both process data identically
- Swap features through config lists, not code edits

Tests     tests/test_features.py
"""
import logging

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def _numeric_steps(log: bool) -> Pipeline:
    """Median imputation (with a missing flag), optional log1p, then standard scaling."""
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if log:
        steps.append(("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def _categorical_steps(rare_category_minimum: int) -> Pipeline:
    """Fill missing with 'Unknown', then one-hot encode.

    Rare and unseen categories share one 'infrequent' column.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=rare_category_minimum)),
    ])


def _check_columns(numeric_columns: list[str], categorical_columns: list[str],
                   log_columns: list[str], rare_category_minimum: int) -> None:
    """Raise ValueError if the column lists or the rare-category minimum are unusable."""
    if not numeric_columns and not categorical_columns:
        raise ValueError("numeric_columns and categorical_columns are both empty: "
                         "nothing to transform")
    overlap = set(numeric_columns) & set(categorical_columns)
    if overlap:
        raise ValueError(f"Columns listed as both numeric and categorical: {sorted(overlap)}")
    not_numeric = set(log_columns) - set(numeric_columns)
    if not_numeric:
        raise ValueError(f"log_columns must also be numeric_columns: {sorted(not_numeric)}")
    if rare_category_minimum < 1:
        raise ValueError(f"rare_category_minimum must be at least 1, got {rare_category_minimum}")


def get_feature_preprocessor(numeric_columns: list[str], categorical_columns: list[str],
                             log_columns: list[str] | None = None,
                             rare_category_minimum: int = 100) -> ColumnTransformer:
    """Return the unfitted feature recipe.

    Args:
        numeric_columns: Numeric inputs (median-imputed with a missing flag, then scaled).
        categorical_columns: Categorical inputs (missing -> 'Unknown', then one-hot encoded).
        log_columns: Numeric columns that also get log1p before scaling. None = no log transform.
        rare_category_minimum: Categories seen fewer times than this in training share
            one 'infrequent' column.

    Returns:
        An unfitted ColumnTransformer. train.py fits it.

    Raises:
        ValueError: Both lists empty, a column in both lists, a log column that is not numeric,
            or rare_category_minimum below 1.
    """
    numeric_columns = list(numeric_columns or [])
    categorical_columns = list(categorical_columns or [])
    log_columns = list(log_columns or [])
    _check_columns(numeric_columns, categorical_columns, log_columns, rare_category_minimum)

    plain_numeric = [col for col in numeric_columns if col not in log_columns]
    transformers = []
    if log_columns:
        transformers.append(("log", _numeric_steps(log=True), log_columns))
    if plain_numeric:
        transformers.append(("num", _numeric_steps(log=False), plain_numeric))
    if categorical_columns:
        transformers.append(("cat", _categorical_steps(rare_category_minimum), categorical_columns))

    logger.info("Feature recipe: %s log, %s numeric, %s categorical columns",
                len(log_columns), len(plain_numeric), len(categorical_columns))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)
