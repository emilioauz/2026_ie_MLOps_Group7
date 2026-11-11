"""
validate.py  [Step 3: VALIDATE, the data quality gate]
======================================================
Check the cleaned data before any expensive step runs. Validate, never fix: every failed
rule stops the pipeline with a clear ValueError that names the rule and the rows affected.

Input     Clean DataFrame + the rules (column names and ranges come from config.yaml via main.py)
Output    True when every rule passes
Called by main.py (training data and inference data)
Next      the train/validation/test split in main.py

Rules (each one is skipped when its parameter is None)
    required_columns   every listed column is present
    target_column      target present, no missing values, only 0/1 (training data)
    id_column          no missing and no duplicate IDs
    positive_columns   numeric and strictly greater than 0
    duration_range     values of duration_column inside [low, high]
    launch_year_range  values of year_column inside [low, high]
    forbidden_columns  none of these columns is present (leakage)

Tests     tests/test_validate.py
"""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def _check_frame(df: pd.DataFrame, required_columns: Sequence[str]) -> None:
    """Raise if ``df`` is not a non-empty DataFrame holding every required column."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"validate_dataframe expects a pandas DataFrame, got {type(df).__name__}")
    if df.empty:
        raise ValueError("DataFrame is empty: nothing to validate")
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _check_target(df: pd.DataFrame, target_column: str) -> None:
    """Raise if the target is absent, has missing values, or holds values other than 0/1."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing")
    n_missing = int(df[target_column].isna().sum())
    if n_missing:
        raise ValueError(f"Target column '{target_column}' has {n_missing} missing value(s)")
    if not df[target_column].isin([0, 1]).all():
        raise ValueError(f"Target column '{target_column}' must hold only 0 and 1")


def _check_ids(df: pd.DataFrame, id_column: str) -> None:
    """Raise if the ID column is absent, has missing values or repeats an ID."""
    if id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' is missing")
    if df[id_column].isna().any():
        raise ValueError(f"ID column '{id_column}' has missing values")
    duplicates = int(df[id_column].duplicated().sum())
    if duplicates:
        raise ValueError(f"ID column '{id_column}' has {duplicates} duplicate value(s)")


def _check_positive(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise if a listed column is not numeric or has a value <= 0 (missing is allowed)."""
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' is missing, cannot check it is positive")
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Column '{column}' must be numeric")
        bad = int((df[column] <= 0).sum())
        if bad:
            raise ValueError(f"{bad} row(s) have {column} <= 0")


def _check_range(
    df: pd.DataFrame, column: str | None, value_range: Sequence[float], rule: str
) -> None:
    """Raise if values of ``column`` fall outside ``value_range`` (missing values are allowed)."""
    if column is None:
        raise ValueError(
            f"{rule} is set but its column name is not: pass the matching *_column argument"
        )
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is missing, cannot check {rule}")
    if len(value_range) != 2:
        raise ValueError(f"{rule} must be [low, high], got {list(value_range)}")
    low, high = value_range
    values = df[column].dropna()
    outside = int((~values.between(low, high)).sum())
    if outside:
        raise ValueError(f"{outside} row(s) have {column} outside [{low}, {high}]")


def _check_forbidden(df: pd.DataFrame, forbidden_columns: Sequence[str]) -> None:
    """Raise if any leakage column is still present."""
    present = [column for column in forbidden_columns if column in df.columns]
    if present:
        raise ValueError(f"Leakage columns must not be present: {present}")


# Signature fixed by the course contract; the keyword-only rules are the team's extras.
# pylint: disable-next=too-many-arguments
def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
    target_column: str | None = None,
    *,
    duration_range: Sequence[float] | None = None,
    launch_year_range: Sequence[float] | None = None,
    forbidden_columns: Sequence[str] | None = None,
    id_column: str | None = None,
    positive_columns: Sequence[str] | None = None,
    duration_column: str | None = None,
    year_column: str | None = None,
) -> bool:
    """Check the data against every rule that is switched on; never change it.

    Args:
        df: Cleaned data.
        required_columns: Columns that must be present.
        target_column: Target name for training data; None for inference data.
        duration_range: ``[low, high]`` for ``duration_column``; None skips the rule.
        launch_year_range: ``[low, high]`` for ``year_column``; None skips the rule.
        forbidden_columns: Leakage columns that must be absent; None skips the rule.
        id_column: Column that must be unique and complete; None skips the rule.
        positive_columns: Columns that must be numeric and > 0; None skips the rule.
        duration_column: Column checked by ``duration_range``.
        year_column: Column checked by ``launch_year_range``.

    Returns:
        True when every active rule passes.

    Raises:
        TypeError: If ``df`` is not a DataFrame.
        ValueError: On the first rule that fails, naming the rule and the rows affected.
    """
    _check_frame(df, required_columns)
    if target_column is not None:
        _check_target(df, target_column)
    if id_column is not None:
        _check_ids(df, id_column)
    if positive_columns:
        _check_positive(df, positive_columns)
    if duration_range is not None:
        _check_range(df, duration_column, duration_range, "duration_range")
    if launch_year_range is not None:
        _check_range(df, year_column, launch_year_range, "launch_year_range")
    if forbidden_columns:
        _check_forbidden(df, forbidden_columns)
    logger.info("Validation passed: %d rows x %d columns", len(df), df.shape[1])
    return True
