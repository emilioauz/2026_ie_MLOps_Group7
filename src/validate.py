
"""
validate.py  [Step 3: VALIDATE, the data quality gate]
======================================================
Purpose   Check schema, types and business rules BEFORE expensive steps run
Input     Clean DataFrame + the rules (required columns, allowed target values, ranges)
Output    True if valid, otherwise a clear error that stops the pipeline
Called by main.py (training) and api.py (stage 3, every live request)
Next      the split in main.py

Why it matters
- Garbage in, garbage out: stop bad data at the door, not deep inside training
- Validate, never fix. Fixing belongs in clean_data.py
- Rules live in config, so they are versioned and easy to change

Tests     tests/test_validate.py
"""

import pandas as pd


# Required input columns from config.yaml.
REQUIRED_COLUMNS = [
    "ID",
    "name",
    "category",
    "main_category",
    "country",
    "goal_usd",
    "duration_days",
    "launch_month",
    "launch_year",
    "name_words",
]

ID_COLUMN = "ID"
TARGET_COLUMN = "success"


def validate_dataframe(
    df,
    *,
    duration_range=None,
    launch_year_range=None,
    forbidden_columns=None,
):
    """
    Validate the structure and business rules of a campaign DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to validate.
    duration_range : tuple, optional
        Minimum and maximum allowed campaign duration.
    launch_year_range : tuple, optional
        Minimum and maximum allowed launch year.
    forbidden_columns : list, optional
        Columns that must not appear, such as target leakage columns.

    Returns
    -------
    bool
        True if all applicable validation checks pass.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.
    ValueError
        If one or more validation checks fail.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("DataFrame must not be empty.")

    # 1. Check that all required input columns exist.
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # 2. Check campaign IDs.
    if df[ID_COLUMN].isna().any():
        raise ValueError("ID column contains missing values.")

    if df[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate campaign IDs found in ID column.")

    # 3. Check numeric columns and missing values.
    numeric_columns = [
        "goal_usd",
        "duration_days",
        "launch_year",
    ]

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"{column} must contain numeric values.")

        if df[column].isna().any():
            raise ValueError(f"{column} contains missing values.")

    # 4. Check that the investment goal is positive.
    if (df["goal_usd"] <= 0).any():
        raise ValueError("goal_usd must be greater than 0.")

    # 5. Check campaign duration when a range is provided.
    if duration_range is not None:
        minimum, maximum = duration_range
        if not df["duration_days"].between(
            minimum, maximum
        ).all():
            raise ValueError(
                f"duration_days must be between {minimum} and {maximum}."
            )

    # 6. Check launch year when a range is provided.
    if launch_year_range is not None:
        minimum, maximum = launch_year_range
        if not df["launch_year"].between(
            minimum, maximum
        ).all():
            raise ValueError(
                f"launch_year must be between {minimum} and {maximum}."
            )

    # 7. Check that forbidden (leakage) columns are absent.
    if forbidden_columns is not None:
        present_forbidden = [
            column for column in forbidden_columns
            if column in df.columns
        ]
        if present_forbidden:
            raise ValueError(
                f"Forbidden leakage columns found: {present_forbidden}"
            )

    # 8. Validate the target only when it is supplied.
    # This allows the same validator to accept inference data.
    if TARGET_COLUMN in df.columns:
        if df[TARGET_COLUMN].isna().any():
            raise ValueError("success contains missing values.")

        if not df[TARGET_COLUMN].isin([0, 1]).all():
            raise ValueError("success must contain only 0 and 1.")

    return True
