"""
clean_data.py  [Step 2: CLEAN]
==============================
Port of the notebook's clean_campaigns(): stateless fixes only, so the same input always
gives the same output and nothing is learned from the data (learning happens after the split).

Two modes
    training   target_column given: keep rows with a final outcome, apply the business scope,
               build the 0/1 target
    inference  target_column None: keep every row, so every new campaign gets scored

Every rule is switched on by a parameter from config.yaml (via main.py); None means "rule off",
so a call with only the contract inputs still works.

columns / derived  (config: cleaning.columns, cleaning.derived)
    columns maps a role to a RAW column name:   launched, deadline, goal, name, country
    derived maps a role to the NEW column name: goal, duration, month, year, name_words

Tests     tests/test_clean_data.py
"""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def _check_input(df_raw: pd.DataFrame) -> None:
    """Raise if the input is not a non-empty DataFrame."""
    if not isinstance(df_raw, pd.DataFrame):
        raise TypeError(f"clean_dataframe expects a pandas DataFrame, got {type(df_raw).__name__}")
    if df_raw.empty:
        raise ValueError("Raw DataFrame is empty: nothing to clean")


def _require(df: pd.DataFrame, columns: Sequence[str], purpose: str) -> None:
    """Raise a clear error naming any column that ``purpose`` needs but ``df`` lacks."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s) {missing} needed for {purpose}")


def _derive_dates(
    out: pd.DataFrame,
    columns: Mapping[str, str],
    derived: Mapping[str, str],
    first_valid_year: int | None,
) -> None:
    """Add duration, launch month and year; launch dates before first_valid_year go missing."""
    launched = pd.to_datetime(out[columns["launched"]], errors="coerce")
    if first_valid_year is not None:
        launched = launched.where(launched.dt.year >= first_valid_year)
    if "deadline" in columns:
        deadline = pd.to_datetime(out[columns["deadline"]], errors="coerce")
        out[derived["duration"]] = (deadline - launched.dt.normalize()).dt.days
    out[derived["month"]] = launched.dt.month
    out[derived["year"]] = launched.dt.year


def _derive_columns(
    df: pd.DataFrame,
    columns: Mapping[str, str],
    derived: Mapping[str, str] | None,
    first_valid_year: int | None,
    min_goal_usd: float | None,
) -> pd.DataFrame:
    """Build the model inputs the notebook builds, then drop the raw columns they replace."""
    if not derived:
        raise ValueError(
            "'columns' is set but 'derived' is not: give the new column names in cleaning.derived"
        )
    _require(df, list(columns.values()), "cleaning.columns")
    out = df.copy()
    if "launched" in columns:
        _derive_dates(out, columns, derived, first_valid_year)
    if "goal" in columns:
        goal = out[columns["goal"]]
        out[derived["goal"]] = (
            goal.where(goal >= min_goal_usd) if min_goal_usd is not None else goal
        )
    if "name" in columns:
        out[columns["name"]] = out[columns["name"]].fillna("")
        out[derived["name_words"]] = out[columns["name"]].str.split().str.len()
    consumed = [columns[role] for role in ("launched", "deadline", "goal") if role in columns]
    return out.drop(columns=[column for column in consumed if column not in derived.values()])


def _training_rows(
    df: pd.DataFrame,
    target_column: str,
    state_column: str | None,
    training_states: Mapping[str, int] | None,
) -> pd.DataFrame:
    """Keep rows with a final outcome and build the 0/1 target from the state column."""
    if state_column is None:
        _require(df, [target_column], "the target")
        return df[df[target_column].notna()].copy()
    if not training_states:
        raise ValueError(
            "state_column is set but training_states is empty: map each final state to 0 or 1"
        )
    _require(df, [state_column], "the target")
    out = df[df[state_column].isin(list(training_states))].copy()
    out[target_column] = out[state_column].map(training_states).astype(int)
    return out


# Signature fixed by the course contract; the keyword-only rules are the team's extras.
# pylint: disable-next=too-many-arguments
def clean_dataframe(
    df_raw: pd.DataFrame,
    target_column: str | None = None,
    *,
    state_column: str | None = None,
    training_states: Mapping[str, int] | None = None,
    first_valid_year: int | None = None,
    broken_country: str | None = None,
    min_goal_usd: float | None = None,
    min_investable_goal_usd: float | None = None,
    leakage_columns: Sequence[str] | None = None,
    columns: Mapping[str, str] | None = None,
    derived: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Apply the notebook's cleaning rules; the input is never modified.

    Args:
        df_raw: Raw data from load_raw_data.
        target_column: Target name (training mode); None keeps every row (inference mode).
        state_column: Column holding the final outcome, mapped to the target by training_states.
        training_states: e.g. ``{"successful": 1, "failed": 0}``; other states are dropped.
        first_valid_year: Launch dates before this year become missing.
        broken_country: A known bad country code that becomes missing.
        min_goal_usd: Goals below this become missing.
        min_investable_goal_usd: Training rows with a smaller goal are dropped (business scope).
        leakage_columns: Columns only known after the outcome; dropped if present.
        columns: Raw column names by role (launched, deadline, goal, name, country).
        derived: New column names by role (goal, duration, month, year, name_words).

    Returns:
        Clean data with a fresh RangeIndex.

    Raises:
        TypeError: If ``df_raw`` is not a DataFrame.
        ValueError: If it is empty, a needed column is missing, or a rule is half-configured.
    """
    _check_input(df_raw)
    out = df_raw.drop(columns=list(leakage_columns or []), errors="ignore")
    if columns:
        out = _derive_columns(out, columns, derived, first_valid_year, min_goal_usd)
        if broken_country is not None and "country" in columns:
            out[columns["country"]] = out[columns["country"]].where(
                out[columns["country"]] != broken_country
            )
    if target_column is not None:
        out = _training_rows(out, target_column, state_column, training_states)
        if min_investable_goal_usd is not None:
            goal_column = (derived or {}).get("goal")
            _require(
                out, [goal_column] if goal_column else ["derived.goal"], "min_investable_goal_usd"
            )
            out = out[out[goal_column] >= min_investable_goal_usd]
    logger.info(
        "Cleaned data: %d of %d rows kept (%s mode)",
        len(out),
        len(df_raw),
        "training" if target_column is not None else "inference",
    )
    return out.reset_index(drop=True)
