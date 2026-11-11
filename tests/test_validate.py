
"""Tests for src/validate.py: every rule passes on good data and raises on bad data.

Tiny synthetic campaigns only; never the real Kickstarter CSV.
"""

import pandas as pd
import pytest

from src.validate import validate_dataframe

REQUIRED = ["ID", "goal_usd", "duration_days", "launch_year", "main_category"]

RULES = {
    "id_column": "ID",
    "positive_columns": ["goal_usd"],
    "duration_range": [1, 92],
    "duration_column": "duration_days",
    "launch_year_range": [2009, 2030],
    "year_column": "launch_year",
    "forbidden_columns": ["pledged", "backers"],
}


def _campaigns() -> pd.DataFrame:
    """Three valid campaigns with a 0/1 target."""
    return pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "goal_usd": [10000.0, 25000.0, 50000.0],
            "duration_days": [30, 45, 60],
            "launch_year": [2015, 2016, 2017],
            "main_category": ["Games", "Music", "Design"],
            "success": [1, 0, 1],
        }
    )


# ---------- unit: happy paths ----------

def test_unit_valid_training_data_returns_true():
    assert validate_dataframe(_campaigns(), REQUIRED, "success", **RULES) is True


def test_unit_contract_call_without_extras_returns_true():
    assert validate_dataframe(_campaigns(), REQUIRED) is True


def test_unit_inference_data_without_target_returns_true():
    df = _campaigns().drop(columns="success")
    assert validate_dataframe(df, REQUIRED, None, **RULES) is True


def test_unit_missing_values_in_range_columns_are_allowed():
    df = _campaigns()
    df.loc[0, "duration_days"] = None
    assert validate_dataframe(df, REQUIRED, "success", **RULES) is True


def test_unit_input_is_not_modified():
    df = _campaigns()
    before = df.copy()
    validate_dataframe(df, REQUIRED, "success", **RULES)
    pd.testing.assert_frame_equal(df, before)


# ---------- unit: every raise ----------

def test_unit_not_a_dataframe_raises():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        validate_dataframe([1, 2], REQUIRED)


def test_unit_empty_dataframe_raises():
    with pytest.raises(ValueError, match="empty"):
        validate_dataframe(_campaigns().iloc[0:0], REQUIRED)


def test_unit_missing_required_column_raises():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(
            _campaigns().drop(columns="main_category"), REQUIRED
        )


def test_unit_missing_target_column_raises():
    with pytest.raises(ValueError, match="Target column 'success' is missing"):
        validate_dataframe(_campaigns().drop(columns="success"), REQUIRED, "success")


def test_unit_target_with_missing_value_raises():
    df = _campaigns()
    df["success"] = [1, None, 0]
    with pytest.raises(ValueError, match="1 missing value"):
        validate_dataframe(df, REQUIRED, "success")


def test_unit_target_not_binary_raises():
    df = _campaigns()
    df.loc[0, "success"] = 2
    with pytest.raises(ValueError, match="only 0 and 1"):
        validate_dataframe(df, REQUIRED, "success")


def test_unit_id_column_absent_raises():
    with pytest.raises(ValueError, match="ID column 'campaign_id' is missing"):
        validate_dataframe(_campaigns(), REQUIRED, id_column="campaign_id")


def test_unit_id_with_missing_value_raises():
    df = _campaigns()
    df["ID"] = [1, None, 3]
    with pytest.raises(ValueError, match="missing values"):
        validate_dataframe(df, REQUIRED, id_column="ID")


def test_unit_duplicate_id_raises():
    df = _campaigns()
    df["ID"] = [1, 1, 3]
    with pytest.raises(ValueError, match="1 duplicate"):
        validate_dataframe(df, REQUIRED, id_column="ID")


def test_unit_positive_column_absent_raises():
    with pytest.raises(ValueError, match="'pledge' is missing"):
        validate_dataframe(_campaigns(), REQUIRED, positive_columns=["pledge"])


def test_unit_positive_column_not_numeric_raises():
    with pytest.raises(ValueError, match="must be numeric"):
        validate_dataframe(
            _campaigns(), REQUIRED, positive_columns=["main_category"]
        )


def test_unit_non_positive_value_raises():
    df = _campaigns()
    df.loc[1, "goal_usd"] = 0
    with pytest.raises(ValueError, match="1 row"):
        validate_dataframe(df, REQUIRED, positive_columns=["goal_usd"])


def test_unit_range_without_column_name_raises():
    with pytest.raises(
        ValueError, match="duration_range is set but its column name is not"
    ):
        validate_dataframe(_campaigns(), REQUIRED, duration_range=[1, 92])


def test_unit_range_column_absent_raises():
    with pytest.raises(ValueError, match="'days' is missing"):
        validate_dataframe(
            _campaigns(),
            REQUIRED,
            duration_range=[1, 92],
            duration_column="days",
        )


def test_unit_range_with_wrong_length_raises():
    with pytest.raises(ValueError, match="must be"):
        validate_dataframe(
            _campaigns(),
            REQUIRED,
            launch_year_range=[2009],
            year_column="launch_year",
        )


def test_unit_value_outside_range_raises():
    df = _campaigns()
    df.loc[2, "launch_year"] = 2005
    with pytest.raises(ValueError, match="outside"):
        validate_dataframe(
            df,
            REQUIRED,
            launch_year_range=[2009, 2030],
            year_column="launch_year",
        )


def test_unit_forbidden_column_present_raises():
    df = _campaigns().assign(backers=[10, 20, 30])
    with pytest.raises(ValueError, match="Leakage columns must not be present"):
        validate_dataframe(
            df, REQUIRED, forbidden_columns=["pledged", "backers"]
        )


# ---------- integration ----------

def test_integration_clean_output_passes_validate():
    """Cleaned training output passes validation with the same rules main.py uses."""
    from src.clean_data import clean_dataframe

    raw = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "name": ["A board game", "An album", "A lamp"],
            "main_category": ["Games", "Music", "Design"],
            "state": ["successful", "failed", "successful"],
            "launched": [
                "2015-01-01 10:00:00",
                "2016-03-01 09:00:00",
                "2017-06-01 12:00:00",
            ],
            "deadline": ["2015-01-31", "2016-04-15", "2017-07-31"],
            "usd_goal_real": [10000.0, 25000.0, 50000.0],
            "backers": [5, 0, 9],
        }
    )

    clean = clean_dataframe(
        raw,
        "success",
        state_column="state",
        training_states={"successful": 1, "failed": 0},
        leakage_columns=["backers"],
        columns={
            "launched": "launched",
            "deadline": "deadline",
            "goal": "usd_goal_real",
            "name": "name",
        },
        derived={
            "goal": "goal_usd",
            "duration": "duration_days",
            "month": "launch_month",
            "year": "launch_year",
            "name_words": "name_words",
        },
    )

    assert validate_dataframe(clean, REQUIRED, "success", **RULES) is True
