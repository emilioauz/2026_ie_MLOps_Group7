
import pandas as pd
import pytest

from src.validate import validate_dataframe


@pytest.fixture
def valid_data():
    """Create a small valid campaign DataFrame for testing."""
    return pd.DataFrame({
        "ID": [101, 102],
        "name": ["Campaign A", "Campaign B"],
        "category": ["Technology", "Art"],
        "main_category": ["Technology", "Art"],
        "country": ["US", "GB"],
        "goal_usd": [5000.0, 8000.0],
        "duration_days": [30, 45],
        "launch_month": [5, 8],
        "launch_year": [2017, 2018],
        "name_words": [2, 2],
    })


def test_valid_dataframe_returns_true(valid_data):
    assert validate_dataframe(
        valid_data,
        duration_range=(1, 92),
        launch_year_range=(2009, 2030),
    ) is True


def test_missing_required_column_raises_error(valid_data):
    valid_data = valid_data.drop(columns=["country"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(valid_data)


def test_duplicate_ids_raise_error(valid_data):
    valid_data.loc[1, "ID"] = 101

    with pytest.raises(ValueError, match="Duplicate campaign IDs"):
        validate_dataframe(valid_data)


def test_missing_id_raises_error(valid_data):
    valid_data.loc[0, "ID"] = None

    with pytest.raises(ValueError, match="ID column contains missing"):
        validate_dataframe(valid_data)


def test_invalid_duration_raises_error(valid_data):
    valid_data.loc[0, "duration_days"] = 100

    with pytest.raises(ValueError, match="duration_days"):
        validate_dataframe(valid_data, duration_range=(1, 92))


def test_invalid_launch_year_raises_error(valid_data):
    valid_data.loc[0, "launch_year"] = 2000

    with pytest.raises(ValueError, match="launch_year"):
        validate_dataframe(valid_data, launch_year_range=(2009, 2030))


def test_zero_goal_raises_error(valid_data):
    valid_data.loc[0, "goal_usd"] = 0

    with pytest.raises(ValueError, match="goal_usd"):
        validate_dataframe(valid_data)


def test_negative_goal_raises_error(valid_data):
    valid_data.loc[0, "goal_usd"] = -100

    with pytest.raises(ValueError, match="goal_usd"):
        validate_dataframe(valid_data)


def test_leakage_column_raises_error(valid_data):
    valid_data["backers"] = [10, 20]

    with pytest.raises(ValueError, match="Forbidden leakage columns"):
        validate_dataframe(
            valid_data,
            forbidden_columns=["pledged", "backers", "usd pledged"],
        )


def test_invalid_target_raises_error(valid_data):
    valid_data["success"] = [1, 2]

    with pytest.raises(ValueError, match="only 0 and 1"):
        validate_dataframe(valid_data)


def test_missing_target_raises_error_when_target_is_present(valid_data):
    valid_data["success"] = [1, None]

    with pytest.raises(ValueError, match="missing values"):
        validate_dataframe(valid_data)


def test_inference_data_without_target_is_valid(valid_data):
    assert "success" not in valid_data.columns

    assert validate_dataframe(
        valid_data,
        duration_range=(1, 92),
        launch_year_range=(2009, 2030),
    ) is True


def test_empty_dataframe_raises_error(valid_data):
    empty_data = valid_data.iloc[0:0]

    with pytest.raises(ValueError, match="must not be empty"):
        validate_dataframe(empty_data)
