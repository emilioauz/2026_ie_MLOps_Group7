
"""Tests for src/clean_data.py: notebook cleaning rules."""

import pandas as pd
import pytest

from src.clean_data import clean_dataframe


COLUMNS = {
    "launched": "launched",
    "deadline": "deadline",
    "goal": "usd_goal_real",
    "name": "name",
    "country": "country",
}

DERIVED = {
    "goal": "goal_usd",
    "duration": "duration_days",
    "month": "launch_month",
    "year": "launch_year",
    "name_words": "name_words",
}

RULES = {
    "state_column": "state",
    "training_states": {"successful": 1, "failed": 0},
    "first_valid_year": 2009,
    "broken_country": 'N,0"',
    "min_goal_usd": 1.0,
    "min_investable_goal_usd": 10000,
    "leakage_columns": ["pledged", "backers"],
    "columns": COLUMNS,
    "derived": DERIVED,
}


def _raw() -> pd.DataFrame:
    """Create six synthetic Kickstarter campaigns."""
    return pd.DataFrame({
        "ID": [1, 2, 3, 4, 5, 6],
        "name": [
            "Great board game", None, "Album",
            "Lamp", "Tiny zine", "Robot kit",
        ],
        "country": ["US", "GB", "US", "US", "US", 'N,0"'],
        "state": [
            "successful", "failed", "live",
            "canceled", "successful", "failed",
        ],
        "launched": [
            "2015-01-01 10:00:00",
            "2016-03-01 09:00:00",
            "2017-06-01 12:00:00",
            "2017-07-01 08:00:00",
            "2017-08-01 08:00:00",
            "1970-01-01 01:00:00",
        ],
        "deadline": [
            "2015-01-31",
            "2016-04-15",
            "2017-07-01",
            "2017-08-01",
            "2017-09-01",
            "2017-02-01",
        ],
        "usd_goal_real": [
            15000.0, 30000.0, 20000.0,
            50000.0, 500.0, 12000.0,
        ],
        "pledged": [16000.0, 10.0, 5.0, 0.0, 600.0, 0.0],
        "backers": [120, 1, 1, 0, 9, 0],
    })


# Unit tests

def test_unit_training_mode_keeps_scope_and_builds_target():
    clean = clean_dataframe(_raw(), "success", **RULES)

    assert list(clean["ID"]) == [1, 2, 6]
    assert list(clean["success"]) == [1, 0, 0]
    assert list(clean.index) == [0, 1, 2]


def test_unit_derived_columns_have_known_values():
    row = clean_dataframe(_raw(), "success", **RULES).iloc[0]

    assert row["goal_usd"] == 15000.0
    assert row["duration_days"] == 30
    assert (row["launch_month"], row["launch_year"]) == (1, 2015)
    assert row["name_words"] == 3


def test_unit_bad_values_become_missing_not_fixed():
    clean = clean_dataframe(_raw(), "success", **RULES).set_index("ID")

    assert pd.isna(clean.loc[6, "launch_year"])
    assert pd.isna(clean.loc[6, "country"])
    assert clean.loc[2, "name_words"] == 0


def test_unit_leakage_and_raw_columns_are_dropped():
    clean = clean_dataframe(_raw(), "success", **RULES)

    assert not {
        "pledged", "backers", "launched", "deadline", "usd_goal_real"
    } & set(clean.columns)


def test_unit_inference_mode_keeps_every_row():
    clean = clean_dataframe(
        _raw().drop(columns="state"),
        None,
        **{**RULES, "state_column": None},
    )

    assert len(clean) == 6
    assert "success" not in clean.columns


def test_unit_contract_call_with_existing_target_drops_missing_target():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, None, 0]})

    assert len(clean_dataframe(df, "y")) == 2


def test_unit_min_goal_rule_without_investable_scope():
    clean = clean_dataframe(
        _raw(),
        None,
        columns=COLUMNS,
        derived=DERIVED,
        min_goal_usd=1000,
    )

    assert pd.isna(clean.set_index("ID").loc[5, "goal_usd"])


def test_unit_input_is_not_modified():
    raw = _raw()
    before = raw.copy()

    clean_dataframe(raw, "success", **RULES)

    pd.testing.assert_frame_equal(raw, before)


# Unit tests: expected errors

def test_unit_not_a_dataframe_raises():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        clean_dataframe("not a frame")


def test_unit_empty_dataframe_raises():
    with pytest.raises(ValueError, match="empty"):
        clean_dataframe(_raw().iloc[0:0])


def test_unit_missing_target_column_raises():
    with pytest.raises(ValueError, match="Missing columns"):
        clean_dataframe(pd.DataFrame({"x": [1]}), "success")


def test_unit_missing_state_column_raises():
    with pytest.raises(ValueError, match="state"):
        clean_dataframe(_raw().drop(columns="state"), "success", **RULES)


def test_unit_state_without_mapping_raises():
    with pytest.raises(ValueError, match="training_states is empty"):
        clean_dataframe(_raw(), "success", state_column="state")


def test_unit_columns_without_derived_raises():
    with pytest.raises(ValueError, match="derived"):
        clean_dataframe(_raw(), None, columns=COLUMNS)


def test_unit_missing_raw_column_raises():
    with pytest.raises(ValueError, match="launched"):
        clean_dataframe(
            _raw().drop(columns="launched"),
            None,
            columns=COLUMNS,
            derived=DERIVED,
        )


def test_unit_investable_scope_without_goal_column_raises():
    with pytest.raises(ValueError, match="min_investable_goal_usd"):
        clean_dataframe(
            _raw(),
            "success",
            state_column="state",
            training_states={"successful": 1, "failed": 0},
            min_investable_goal_usd=10000,
        )


# Integration test

def test_integration_load_raw_data_output_cleans(tmp_path):
    """CSV loaded through utils and load_data cleans like an in-memory frame."""
    from src import utils
    from src.load_data import load_raw_data

    path = tmp_path / "raw.csv"
    utils.save_csv(_raw(), path)

    clean = clean_dataframe(load_raw_data(path), "success", **RULES)

    assert list(clean["ID"]) == [1, 2, 6]


def test_unit_partial_column_map_derives_only_what_is_given():
    clean = clean_dataframe(
        _raw(),
        None,
        columns={"country": "country"},
        derived=DERIVED,
        broken_country='N,0"',
    )

    assert "goal_usd" not in clean.columns
    assert "duration_days" not in clean.columns
    assert pd.isna(clean.loc[5, "country"])


def test_unit_launch_without_deadline_skips_duration():
    clean = clean_dataframe(
        _raw(),
        None,
        columns={"launched": "launched"},
        derived=DERIVED,
    )

    assert "duration_days" not in clean.columns
    assert clean.loc[0, "launch_year"] == 2015
