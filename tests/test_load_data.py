"""Tests for src/load_data.py: reading the raw CSV unchanged and failing fast on bad paths.

Tiny synthetic data and pytest's tmp_path only; never the real Kickstarter CSV.
"""

import pandas as pd
import pytest

from src import utils
from src.load_data import load_raw_data

RAW = pd.DataFrame(
    {
        "ID": [1000002330, 1000003930, 1000004038],
        "name": ["The Songs of Adelaide", "Greeting From Earth", None],
        "category": ["Poetry", "Narrative Film", "Narrative Film"],
        "state": ["failed", "successful", "canceled"],
        "usd_goal_real": [1533.95, 30000.0, 45000.0],
    }
)


# ---------- unit ----------
def test_unit_load_raw_data_returns_file_unchanged(tmp_path):
    path = tmp_path / "raw.csv"
    RAW.to_csv(path, index=False)
    pd.testing.assert_frame_equal(load_raw_data(path), RAW)


def test_unit_load_raw_data_keeps_mixed_column_as_one_type(tmp_path):
    path = tmp_path / "raw.csv"
    pd.DataFrame({"country": ["US"] * 3 + ['N,0"'] + ["1"] * 2}).to_csv(path, index=False)
    df = load_raw_data(path)
    assert len(df) == 6
    assert df["country"].map(type).nunique() == 1


def test_unit_load_raw_data_accepts_string_path(tmp_path):
    path = tmp_path / "raw.csv"
    RAW.to_csv(path, index=False)
    assert len(load_raw_data(str(path))) == 3


def test_unit_load_raw_data_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Raw data file not found"):
        load_raw_data(tmp_path / "missing.csv")


def test_unit_load_raw_data_folder_raises(tmp_path):
    with pytest.raises(IsADirectoryError, match="folder"):
        load_raw_data(tmp_path)


def test_unit_load_raw_data_zero_byte_file_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="0 bytes"):
        load_raw_data(path)


def test_unit_load_raw_data_header_only_raises(tmp_path):
    path = tmp_path / "header.csv"
    path.write_text("ID,name,state\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no rows"):
        load_raw_data(path)


# ---------- integration ----------
def test_integration_save_csv_output_loads_back_unchanged(tmp_path):
    """utils.save_csv writes the file; load_raw_data reads it back identically."""
    path = tmp_path / "raw" / "campaigns.csv"
    utils.save_csv(RAW, path)
    pd.testing.assert_frame_equal(load_raw_data(path), RAW)


def test_integration_config_path_drives_the_load(tmp_path):
    """As in main.py: utils.load_config gives paths.raw_data, load_raw_data reads it."""
    raw_path = tmp_path / "raw" / "campaigns.csv"
    utils.save_csv(RAW, raw_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"paths:\n  raw_data: {raw_path.as_posix()}\n", encoding="utf-8")
    config = utils.load_config(config_path, required_sections=["paths"])
    assert load_raw_data(config["paths"]["raw_data"]).shape == RAW.shape
