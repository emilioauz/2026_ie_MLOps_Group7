"""Tests for src/utils.py: CSV reading and writing, model saving and config loading.

Tiny synthetic data and pytest's tmp_path only; never the real Kickstarter CSV.
"""

import logging

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import utils


def _campaigns() -> pd.DataFrame:
    """Three fake campaigns with a text, a numeric and a missing value."""
    return pd.DataFrame({"ID": [1, 2, 3], "name": ["Board game", "Album", None], "goal_usd": [5000.0, 12000.0, 800.0]})


# ---------- load_csv ----------
def test_unit_load_csv_reads_known_values(tmp_path):
    path = tmp_path / "small.csv"
    path.write_text("ID,goal_usd\n1,5000\n2,12000\n", encoding="utf-8")
    df = utils.load_csv(path)
    assert df.shape == (2, 2)
    assert df.loc[1, "goal_usd"] == 12000
    assert list(df.index) == [0, 1]


def test_unit_load_csv_passes_read_options(tmp_path):
    path = tmp_path / "small.csv"
    path.write_text("ID,goal_usd\n1,5000\n", encoding="utf-8")
    assert list(utils.load_csv(path, usecols=["ID"]).columns) == ["ID"]


def test_unit_load_csv_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="CSV file not found"):
        utils.load_csv(tmp_path / "missing.csv")


def test_unit_load_csv_folder_raises(tmp_path):
    with pytest.raises(IsADirectoryError, match="folder"):
        utils.load_csv(tmp_path)


# ---------- save_csv ----------
def test_unit_save_csv_creates_folders_and_drops_index(tmp_path):
    target = tmp_path / "processed" / "nested" / "clean.csv"
    df = _campaigns().set_index("ID")
    assert utils.save_csv(df, target) is None
    assert target.read_text(encoding="utf-8").splitlines()[0] == "name,goal_usd"


def test_unit_save_csv_rejects_non_dataframe(tmp_path):
    with pytest.raises(TypeError, match="pandas DataFrame"):
        utils.save_csv([1, 2, 3], tmp_path / "out.csv")


def test_unit_save_csv_logs_row_count(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="src.utils"):
        utils.save_csv(_campaigns(), tmp_path / "out.csv")
    assert "Saved 3 rows" in caplog.text


# ---------- save_model / load_model ----------
def test_unit_save_and_load_simple_object(tmp_path):
    path = tmp_path / "models" / "obj.joblib"
    assert utils.save_model({"threshold": 0.29}, path) is None
    assert utils.load_model(path) == {"threshold": 0.29}


def test_unit_save_model_rejects_none(tmp_path):
    with pytest.raises(ValueError, match="no model to save"):
        utils.save_model(None, tmp_path / "model.joblib")


def test_unit_load_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        utils.load_model(tmp_path / "missing.joblib")


# ---------- load_config ----------
def test_unit_load_config_reads_nested_values(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("paths:\n  raw_data: data/raw/x.csv\nsplit:\n  test_size: 0.2\n", encoding="utf-8")
    config = utils.load_config(path, required_sections=["paths", "split"])
    assert config["paths"]["raw_data"] == "data/raw/x.csv"
    assert config["split"]["test_size"] == 0.2


def test_unit_load_config_missing_section_raises(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("paths:\n  raw_data: data/raw/x.csv\n", encoding="utf-8")
    with pytest.raises(KeyError, match="model"):
        utils.load_config(path, required_sections=["paths", "model"])


def test_unit_load_config_invalid_yaml_raises(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("paths: [unclosed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid YAML"):
        utils.load_config(path)


def test_unit_load_config_empty_file_raises(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        utils.load_config(path)


def test_unit_load_config_non_mapping_raises(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="key: value"):
        utils.load_config(path)


def test_unit_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        utils.load_config(tmp_path / "nope.yaml")


# ---------- integration ----------
def test_integration_csv_roundtrip_keeps_values(tmp_path):
    original = _campaigns()
    path = tmp_path / "processed" / "clean.csv"
    utils.save_csv(original, path)
    pd.testing.assert_frame_equal(utils.load_csv(path), original)


def test_integration_fitted_pipeline_roundtrip_predicts_the_same(tmp_path):
    X = pd.DataFrame({"goal_usd": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "duration_days": [30, 30, 45, 45, 60, 60]})
    y = [0, 0, 0, 1, 1, 1]
    pipeline = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())]).fit(X, y)
    path = tmp_path / "models" / "model.joblib"
    utils.save_model(pipeline, path)
    assert (utils.load_model(path).predict_proba(X) == pipeline.predict_proba(X)).all()
