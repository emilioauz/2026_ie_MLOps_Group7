"""Tests for src/main.py: split_data on its own, and main() running the whole pipeline.

The pipeline test writes ~120 synthetic campaigns in the raw Kickstarter layout plus a config
pointing every path into tmp_path, so it never touches the real CSV or the real outputs.
"""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.main import main, split_data


def _raw_campaigns(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Synthetic campaigns with the raw Kickstarter columns and a learnable signal."""
    rng = np.random.default_rng(seed)
    goal = np.round(rng.uniform(10_000, 80_000, n), 2)
    games = rng.random(n) < 0.5
    success = (games & (goal < 50_000)) | (rng.random(n) < 0.15)
    launched = pd.Timestamp("2014-01-01") + pd.to_timedelta(rng.integers(0, 1000, n), "D")
    return pd.DataFrame(
        {
            "ID": np.arange(1, n + 1),
            "name": ["Campaign number %d" % i for i in range(n)],
            "category": np.where(games, "Tabletop Games", "Rock"),
            "main_category": np.where(games, "Games", "Music"),
            "deadline": (launched + pd.to_timedelta(30, "D")).strftime("%Y-%m-%d"),
            "launched": launched.strftime("%Y-%m-%d %H:%M:%S"),
            "state": np.where(success, "successful", "failed"),
            "backers": rng.integers(0, 100, n),
            "country": rng.choice(["US", "GB"], n),
            "usd_goal_real": goal,
        }
    )


def _write_config(tmp_path, raw_path) -> object:
    """A config like the team's, with every path inside tmp_path and a small model."""
    out = tmp_path / "out"
    config = {
        "paths": {
            "raw_data": str(raw_path),
            "clean_data": str(out / "clean.csv"),
            "inference_data": str(tmp_path / "missing_new_campaigns.csv"),
            "model": str(out / "model.joblib"),
            "predictions": str(out / "predictions.csv"),
            "metrics": str(out / "metrics.json"),
            "log_file": str(out / "pipeline.log"),
        },
        "logging": {"level": "INFO"},
        "data": {
            "id_column": "ID",
            "state_column": "state",
            "target_column": "success",
            "training_states": {"successful": 1, "failed": 0},
            "problem_type": "classification",
        },
        "cleaning": {
            "first_valid_year": 2009,
            "broken_country": 'N,0"',
            "min_goal_usd": 1.0,
            "min_investable_goal_usd": 10000,
            "leakage_columns": ["backers"],
            "columns": {
                "launched": "launched",
                "deadline": "deadline",
                "goal": "usd_goal_real",
                "name": "name",
                "country": "country",
            },
            "derived": {
                "goal": "goal_usd",
                "duration": "duration_days",
                "month": "launch_month",
                "year": "launch_year",
                "name_words": "name_words",
            },
        },
        "validation": {
            "required_columns": ["ID", "main_category", "country", "goal_usd", "duration_days"],
            "positive_columns": ["goal_usd"],
            "duration_column": "duration_days",
            "duration_range": [1, 92],
            "year_column": "launch_year",
            "launch_year_range": [2009, 2030],
        },
        "features": {
            "numeric": ["goal_usd", "duration_days", "launch_month", "launch_year", "name_words"],
            "log": ["goal_usd"],
            "categorical": ["main_category", "category", "country"],
            "rare_category_minimum": 5,
        },
        "split": {"test_size": 0.2, "val_size": 0.2, "random_state": 42, "stratify": True},
        "model": {"type": "random_forest", "params": {"n_estimators": 20}, "random_state": 42},
        "evaluation": {
            "metric": "roc_auc",
            "threshold": 0.70,
            "direction": "higher_is_better",
            "probability_cutoff": 0.5,
            "shortlist_share": 0.1,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path, config


# ---------- unit: split_data ----------
def test_unit_split_data_gives_60_20_20():
    X = pd.DataFrame({"x": range(100)})  # pylint: disable=invalid-name
    y = pd.Series([0, 1] * 50)
    X_train, X_val, X_test, y_train, _, _ = split_data(
        X, y, 0.2, 0.2, 42, True
    )  # pylint: disable=invalid-name
    assert (len(X_train), len(X_val), len(X_test)) == (60, 20, 20)
    assert y_train.mean() == 0.5


def test_unit_split_data_is_reproducible():
    X = pd.DataFrame({"x": range(50)})  # pylint: disable=invalid-name
    y = pd.Series([0, 1] * 25)
    first = split_data(X, y, 0.2, 0.2, 7, False)[0]
    second = split_data(X, y, 0.2, 0.2, 7, False)[0]
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("test_size, val_size", [(0, 0.2), (0.2, 1.0), (0.5, 0.5)])
def test_unit_split_data_bad_sizes_raise(test_size, val_size):
    with pytest.raises(ValueError, match="Bad split sizes"):
        split_data(
            pd.DataFrame({"x": range(10)}), pd.Series([0, 1] * 5), test_size, val_size, 0, False
        )


# ---------- integration: the whole pipeline ----------
def test_integration_main_runs_pipeline_and_writes_outputs(tmp_path):
    raw_path = tmp_path / "raw.csv"
    _raw_campaigns().to_csv(raw_path, index=False)
    config_path, config = _write_config(tmp_path, raw_path)
    main(config_path)
    paths = config["paths"]
    for key in ("clean_data", "model", "predictions", "metrics", "log_file"):
        assert (tmp_path / paths[key]).exists(), key
    metrics = json.loads((tmp_path / paths["metrics"]).read_text(encoding="utf-8"))
    assert 0 <= metrics["roc_auc"] <= 1 and isinstance(metrics["gate_met"], bool)
    predictions = pd.read_csv(paths["predictions"])
    assert list(predictions.columns) == ["ID", "prediction", "probability"]


def test_integration_main_scores_inference_file_when_present(tmp_path):
    raw_path = tmp_path / "raw.csv"
    _raw_campaigns().to_csv(raw_path, index=False)
    config_path, config = _write_config(tmp_path, raw_path)
    new = _raw_campaigns(n=8, seed=1).drop(columns=["state", "backers"])
    new.to_csv(config["paths"]["inference_data"], index=False)
    main(config_path)
    assert len(pd.read_csv(config["paths"]["predictions"])) == 8


def test_integration_main_exits_1_on_bad_data(tmp_path):
    raw_path = tmp_path / "raw.csv"
    _raw_campaigns().drop(columns="main_category").to_csv(raw_path, index=False)
    config_path, _ = _write_config(tmp_path, raw_path)
    with pytest.raises(SystemExit) as stop:
        main(config_path)
    assert stop.value.code == 1


def test_integration_main_exits_1_on_missing_config(tmp_path):
    with pytest.raises(SystemExit) as stop:
        main(tmp_path / "no_config.yaml")
    assert stop.value.code == 1


def test_integration_gate_not_met_is_reported(tmp_path):
    raw_path = tmp_path / "raw.csv"
    _raw_campaigns().to_csv(raw_path, index=False)
    config_path, config = _write_config(tmp_path, raw_path)
    config["evaluation"]["threshold"] = 1.01
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    main(config_path)
    log_text = (tmp_path / config["paths"]["log_file"]).read_text(encoding="utf-8")
    assert "Quality gate NOT MET" in log_text
    assert (
        json.loads((tmp_path / config["paths"]["metrics"]).read_text(encoding="utf-8"))["gate_met"]
        is False
    )
