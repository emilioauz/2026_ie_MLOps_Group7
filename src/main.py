"""
main.py  [Step 0: ORCHESTRATOR]
================================
Run the whole pipeline, in order, with one command: python -m src.main

    load -> clean -> validate -> split -> features -> train -> evaluate -> save -> infer

Every value comes from config.yaml. main.py only calls the other modules and passes data
between them; cleaning, feature and model logic live in their own modules.

Outputs (paths in config.yaml)
    paths.clean_data    cleaned training data
    paths.model         fitted pipeline (preprocessing + model)
    paths.metrics       validation metrics and the quality gate result
    paths.predictions   scores for paths.inference_data (test split if that file is absent)
    paths.log_file      run log

Tests     tests/test_main.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src import clean_data, evaluate, features, infer, load_data, train, utils, validate

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SECTIONS = [
    "paths",
    "data",
    "cleaning",
    "validation",
    "features",
    "split",
    "model",
    "evaluation",
]


# Signature fixed by the course contract.
# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def split_data(
    X: pd.DataFrame,  # pylint: disable=invalid-name
    y: pd.Series,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify: bool,
) -> tuple:
    """Split into train, validation and test sets; both sizes are shares of the FULL data.

    Returns:
        ``(X_train, X_val, X_test, y_train, y_val, y_test)``

    Raises:
        ValueError: If a size is not between 0 and 1, or test_size + val_size >= 1.
    """
    if not 0 < test_size < 1 or not 0 < val_size < 1 or test_size + val_size >= 1:
        raise ValueError(
            f"Bad split sizes: test_size={test_size}, val_size={val_size}; "
            "each must be in (0, 1) and together below 1"
        )
    X_rest, X_test, y_rest, y_test = train_test_split(  # pylint: disable=invalid-name
        X, y, test_size=test_size, random_state=random_state, stratify=y if stratify else None
    )
    val_share = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(  # pylint: disable=invalid-name
        X_rest,
        y_rest,
        test_size=val_share,
        random_state=random_state,
        stratify=y_rest if stratify else None,
    )
    logger.info(
        "Split: %d train, %d validation, %d test rows", len(X_train), len(X_val), len(X_test)
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _path(cfg: dict, key: str) -> Path:
    """Resolve ``paths.<key>`` against the project root (absolute paths are kept)."""
    return PROJECT_ROOT / cfg["paths"][key]


def _configure_logging(level: str, log_file: Path) -> None:
    """Send log lines to the terminal and to the log file, in one format."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )


def _clean_kwargs(cfg: dict, training: bool) -> dict[str, Any]:
    """Cleaning rules from config; the state column is only used in training mode."""
    rules = dict(cfg["cleaning"])
    if training:
        rules["state_column"] = cfg["data"]["state_column"]
        rules["training_states"] = cfg["data"]["training_states"]
    return rules


def _validate(df: pd.DataFrame, cfg: dict, target: str | None) -> None:
    """Run validate_dataframe with the rules in config."""
    rules = dict(cfg["validation"])
    required = rules.pop("required_columns")
    validate.validate_dataframe(
        df,
        required,
        target,
        id_column=cfg["data"]["id_column"],
        forbidden_columns=cfg["cleaning"]["leakage_columns"],
        **rules,
    )


def _report_gate(metrics: dict[str, float], cfg: dict) -> bool:
    """Log the quality gate as MET or NOT MET and return whether it is met."""
    gate = cfg["evaluation"]
    value = metrics[gate["metric"]]
    higher = gate["direction"] == "higher_is_better"
    met = value >= gate["threshold"] if higher else value <= gate["threshold"]
    log = logger.info if met else logger.warning
    log(
        "Quality gate %s: %s = %.3f (threshold %s, %s)",
        "MET" if met else "NOT MET",
        gate["metric"],
        value,
        gate["threshold"],
        gate["direction"],
    )
    return met


def _predict(cfg: dict, model, columns: list[str], fallback: pd.DataFrame) -> pd.DataFrame:
    """Score paths.inference_data if it exists, otherwise the held-out test rows."""
    id_column = cfg["data"]["id_column"]
    source = _path(cfg, "inference_data")
    if source.exists():
        new = clean_data.clean_dataframe(
            load_data.load_raw_data(source), None, **_clean_kwargs(cfg, training=False)
        )
        _validate(new, cfg, None)
    else:
        logger.warning("No inference file at %s: scoring the test split instead", source)
        new = fallback
    scores = infer.run_inference(
        model, new[columns], include_proba=True, threshold=cfg["evaluation"]["probability_cutoff"]
    )
    return pd.concat([new[[id_column]], scores], axis=1)


def _train_and_evaluate(cfg: dict, columns: list[str], train_set: tuple, val_set: tuple) -> tuple:
    """Fit the feature recipe + model on the training split and score the validation split."""
    feat_cfg, problem_type = cfg["features"], cfg["data"]["problem_type"]
    preprocessor = features.get_feature_preprocessor(
        feat_cfg["numeric"],
        feat_cfg["categorical"],
        feat_cfg["log"],
        feat_cfg["rare_category_minimum"],
    )
    model_cfg, eval_cfg = cfg["model"], cfg["evaluation"]
    model = train.train_model(
        train_set[0][columns],
        train_set[1],
        preprocessor,
        problem_type,
        {"type": model_cfg["type"], "params": model_cfg["params"]},
        model_cfg["random_state"],
    )
    metrics = evaluate.evaluate_model(
        model,
        val_set[0][columns],
        val_set[1],
        problem_type,
        threshold=eval_cfg["probability_cutoff"],
        shortlist_share=eval_cfg["shortlist_share"],
    )
    return model, metrics


def _load_clean_data(cfg: dict) -> pd.DataFrame:
    """Load, clean and validate the training data, and save the clean copy."""
    target = cfg["data"]["target_column"]
    raw = load_data.load_raw_data(_path(cfg, "raw_data"))
    clean = clean_data.clean_dataframe(raw, target, **_clean_kwargs(cfg, training=True))
    _validate(clean, cfg, target)
    utils.save_csv(clean, _path(cfg, "clean_data"))
    return clean


def _save_metrics(metrics: dict[str, float], gate_met: bool, path: Path) -> None:
    """Write the validation metrics and the gate result as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**metrics, "gate_met": gate_met}, indent=2), encoding="utf-8")


def run(cfg: dict) -> dict[str, float]:
    """Run every step once and write every output; returns the validation metrics."""
    clean = _load_clean_data(cfg)
    columns = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    X_train, X_val, X_test, y_train, y_val, _ = split_data(  # pylint: disable=invalid-name
        clean[columns + [cfg["data"]["id_column"]]],
        clean[cfg["data"]["target_column"]],
        **cfg["split"],
    )
    model, metrics = _train_and_evaluate(cfg, columns, (X_train, y_train), (X_val, y_val))
    gate_met = _report_gate(metrics, cfg)
    utils.save_model(model, _path(cfg, "model"))
    _save_metrics(metrics, gate_met, _path(cfg, "metrics"))
    utils.save_csv(_predict(cfg, model, columns, X_test), _path(cfg, "predictions"))
    logger.info("Pipeline finished")
    return metrics


def main(config_path: Path | None = None) -> None:
    """Entry point: load config, run the pipeline; on failure log the reason and exit 1."""
    try:
        cfg = utils.load_config(config_path or PROJECT_ROOT / "config.yaml", REQUIRED_SECTIONS)
        _configure_logging(cfg["logging"]["level"], _path(cfg, "log_file"))
        run(cfg)
    except (ValueError, TypeError, KeyError, OSError) as error:
        logger.error("Pipeline stopped: %s", error)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
