"""
utils.py  [Shared plumbing]
===========================
Small, reusable read and write helpers that every other module calls.

Functions
    load_csv      read a CSV into a DataFrame
    save_csv      write a DataFrame to CSV, creating the folder if needed
    save_model    save the fitted pipeline with joblib, creating the folder if needed
    load_model    load an object saved by save_model
    load_config   read config.yaml into a dictionary

Rule
    No business logic: no column names, no target, no dataset paths. Callers pass every
    path in; the values come from config.yaml through main.py.

Tests     tests/test_utils.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import joblib
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _require_file(filepath: str | Path, what: str) -> Path:
    """Return ``filepath`` as a Path, or raise if it is missing or is a folder.

    Args:
        filepath: Location to check.
        what: Word used in the error message, e.g. "Config file".

    Raises:
        FileNotFoundError: If nothing exists at ``filepath``.
        IsADirectoryError: If ``filepath`` is a folder, not a file.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"{what} path is a folder, not a file: {path}")
    return path


def load_csv(filepath: Path, **read_kwargs: Any) -> pd.DataFrame:
    """Read a CSV file into a DataFrame, exactly as it is on disk.

    Args:
        filepath: Location of the CSV file.
        **read_kwargs: Optional extra arguments for ``pandas.read_csv``,
            e.g. ``low_memory=False`` or ``usecols``.

    Returns:
        The file contents with a default RangeIndex.

    Raises:
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If ``filepath`` is a folder.
    """
    path = _require_file(filepath, "CSV file")
    df = pd.read_csv(path, **read_kwargs)
    logger.info("Read %d rows x %d columns from %s", len(df), df.shape[1], path.name)
    return df


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """Write a DataFrame to CSV without its index, creating parent folders when missing.

    Args:
        df: Data to write.
        filepath: Destination file, e.g. ``data/processed/clean.csv``.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"save_csv expects a pandas DataFrame, got {type(df).__name__}")
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %d rows to %s", len(df), path)


def save_model(model: Any, filepath: Path) -> None:
    """Save a fitted model or pipeline with joblib, creating parent folders when missing.

    Args:
        model: Any picklable object, normally the fitted scikit-learn pipeline.
        filepath: Destination file, e.g. ``models/model.joblib``.

    Raises:
        ValueError: If ``model`` is None.
    """
    if model is None:
        raise ValueError("save_model received None: there is no model to save")
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)


def load_model(filepath: Path) -> Any:
    """Load an object saved by :func:`save_model`.

    Only load files this pipeline wrote: joblib files can run code when loaded.

    Args:
        filepath: Location of the ``.joblib`` file.

    Returns:
        The saved object.

    Raises:
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If ``filepath`` is a folder.
    """
    path = _require_file(filepath, "Model file")
    model = joblib.load(path)
    logger.info("Loaded model from %s", path)
    return model


def load_config(filepath: Path, required_sections: Iterable[str] | None = None) -> dict:
    """Read a YAML config file into a dictionary.

    Args:
        filepath: Location of the YAML file, normally ``config.yaml``.
        required_sections: Optional top-level keys that must be present,
            e.g. ``["paths", "data"]``. None skips the check.

    Returns:
        The configuration as a (possibly nested) dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If ``filepath`` is a folder.
        ValueError: If the YAML is invalid, the file is empty, or its top level
            is not a mapping of keys to values.
        KeyError: If a required section is missing.
    """
    path = _require_file(filepath, "Config file")
    with path.open(encoding="utf-8") as handle:
        try:
            config = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ValueError(f"Config file is not valid YAML: {path} ({exc})") from exc
    if config is None:
        raise ValueError(f"Config file is empty: {path}")
    if not isinstance(config, dict):
        raise ValueError(f"Config file must hold key: value pairs at the top level: {path}")
    missing = [section for section in (required_sections or []) if section not in config]
    if missing:
        raise KeyError(f"{path.name} is missing required sections: {missing}")
    logger.info("Loaded config from %s with sections %s", path.name, list(config))
    return config
