"""
load_data.py  [Step 1: LOAD]
============================
Read the raw dataset into a DataFrame, exactly as it is on disk.

Input     Path to the raw CSV (config: paths.raw_data)
Output    Raw pandas DataFrame, untouched, with a default RangeIndex
Called by main.py          Uses   utils.load_csv
Next      clean_data.py

Why it matters
- One entry point for data: swap the CSV for a database later without touching other modules
- Read only: the raw file is never modified, so every run starts from the same data
- Fail fast with a clear message if the path is missing, is a folder, or the file is empty

Tests     tests/test_load_data.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src import utils

logger = logging.getLogger(__name__)


def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """Read the raw CSV file without changing any value.

    Columns are read in one pass (``low_memory=False``) so a column holding mixed
    values gets one consistent type instead of a per-chunk guess.

    Args:
        raw_data_path: Location of the raw CSV (config: ``paths.raw_data``).

    Returns:
        The raw data with a default RangeIndex.

    Raises:
        FileNotFoundError: If the path does not exist.
        IsADirectoryError: If the path is a folder.
        ValueError: If the file is empty (0 bytes) or holds a header but no rows.
    """
    path = Path(raw_data_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path} (download steps: README)")
    if path.is_dir():
        raise IsADirectoryError(f"Raw data path is a folder, expected a CSV file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Raw data file is empty (0 bytes): {path}")
    df = utils.load_csv(path, low_memory=False)
    if df.empty:
        raise ValueError(f"Raw data file has a header but no rows: {path}")
    logger.info("Loaded raw data: %d rows x %d columns", len(df), df.shape[1])
    return df
