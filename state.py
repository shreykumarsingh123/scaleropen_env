"""
Data Health Report Generator.

Scans a pandas DataFrame and produces a structured "Health Report"
dictionary describing statistical issues (missing values, skew,
scale mismatch, outliers, class imbalance).  The report is designed
to be serialised and fed to an LLM for downstream decision-making.
"""

import pandas as pd
import numpy as np

from config import GlobalConfig


# --------------- Helper Functions ---------------


def check_missing(df: pd.DataFrame) -> dict[str, int]:
    """Return columns that contain missing values and their counts."""
    counts = df.isnull().sum()
    return {col: int(cnt) for col, cnt in counts.items() if cnt > 0}


def check_skew(df: pd.DataFrame, config: GlobalConfig) -> dict[str, float]:
    """Return numeric columns whose absolute skewness exceeds the threshold."""
    numeric_df = df.select_dtypes(include="number")
    skew_values = numeric_df.skew()
    return {
        col: round(float(skew), 4)
        for col, skew in skew_values.items()
        if abs(skew) > config.skew_threshold
    }


def check_scale_mismatch(df: pd.DataFrame, config: GlobalConfig) -> bool:
    """Check whether the ratio of max-to-min std deviation exceeds the threshold."""
    numeric_df = df.select_dtypes(include="number")
    stds = numeric_df.std()

    # Exclude columns with zero standard deviation
    non_zero_stds = stds[stds > 0]

    if non_zero_stds.empty or len(non_zero_stds) < 2:
        return False

    ratio = non_zero_stds.max() / non_zero_stds.min()
    return bool(ratio > config.scale_ratio)


def check_outliers(df: pd.DataFrame, config: GlobalConfig) -> dict[str, int]:
    """Detect outliers via the IQR method and return per-column counts."""
    numeric_df = df.select_dtypes(include="number")
    outlier_counts: dict[str, int] = {}

    for col in numeric_df.columns:
        q1 = numeric_df[col].quantile(0.25)
        q3 = numeric_df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - config.outlier_iqr_factor * iqr
        upper = q3 + config.outlier_iqr_factor * iqr

        count = int(((numeric_df[col] < lower) | (numeric_df[col] > upper)).sum())
        if count >= config.outlier_min_count:
            outlier_counts[col] = count

    return outlier_counts


def check_imbalance(
    df: pd.DataFrame,
    target_col: str,
    config: GlobalConfig,
    task_type: str,
) -> str | None:
    """
    For classification tasks, check whether the minority class ratio
    falls below the configured imbalance threshold.

    Returns a descriptive string if imbalanced, otherwise None.
    """
    if task_type != "classification":
        return None

    class_counts = df[target_col].value_counts()
    total = len(df)

    if total == 0:
        return None

    min_ratio = class_counts.min() / total

    if min_ratio < config.imbalance_threshold:
        minority_class = class_counts.idxmin()
        return (
            f"Imbalanced target '{target_col}': class '{minority_class}' "
            f"has ratio {min_ratio:.4f} (threshold {config.imbalance_threshold})"
        )

    return None


def check_wrong_dtypes(df: pd.DataFrame) -> dict[str, str]:
    """
    Flag columns that are stored as string/object but appear to
    contain numeric values — likely ingest errors.

    Returns
    -------
    dict[str, str]
        ``{column_name: "stored as <dtype>, appears numeric"}``
    """
    wrong: dict[str, str] = {}
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        sample = df[col].dropna().head(50)
        try:
            pd.to_numeric(sample)
            wrong[col] = f"stored as {df[col].dtype}, appears numeric"
        except (ValueError, TypeError):
            pass
    return wrong


# --------------- Main Orchestrator ---------------


def get_data_health(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    config: GlobalConfig,
) -> dict:
    """
    Run every diagnostic check and return a single Health Report dict.

    Keys
    ----
    missing_values     : dict[str, int]   – columns with missing counts
    high_skew_columns  : dict[str, float] – columns exceeding skew threshold
    scale_mismatch     : bool             – whether scale ratio is exceeded
    outliers           : dict[str, int]   – columns with outlier counts
    class_imbalance    : str | None       – description or None
    wrong_dtypes       : dict[str, str]   – columns stored as str but numeric
    dtypes             : dict[str, str]   – column name → dtype string
    """
    return {
        "missing_values": check_missing(df),
        "high_skew_columns": check_skew(df, config),
        "scale_mismatch": check_scale_mismatch(df, config),
        "outliers": check_outliers(df, config),
        "class_imbalance": check_imbalance(df, target_col, config, task_type),
        "wrong_dtypes": check_wrong_dtypes(df),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def get_health_report(
    df: pd.DataFrame,
    config: GlobalConfig,
    target_col: str | None = None,
    task_type: str | None = None,
) -> dict:
    """Produce a data health report.

    This is the preferred entry point for ``env.py``.  When *target_col*
    and *task_type* are both provided the class-imbalance check runs;
    otherwise it is safely set to ``None``.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to diagnose.
    config : GlobalConfig
        Pipeline-wide thresholds.
    target_col : str | None, optional
        Name of the target column.
    task_type : str | None, optional
        ``"regression"`` or ``"classification"``.

    Returns
    -------
    dict
        Same schema as :func:`get_data_health`.
    """
    if target_col is not None and task_type is not None:
        imbalance = check_imbalance(df, target_col, config, task_type)
    else:
        imbalance = None

    return {
        "missing_values": check_missing(df),
        "high_skew_columns": check_skew(df, config),
        "scale_mismatch": check_scale_mismatch(df, config),
        "outliers": check_outliers(df, config),
        "class_imbalance": imbalance,
        "wrong_dtypes": check_wrong_dtypes(df),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
