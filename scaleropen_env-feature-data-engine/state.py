"""
state.py
────────
Data-health inspector for the Silent Data Debugger.

Scans a pandas DataFrame and produces a structured health report describing
statistical issues (missing values, skewness, scale mismatch, outliers,
class imbalance, wrong dtypes).  The report is designed to be serialised
and fed to an LLM agent for downstream decision-making.

Public API
──────────
get_state(df, config)                        → str   (JSON — used by SilentDebuggerEnv)
get_health_report(df, config, ...)           → dict  (used by pipeline.py)
get_data_health(df, target_col, task_type, config) → dict
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import GlobalConfig


# ── Individual diagnostic helpers ───────────────────────────────────────────

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
    """
    Return True if the std-deviation ratio of numeric columns exceeds
    ``config.scale_ratio``.  Uses std rather than raw range so that
    columns with naturally large absolute values (e.g. salaries) are
    compared fairly against unit-scale features.
    """
    numeric_df = df.select_dtypes(include="number")
    stds = numeric_df.std()
    non_zero = stds[stds > 0]
    if non_zero.empty or len(non_zero) < 2:
        return False
    return bool(non_zero.max() / non_zero.min() > config.scale_ratio)


def check_outliers(df: pd.DataFrame, config: GlobalConfig) -> dict[str, int]:
    """
    Detect outliers via the IQR method and return per-column counts.

    A value is an outlier if it falls below
    ``Q1 - factor * IQR`` or above ``Q3 + factor * IQR``.
    Only columns with at least ``config.outlier_min_count`` outliers
    are included.
    """
    numeric_df = df.select_dtypes(include="number")
    result: dict[str, int] = {}
    for col in numeric_df.columns:
        q1 = float(numeric_df[col].quantile(0.25))
        q3 = float(numeric_df[col].quantile(0.75))
        iqr = q3 - q1
        if iqr < 1e-5:
            continue
        lower = q1 - config.outlier_iqr_factor * iqr
        upper = q3 + config.outlier_iqr_factor * iqr
        count = int(((numeric_df[col] < lower) | (numeric_df[col] > upper)).sum())
        if count >= config.outlier_min_count:
            result[col] = count
    return result


def check_imbalance(
    df: pd.DataFrame,
    target_col: str,
    config: GlobalConfig,
    task_type: str,
) -> str | None:
    """
    For **classification** tasks, check whether the minority-class ratio
    falls below ``config.imbalance_threshold``.

    Returns a descriptive string if imbalanced, otherwise ``None``.
    """
    if task_type != "classification":
        return None

    counts = df[target_col].value_counts()
    total = len(df)
    if total == 0:
        return None

    min_ratio = float(counts.min()) / total
    if min_ratio < config.imbalance_threshold:
        minority = counts.idxmin()
        return (
            f"Imbalanced target '{target_col}': class '{minority}' "
            f"has ratio {min_ratio:.4f} (threshold {config.imbalance_threshold})"
        )
    return None


def check_wrong_dtypes(df: pd.DataFrame) -> dict[str, str]:
    """
    Flag columns stored as object/string that appear to contain numeric
    values — likely caused by ingest or encoding errors.

    A column is flagged if ``pd.to_numeric`` succeeds on a 50-row sample.
    """
    wrong: dict[str, str] = {}
    for col in df.select_dtypes(include=["object", "string"]).columns:
        sample = df[col].dropna().head(50)
        try:
            pd.to_numeric(sample)
            wrong[col] = f"stored as {df[col].dtype}, appears numeric"
        except (ValueError, TypeError):
            pass
    return wrong


# ── Orchestrators ────────────────────────────────────────────────────────────

def get_data_health(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    config: GlobalConfig,
) -> dict:
    """
    Run every diagnostic and return a single health-report dict.

    Keys
    ----
    missing_values     : dict[str, int]
    high_skew_columns  : dict[str, float]
    scale_mismatch     : bool
    outliers           : dict[str, int]
    class_imbalance    : str | None
    wrong_dtypes       : dict[str, str]
    dtypes             : dict[str, str]
    """
    return {
        "missing_values":    check_missing(df),
        "high_skew_columns": check_skew(df, config),
        "scale_mismatch":    check_scale_mismatch(df, config),
        "outliers":          check_outliers(df, config),
        "class_imbalance":   check_imbalance(df, target_col, config, task_type),
        "wrong_dtypes":      check_wrong_dtypes(df),
        "dtypes":            {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def get_health_report(
    df: pd.DataFrame,
    config: GlobalConfig,
    target_col: str | None = None,
    task_type: str | None = None,
) -> dict:
    """
    Preferred entry point for ``pipeline.py``.

    When both *target_col* and *task_type* are supplied the imbalance check
    runs; otherwise it is safely set to ``None``.

    Returns the same schema as :func:`get_data_health`.
    """
    imbalance = (
        check_imbalance(df, target_col, config, task_type)
        if (target_col is not None and task_type is not None)
        else None
    )
    return {
        "missing_values":    check_missing(df),
        "high_skew_columns": check_skew(df, config),
        "scale_mismatch":    check_scale_mismatch(df, config),
        "outliers":          check_outliers(df, config),
        "class_imbalance":   imbalance,
        "wrong_dtypes":      check_wrong_dtypes(df),
        "dtypes":            {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def get_state(df: pd.DataFrame, config: GlobalConfig) -> str:
    """
    Return a JSON string version of the health report.

    This is the interface used by ``SilentDebuggerEnv.reset()`` and
    ``SilentDebuggerEnv.step()`` to populate ``obs["state"]``.
    The agent parses this string to build its fix prompt.
    """
    report = get_health_report(df, config)
    return json.dumps(report, indent=4)