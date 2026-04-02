"""
test_pipeline.py
────────────────
End-to-end integration test for the Silent Data Debugger pipeline.

Loads the synthetic dataset, runs the full pipeline, and prints a
structured diagnostic report.  Exits with code 1 if any critical
check fails, 0 if all pass.
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
import pandas as pd

from config import EnvConfig
from env import run_pipeline
from synthetic_data import get_synthetic_dataset


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def print_section(title: str) -> None:
    """Print a clearly separated section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    # ── Load synthetic dataset ────────────────────────────────────────────
    df_raw, target_col = get_synthetic_dataset()

    # Pre-fix: cast performance_score to numeric (mimics agent fix)
    df_raw["performance_score"] = pd.to_numeric(
        df_raw["performance_score"], errors="coerce"
    )

    # Keep a snapshot of original columns for later comparison
    original_columns: list[str] = list(df_raw.columns)
    original_col_count: int = len(original_columns)

    # ══════════════════════════════════════════════════════════════════════
    # Section 1 — Dataset Info
    # ══════════════════════════════════════════════════════════════════════
    print_section("Section 1 -- Dataset Info")

    print(f"  Shape          : {df_raw.shape}")
    print(f"  Target column  : {target_col}")

    null_counts: pd.Series = df_raw.isnull().sum()
    nulls: pd.Series = null_counts[null_counts > 0]
    print(f"  Null columns   : {len(nulls)}")
    for col_name, cnt in nulls.items():
        print(f"    {col_name:.<30s} {cnt:>5d}  ({cnt / len(df_raw) * 100:.1f}%)")

    # Wrong dtypes (object cols that look numeric)
    wrong_dtype_cols: list[str] = []
    for col in df_raw.select_dtypes(include=["object", "string"]).columns:
        sample: pd.Series = df_raw[col].dropna().head(50)
        try:
            pd.to_numeric(sample)
            wrong_dtype_cols.append(col)
        except (ValueError, TypeError):
            pass
    print(f"  Wrong dtype cols: {wrong_dtype_cols if wrong_dtype_cols else 'None'}")

    salary_skew: float = float(df_raw[target_col].skew())
    print(f"  Target skewness : {salary_skew:.2f}")

    # ══════════════════════════════════════════════════════════════════════
    # Section 2 — Pipeline Execution
    # ══════════════════════════════════════════════════════════════════════
    print_section("Section 2 -- Pipeline Execution")

    print("  Running pipeline...")
    result: dict = {}
    try:
        result = run_pipeline(
            df=df_raw,
            target_col=target_col,
            config=EnvConfig(),
        )
    except Exception:
        print("  FATAL: Pipeline raised an unhandled exception:")
        traceback.print_exc()
        sys.exit(1)

    has_pipeline_error: bool = "pipeline_error" in result
    print(f"  pipeline_error present: {has_pipeline_error}")
    if has_pipeline_error:
        print(f"  pipeline_error value : {result['pipeline_error']}")

    # ══════════════════════════════════════════════════════════════════════
    # Section 3 — Health Report
    # ══════════════════════════════════════════════════════════════════════
    print_section("Section 3 -- Health Report")

    health: dict = result.get("health_report", {})
    expected_health_keys: list[str] = [
        "missing_values", "high_skew_columns", "scale_mismatch",
        "outliers", "class_imbalance", "wrong_dtypes", "dtypes",
    ]
    missing_health_keys: list[str] = [
        k for k in expected_health_keys if k not in health
    ]

    for key in expected_health_keys:
        val = health.get(key, "<MISSING>")
        val_str: str = str(val)
        if len(val_str) > 200:
            val_str = val_str[:200] + "..."
        print(f"  {key:.<30s} {val_str}")

    if missing_health_keys:
        print(f"  WARNING: Missing keys: {missing_health_keys}")

    # ══════════════════════════════════════════════════════════════════════
    # Section 4 — Fix Evaluation
    # ══════════════════════════════════════════════════════════════════════
    print_section("Section 4 -- Fix Evaluation")

    print(f"  task_type  : {result.get('task_type', '<missing>')}")
    print(f"  metric     : {result.get('metric', '<missing>')}")
    print(f"  before     : {result.get('before', '<missing>')}")
    print(f"  after      : {result.get('after', '<missing>')}")
    print(f"  reward     : {result.get('reward', '<missing>')}")
    print(f"  improved   : {result.get('improved', '<missing>')}")

    exec_error = result.get("exec_error")
    print(f"  exec_error : {exec_error}")

    fix_code: str = result.get("fix_code", "")
    preview: str = fix_code[:400] if fix_code else "(empty)"
    print(f"  fix_code (first 400 chars):")
    for line in preview.split("\n"):
        print(f"    {line}")

    # ══════════════════════════════════════════════════════════════════════
    # Section 5 — Feature Engineering
    # ══════════════════════════════════════════════════════════════════════
    print_section("Section 5 -- Feature Engineering")

    feature_log: list[dict] = result.get("feature_log", [])
    print(f"  Total iterations : {len(feature_log)}")

    kept: list[dict] = [e for e in feature_log if e.get("kept", False)]
    discarded: list[dict] = [e for e in feature_log if not e.get("kept", False)]

    print(f"  Features kept    : {len(kept)}")
    for entry in kept:
        print(f"    + {entry.get('feature_name', '?'):.<30s} delta={entry.get('delta', 0)}")

    print(f"  Features discarded: {len(discarded)}")
    for entry in discarded:
        name: str = entry.get("feature_name", "") or "(failed)"
        reason: str = entry.get("reason", "unknown")
        print(f"    - {name:.<30s} {reason}")

    # ══════════════════════════════════════════════════════════════════════
    # Section 6 — Correlation Filtering
    # ══════════════════════════════════════════════════════════════════════
    print_section("Section 6 -- Correlation Filtering")

    drop_report: dict = result.get("drop_report", {})
    print(f"  n_features_before       : {drop_report.get('n_features_before', '<missing>')}")
    print(f"  n_features_after        : {drop_report.get('n_features_after', '<missing>')}")
    print(f"  dropped_low_target_corr : {drop_report.get('dropped_low_target_corr', [])}")
    print(f"  dropped_redundant       : {drop_report.get('dropped_redundant', [])}")
    print(f"  kept_features           : {drop_report.get('kept_features', [])}")

    # ══════════════════════════════════════════════════════════════════════
    # Section 7 — Final Dataset
    # ══════════════════════════════════════════════════════════════════════
    print_section("Section 7 -- Final Dataset")

    final_df: pd.DataFrame = result.get("final_df", pd.DataFrame())
    print(f"  Shape        : {final_df.shape}")
    print(f"  Columns      : {list(final_df.columns)}")

    target_present: bool = target_col in final_df.columns
    print(f"  Target present: {target_present}")

    # Flag numeric-to-object regressions (categoricals are fine)
    original_numeric: set[str] = set(
        df_raw.select_dtypes(include="number").columns
    )
    suspect_object: list[str] = []
    for col in final_df.select_dtypes(include=["object", "string"]).columns:
        if col in original_numeric:
            suspect_object.append(col)
    if suspect_object:
        print(f"  WARNING: Numeric->object regression: {suspect_object}")
    else:
        print(f"  No numeric->object regressions detected")

    # ══════════════════════════════════════════════════════════════════════
    # Section 8 — Critical Checks
    # ══════════════════════════════════════════════════════════════════════
    print_section("Section 8 -- Critical Checks")

    checks: list[tuple[str, bool]] = []

    # Check 1: No pipeline_error
    checks.append((
        "No pipeline_error in result",
        "pipeline_error" not in result,
    ))

    # Check 2: exec_error is None
    checks.append((
        "exec_error is None",
        result.get("exec_error") is None,
    ))

    # Check 3: task_type == "regression"
    checks.append((
        "task_type == 'regression'",
        result.get("task_type") == "regression",
    ))

    # Check 4: reward >= 0.0
    checks.append((
        "reward >= 0.0",
        isinstance(result.get("reward"), (int, float)) and result["reward"] >= 0.0,
    ))

    # Check 5: improved is True or False (not None)
    checks.append((
        "improved is bool (not None)",
        result.get("improved") is True or result.get("improved") is False,
    ))

    # Check 6: age/age_copy redundancy resolved
    dropped_redundant: list[str] = drop_report.get("dropped_redundant", [])
    checks.append((
        "age_copy OR age in dropped_redundant",
        "age_copy" in dropped_redundant or "age" in dropped_redundant,
    ))

    # Check 7: random_noise in dropped_low_target_corr
    dropped_low: list[str] = drop_report.get("dropped_low_target_corr", [])
    checks.append((
        "random_noise in dropped_low_target_corr",
        "random_noise" in dropped_low,
    ))

    # Check 8: target col in final_df
    checks.append((
        "target col present in final_df",
        target_col in final_df.columns,
    ))

    # Check 9: final_df has fewer columns than original
    checks.append((
        "final_df has fewer columns than original",
        len(final_df.columns) < original_col_count,
    ))

    # Check 10: feature_log is a list
    checks.append((
        "feature_log is a list",
        isinstance(result.get("feature_log"), list),
    ))

    # Print results
    passed: int = 0
    for label, ok in checks:
        status: str = "PASS" if ok else "FAIL"
        marker: str = "+" if ok else "X"
        print(f"  [{marker}] {status}: {label}")
        if ok:
            passed += 1

    total: int = len(checks)
    print(f"\n  Summary: {passed}/{total} checks passed")

    if passed < total:
        print("\n  RESULT: FAILED -- some critical checks did not pass")
        sys.exit(1)
    else:
        print("\n  RESULT: ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
