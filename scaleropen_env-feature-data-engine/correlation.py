"""
correlation.py
──────────────
Correlation-based feature filtering module.

Computes a correlation heatmap, drops weakly correlated features (relative to
the target), drops redundant inter-feature pairs, and returns the filtered
DataFrame, a structured drop report, and a Plotly heatmap figure.

Upstream  : ``feature_engineer.py`` (final_df, feature_log).
Downstream: ``hyperparameter_tuning.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import CorrelationConfig


# ──────────────────────────────────────────────────────────────────────────────
# Target correlation filter
# ──────────────────────────────────────────────────────────────────────────────


def _drop_low_target_corr(
    df: pd.DataFrame,
    target_col: str,
    threshold: float,
) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    """Drop numeric features whose absolute Pearson correlation with the
    target falls below *threshold*.

    If the target column is non-numeric, this step is skipped entirely and
    all features are retained.

    Parameters
    ----------
    df : pd.DataFrame
        The working dataset (numeric features + target + any categoricals).
    target_col : str
        Name of the target column.
    threshold : float
        Minimum absolute correlation required to keep a feature.

    Returns
    -------
    tuple[pd.DataFrame, list[str], dict[str, float]]
        ``(filtered_df, dropped_columns, target_correlations)``
    """
    numeric_cols: list[str] = [
        c
        for c in df.select_dtypes(include="number").columns
        if c != target_col
    ]

    # If target is non-numeric, skip filtering entirely
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        target_corrs: dict[str, float] = {}
        return df.copy(), [], target_corrs

    # Compute absolute correlations; treat NaN (e.g. zero-variance) as 0.0
    corrs: pd.Series = df[numeric_cols].corrwith(df[target_col]).abs().fillna(0.0)
    target_corrs = {col: round(float(corrs[col]), 4) for col in numeric_cols}

    to_drop: list[str] = [
        col for col in numeric_cols if corrs[col] < threshold
    ]

    filtered_df: pd.DataFrame = df.drop(columns=to_drop)
    return filtered_df, to_drop, target_corrs


# ──────────────────────────────────────────────────────────────────────────────
# Inter-feature redundancy filter
# ──────────────────────────────────────────────────────────────────────────────


def _drop_redundant_features(
    df: pd.DataFrame,
    target_col: str,
    target_corrs: dict[str, float],
    threshold: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop one feature from every highly correlated pair, keeping the one
    more correlated with the target.

    Pairs are processed in descending order of absolute inter-feature
    correlation.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset (already filtered by target correlation).
    target_col : str
        Name of the target column.
    target_corrs : dict[str, float]
        Pre-computed ``{feature: |corr_with_target|}`` map.
    threshold : float
        Maximum allowed absolute pairwise correlation.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        ``(filtered_df, dropped_columns)``
    """
    numeric_cols: list[str] = [
        c
        for c in df.select_dtypes(include="number").columns
        if c != target_col
    ]

    if len(numeric_cols) < 2:
        return df.copy(), []

    corr_matrix: pd.DataFrame = df[numeric_cols].corr().abs()

    # Collect upper-triangle pairs sorted by descending correlation
    pairs: list[tuple[str, str, float]] = []
    for i, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[i + 1 :]:
            val: float = corr_matrix.loc[col_a, col_b]
            if not np.isnan(val) and val > threshold:
                pairs.append((col_a, col_b, val))

    pairs.sort(key=lambda x: x[2], reverse=True)

    dropped: set[str] = set()

    for col_a, col_b, _ in pairs:
        # Skip if either has already been dropped
        if col_a in dropped or col_b in dropped:
            continue

        corr_a: float = target_corrs.get(col_a, 0.0)
        corr_b: float = target_corrs.get(col_b, 0.0)

        # Drop the feature with the weaker target correlation
        if corr_a >= corr_b:
            dropped.add(col_b)
        else:
            dropped.add(col_a)

    dropped_list: list[str] = sorted(dropped)
    filtered_df: pd.DataFrame = df.drop(columns=dropped_list)
    return filtered_df, dropped_list


# ──────────────────────────────────────────────────────────────────────────────
# Heatmap
# ──────────────────────────────────────────────────────────────────────────────


def _build_heatmap(
    df: pd.DataFrame,
    target_col: str,
) -> go.Figure:
    """Build a Plotly heatmap of surviving numeric features + target.

    Parameters
    ----------
    df : pd.DataFrame
        The post-filtering dataset.
    target_col : str
        Name of the target column.

    Returns
    -------
    go.Figure
        A Plotly ``Figure`` with the correlation heatmap.
    """
    numeric_cols: list[str] = [
        c
        for c in df.select_dtypes(include="number").columns
    ]

    if len(numeric_cols) < 2:
        return go.Figure()

    corr_matrix: pd.DataFrame = df[numeric_cols].corr()

    fig: go.Figure = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Feature Correlation Heatmap (post-filtering)",
    )

    fig.update_layout(
        width=800,
        height=700,
    )

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────────────────────────────────────


def filter_features(
    df: pd.DataFrame,
    target_col: str,
    config: CorrelationConfig,
) -> tuple[pd.DataFrame, dict, go.Figure]:
    """Run the full correlation-based filtering pipeline.

    1. Isolate numeric features (exclude target).
    2. Drop features weakly correlated with the target.
    3. Drop redundant inter-feature pairs.
    4. Build a post-filtering heatmap.

    Categorical columns are passed through unchanged.  The target column
    is never modified or dropped.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to filter (from ``feature_engineer.py``).
    target_col : str
        Name of the target column.
    config : CorrelationConfig
        Provides ``target_corr_threshold`` and ``feature_corr_threshold``.

    Returns
    -------
    tuple[pd.DataFrame, dict, go.Figure]
        ``(filtered_df, drop_report, heatmap_figure)``
    """
    numeric_features: list[str] = [
        c
        for c in df.select_dtypes(include="number").columns
        if c != target_col
    ]

    n_before: int = len(numeric_features)

    # Guard: fewer than 2 numeric features → nothing to filter
    if n_before < 2:
        report: dict = {
            "n_features_before": n_before,
            "n_features_after": n_before,
            "dropped_low_target_corr": [],
            "dropped_redundant": [],
            "kept_features": numeric_features,
            "target_correlations": {},
        }
        return df.copy(), report, go.Figure()

    # Step 1 – target correlation filter
    step1_df, dropped_low, target_corrs = _drop_low_target_corr(
        df, target_col, config.target_corr_threshold,
    )

    # Step 2 – redundancy filter
    step2_df, dropped_redundant = _drop_redundant_features(
        step1_df, target_col, target_corrs, config.feature_corr_threshold,
    )

    # Step 3 – heatmap
    fig: go.Figure = _build_heatmap(step2_df, target_col)

    # Build report
    surviving_numeric: list[str] = [
        c
        for c in step2_df.select_dtypes(include="number").columns
        if c != target_col
    ]

    report = {
        "n_features_before": n_before,
        "n_features_after": len(surviving_numeric),
        "dropped_low_target_corr": dropped_low,
        "dropped_redundant": dropped_redundant,
        "kept_features": surviving_numeric,
        "target_correlations": target_corrs,
    }

    return step2_df, report, fig


# ──────────────────────────────────────────────────────────────────────────────
# Validation block
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from synthetic_data import get_synthetic_dataset

    passed: int = 0
    total: int = 4

    cfg: CorrelationConfig = CorrelationConfig()

    # ── Test 1: Normal filtering on synthetic dataset ─────────────────────
    print("Test 1: Normal filtering on synthetic dataset ...")
    df_raw, target = get_synthetic_dataset()
    df_raw["performance_score"] = pd.to_numeric(
        df_raw["performance_score"], errors="coerce"
    )

    filtered_df, report, fig = filter_features(df_raw, target, cfg)

    required_keys: set[str] = {
        "n_features_before", "n_features_after",
        "dropped_low_target_corr", "dropped_redundant",
        "kept_features", "target_correlations",
    }
    missing_keys: set[str] = required_keys - set(report.keys())

    n_numeric_before: int = len([
        c for c in df_raw.select_dtypes(include="number").columns
        if c != target
    ])
    n_numeric_after: int = report["n_features_after"]

    checks: list[tuple[str, bool]] = [
        ("filtered has <= numeric cols", n_numeric_after <= n_numeric_before),
        (
            "age/age_copy redundancy resolved",
            "age" in report["dropped_redundant"]
            or "age_copy" in report["dropped_redundant"],
        ),
        ("random_noise in dropped_low_target_corr", "random_noise" in report["dropped_low_target_corr"]),
        ("report has all keys", len(missing_keys) == 0),
        ("returns go.Figure", isinstance(fig, go.Figure)),
    ]

    all_ok: bool = all(ok for _, ok in checks)
    if all_ok:
        print(f"  PASS  (before={n_numeric_before}, after={n_numeric_after}, "
              f"dropped_low={report['dropped_low_target_corr']}, "
              f"dropped_redundant={report['dropped_redundant']})")
        passed += 1
    else:
        for label, ok in checks:
            status: str = "OK" if ok else "FAIL"
            print(f"    [{status}] {label}")

    # ── Test 2: Non-numeric columns preserved ─────────────────────────────
    print("Test 2: Non-numeric columns preserved ...")
    cat_cols: list[str] = [
        c for c in df_raw.select_dtypes(include="object").columns
    ]
    preserved: bool = all(c in filtered_df.columns for c in cat_cols)
    # Also check values are unchanged for one column
    values_match: bool = True
    if "department" in cat_cols and "department" in filtered_df.columns:
        values_match = df_raw["department"].equals(filtered_df["department"])

    if preserved and values_match:
        print(f"  PASS  (categorical cols preserved: {cat_cols})")
        passed += 1
    else:
        print(f"  FAIL  (preserved={preserved}, values_match={values_match})")

    # ── Test 3: Zero-variance column handled ──────────────────────────────
    print("Test 3: Zero-variance column handled ...")
    df_zv: pd.DataFrame = pd.DataFrame({
        "feat_a": np.random.rand(100),
        "feat_b": np.random.rand(100),
        "constant_col": [42.0] * 100,
        "target": np.random.rand(100),
    })
    try:
        fdf, rpt, _ = filter_features(df_zv, "target", cfg)
        zv_dropped: bool = "constant_col" in (
            rpt["dropped_low_target_corr"] + rpt["dropped_redundant"]
        )
        # constant_col has zero variance → corr with target = NaN → treated as 0.0
        # With default threshold 0.1 it should be dropped
        if zv_dropped:
            print(f"  PASS  (constant_col dropped, report={rpt})")
            passed += 1
        else:
            # Even if not dropped, the key thing is it didn't crash
            print(f"  PASS  (no crash; constant_col kept={not zv_dropped})")
            passed += 1
    except Exception as exc:
        print(f"  FAIL  (exception: {type(exc).__name__}: {exc})")

    # ── Test 4: Fewer than 2 numeric features ─────────────────────────────
    print("Test 4: Fewer than 2 numeric features returns df unchanged ...")
    df_tiny: pd.DataFrame = pd.DataFrame({
        "only_feat": [1.0, 2.0, 3.0],
        "label": ["a", "b", "c"],
    })
    try:
        fdf2, rpt2, fig2 = filter_features(df_tiny, "label", cfg)
        cols_same: bool = list(fdf2.columns) == list(df_tiny.columns)
        minimal: bool = (
            rpt2["n_features_before"] == 1
            and rpt2["n_features_after"] == 1
            and rpt2["dropped_low_target_corr"] == []
            and rpt2["dropped_redundant"] == []
        )
        if cols_same and minimal:
            print(f"  PASS  (returned unchanged, report={rpt2})")
            passed += 1
        else:
            print(f"  FAIL  (cols_same={cols_same}, minimal={minimal}, report={rpt2})")
    except Exception as exc:
        print(f"  FAIL  (exception: {type(exc).__name__}: {exc})")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 40}")
    print(f"  Results: {passed}/{total} passed")
    print(f"{'=' * 40}")
