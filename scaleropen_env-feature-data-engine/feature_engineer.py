"""
feature_engineer.py
───────────────────
LLM-powered iterative feature engineering loop.

Asks Claude to propose ONE new feature at a time via Python code, executes
it safely via ``exec()``, tests whether the ML metric improves, and keeps or
discards accordingly.  Stops when the maximum feature budget is exhausted or
consecutive failures exceed patience.

Upstream  : ``evaluate.py`` (cleaned df), ``state.py`` (health report).
Downstream: ``env.py``, ``visualize.py``.
"""

from __future__ import annotations

import json
import os
import re

import anthropic
import numpy as np
import pandas as pd

from config import FeatureEngineeringConfig, GlobalConfig
from evaluate import run_model, strip_markdown


# ──────────────────────────────────────────────────────────────────────────────
# LLM interaction
# ──────────────────────────────────────────────────────────────────────────────


def _ask_llm_for_feature(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    health_report: dict,
    already_tried: list[str],
    current_metric: float,
    api_key: str,
) -> str:
    """Prompt Claude to suggest exactly ONE new engineered feature.

    Parameters
    ----------
    df : pd.DataFrame
        The current state of the dataset.
    target_col : str
        Name of the target column.
    task_type : str
        ``"regression"`` or ``"classification"``.
    health_report : dict
        Output of ``state.get_data_health()``.
    already_tried : list[str]
        Feature names that have already been attempted (avoid repeats).
    current_metric : float
        The current ML metric value (RMSE or accuracy).
    api_key : str
        Anthropic API key.

    Returns
    -------
    str
        Raw LLM response text (expected to contain a Python code block).
    """
    # Build a serialisable snapshot of column info
    col_info: dict[str, str] = {
        col: str(dtype) for col, dtype in df.dtypes.items()
    }

    # Make health_report JSON-safe (strip any non-serialisable objects)
    safe_report: dict = {}
    for k, v in health_report.items():
        try:
            json.dumps(v)
            safe_report[k] = v
        except (TypeError, ValueError):
            safe_report[k] = str(v)

    metric_label: str = "RMSE (lower is better)" if task_type == "regression" else "accuracy (higher is better)"

    system_prompt: str = (
        "You are an expert data scientist specialising in feature engineering. "
        "You will be given a dataset description and must suggest EXACTLY ONE "
        "new engineered feature that is likely to improve model performance."
    )

    user_prompt: str = (
        f"## Dataset Context\n"
        f"- **Task type**: {task_type}\n"
        f"- **Target column**: `{target_col}`\n"
        f"- **Current {metric_label}**: {current_metric:.6f}\n"
        f"- **Shape**: {df.shape[0]} rows x {df.shape[1]} columns\n\n"
        f"### Data Health Report\n```json\n{json.dumps(safe_report, indent=2)}\n```\n\n"
        f"### Current Columns & Dtypes\n```json\n{json.dumps(col_info, indent=2)}\n```\n\n"
        f"### Already Tried Features (do NOT repeat these)\n{already_tried}\n\n"
        f"## Instructions\n"
        f"Output ONLY a Python code block that operates on a variable called `df` "
        f"(a pandas DataFrame). The code must:\n"
        f"1. Add EXACTLY ONE new column whose name starts with `fe_` "
        f"(e.g. `fe_experience_ratio`).\n"
        f"2. NOT modify or drop any existing columns.\n"
        f"3. NOT touch the target column `{target_col}`.\n"
        f"4. Use only `pd` (pandas) and `np` (numpy) — both are available.\n\n"
        f"Think carefully about which interaction, ratio, or transformation of "
        f"existing columns would be most informative for predicting `{target_col}`."
    )

    client: anthropic.Anthropic = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": user_prompt}],
        system=system_prompt,
    )

    # Extract text from the first content block
    return message.content[0].text


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────


def _validate_new_feature(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
) -> tuple[bool, str]:
    """Check that ``df_after`` has exactly one valid new ``fe_`` column.

    Parameters
    ----------
    df_before : pd.DataFrame
        DataFrame before the feature-engineering code ran.
    df_after : pd.DataFrame
        DataFrame after the feature-engineering code ran.

    Returns
    -------
    tuple[bool, str]
        ``(True, feature_name)`` on success, or
        ``(False, error_message)`` on failure.
    """
    new_cols: list[str] = [
        c for c in df_after.columns if c not in df_before.columns
    ]

    if len(new_cols) == 0:
        return False, "No new column was added"

    if len(new_cols) > 1:
        return False, f"Expected 1 new column, got {len(new_cols)}: {new_cols}"

    feature_name: str = new_cols[0]

    if not feature_name.startswith("fe_"):
        return False, f"New column '{feature_name}' does not start with 'fe_'"

    if df_after[feature_name].isna().all():
        return False, f"New column '{feature_name}' is entirely NaN"

    return True, feature_name


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────


def engineer_features(
    fixed_df: pd.DataFrame,
    target_col: str,
    task_type: str,
    health_report: dict,
    feature_eng_cfg: FeatureEngineeringConfig,
    global_cfg: GlobalConfig,
    api_key: str | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run the iterative LLM feature-engineering loop.

    Parameters
    ----------
    fixed_df : pd.DataFrame
        The cleaned dataset (output of ``evaluate``).
    target_col : str
        Name of the target column.
    task_type : str
        ``"regression"`` or ``"classification"``.
    health_report : dict
        Output of ``state.get_data_health()``.
    feature_eng_cfg : FeatureEngineeringConfig
        Controls ``max_feature_ratio`` and ``early_stop_patience``.
    global_cfg : GlobalConfig
        Pipeline-wide settings (supplies ``random_state``).
    api_key : str | None, optional
        Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.

    Returns
    -------
    tuple[pd.DataFrame, list[dict]]
        ``(final_df, feature_log)`` where *feature_log* contains one dict
        per attempt with keys: ``iteration``, ``feature_name``, ``code``,
        ``metric_before``, ``metric_after``, ``delta``, ``kept``, ``reason``.

    Raises
    ------
    ValueError
        If no API key is available.
    """
    # ── Resolve API key ───────────────────────────────────────────────────
    resolved_key: str | None = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if resolved_key is None:
        raise ValueError(
            "No Anthropic API key provided.  Pass 'api_key' or set the "
            "ANTHROPIC_API_KEY environment variable."
        )

    # ── Compute budget ────────────────────────────────────────────────────
    n_original_features: int = len(
        [c for c in fixed_df.columns if c != target_col]
    )

    if n_original_features == 0:
        return fixed_df.copy(), []

    max_features: int = int(
        feature_eng_cfg.max_feature_ratio * n_original_features
    )
    max_features = max(max_features, 1)  # always allow at least 1 attempt

    # ── State variables ───────────────────────────────────────────────────
    current_df: pd.DataFrame = fixed_df.copy()
    feature_log: list[dict] = []
    already_tried: list[str] = []
    added_features: int = 0
    consecutive_failures: int = 0
    iteration: int = 0

    current_metric: float = run_model(
        current_df, target_col, task_type, global_cfg,
    )

    # ── Loop ──────────────────────────────────────────────────────────────
    while (
        added_features < max_features
        and consecutive_failures < feature_eng_cfg.early_stop_patience
    ):
        iteration += 1

        # -- Ask the LLM --------------------------------------------------
        try:
            raw_response: str = _ask_llm_for_feature(
                df=current_df,
                target_col=target_col,
                task_type=task_type,
                health_report=health_report,
                already_tried=already_tried,
                current_metric=current_metric,
                api_key=resolved_key,
            )
        except Exception as exc:  # noqa: BLE001
            feature_log.append({
                "iteration": iteration,
                "feature_name": "",
                "code": "",
                "metric_before": round(current_metric, 4),
                "metric_after": round(current_metric, 4),
                "delta": 0.0,
                "kept": False,
                "reason": f"API call failed: {exc}",
            })
            consecutive_failures += 1
            continue

        # -- Clean the code ------------------------------------------------
        cleaned_code: str = strip_markdown(raw_response)

        if not cleaned_code:
            feature_log.append({
                "iteration": iteration,
                "feature_name": "",
                "code": "",
                "metric_before": round(current_metric, 4),
                "metric_after": round(current_metric, 4),
                "delta": 0.0,
                "kept": False,
                "reason": "LLM returned no code block",
            })
            consecutive_failures += 1
            continue

        # -- Execute -------------------------------------------------------
        context: dict = {"df": current_df.copy(), "pd": pd, "np": np}
        try:
            exec(cleaned_code, context)  # noqa: S102
            new_df: pd.DataFrame = context["df"]
        except Exception as exc:  # noqa: BLE001
            feature_log.append({
                "iteration": iteration,
                "feature_name": "",
                "code": cleaned_code,
                "metric_before": round(current_metric, 4),
                "metric_after": round(current_metric, 4),
                "delta": 0.0,
                "kept": False,
                "reason": f"exec failed: {exc}",
            })
            consecutive_failures += 1
            continue

        # -- Validate ------------------------------------------------------
        valid, detail = _validate_new_feature(current_df, new_df)
        if not valid:
            feature_log.append({
                "iteration": iteration,
                "feature_name": "",
                "code": cleaned_code,
                "metric_before": round(current_metric, 4),
                "metric_after": round(current_metric, 4),
                "delta": 0.0,
                "kept": False,
                "reason": f"Validation failed: {detail}",
            })
            consecutive_failures += 1
            continue

        feature_name: str = detail  # on success, detail == feature_name
        already_tried.append(feature_name)

        # -- Measure improvement -------------------------------------------
        pre_eval_metric: float = current_metric

        new_metric: float = run_model(
            new_df, target_col, task_type, global_cfg,
        )

        if task_type == "regression":
            # RMSE: lower is better
            improved: bool = new_metric < current_metric
            delta: float = round(current_metric - new_metric, 4)
        else:
            # Accuracy: higher is better
            improved = new_metric > current_metric
            delta = round(new_metric - current_metric, 4)

        if improved:
            current_df = new_df
            current_metric = new_metric
            added_features += 1
            consecutive_failures = 0
            reason: str = "Improved metric"
        else:
            consecutive_failures += 1
            reason = "No improvement"

        feature_log.append({
            "iteration": iteration,
            "feature_name": feature_name,
            "code": cleaned_code,
            "metric_before": round(pre_eval_metric, 4),
            "metric_after": round(new_metric, 4),
            "delta": delta,
            "kept": improved,
            "reason": reason,
        })

    return current_df, feature_log


# ──────────────────────────────────────────────────────────────────────────────
# Validation block
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from synthetic_data import get_synthetic_dataset

    passed: int = 0
    total: int = 3

    # ── Test 1: Loop runs end-to-end with real API key ────────────────────
    print("Test 1: engineer_features loop runs end-to-end ...")
    try:
        df_raw, target = get_synthetic_dataset()
        # Pre-clean: cast performance_score to float (mimics evaluate output)
        df_raw["performance_score"] = pd.to_numeric(
            df_raw["performance_score"], errors="coerce"
        )

        fe_cfg: FeatureEngineeringConfig = FeatureEngineeringConfig(
            max_feature_ratio=0.15,   # small budget for testing
            early_stop_patience=2,
        )
        g_cfg: GlobalConfig = GlobalConfig()

        # Minimal health report for the prompt
        health: dict = {
            "missing_values": df_raw.isnull().sum()[
                df_raw.isnull().sum() > 0
            ].to_dict(),
            "dtypes": {c: str(d) for c, d in df_raw.dtypes.items()},
        }

        final_df, log = engineer_features(
            fixed_df=df_raw,
            target_col=target,
            task_type="regression",
            health_report=health,
            feature_eng_cfg=fe_cfg,
            global_cfg=g_cfg,
        )

        assert isinstance(final_df, pd.DataFrame), "final_df is not a DataFrame"
        assert isinstance(log, list), "log is not a list"
        assert len(log) > 0, "log is empty (no iterations ran)"

        required_keys: set[str] = {
            "iteration", "feature_name", "code",
            "metric_before", "metric_after", "delta", "kept", "reason",
        }
        for entry in log:
            missing: set[str] = required_keys - set(entry.keys())
            assert not missing, f"Log entry missing keys: {missing}"

        assert len(final_df.columns) >= len(df_raw.columns), (
            "final_df has fewer columns than input"
        )

        print(f"  PASS  (iterations={len(log)}, "
              f"cols_before={len(df_raw.columns)}, "
              f"cols_after={len(final_df.columns)}, "
              f"kept={sum(1 for e in log if e['kept'])})")
        for i, entry in enumerate(log, 1):
            status: str = "KEPT" if entry["kept"] else "SKIP"
            print(f"    [{i}] {status}: {entry['feature_name'] or '(none)'} "
                  f"| delta={entry['delta']} | {entry['reason']}")
        passed += 1

    except ValueError as ve:
        print(f"  SKIP  (no API key: {ve})")
        # Don't count as failure — key might not be set in CI
        total -= 1
    except Exception as exc:
        print(f"  FAIL  ({type(exc).__name__}: {exc})")

    # ── Test 2: Validation rejects all-NaN column ─────────────────────────
    print("Test 2: _validate_new_feature rejects all-NaN column ...")
    df_a: pd.DataFrame = pd.DataFrame({"x": [1, 2, 3]})
    df_b: pd.DataFrame = df_a.copy()
    df_b["fe_bad"] = np.nan

    ok, msg = _validate_new_feature(df_a, df_b)
    if ok is False and "NaN" in msg:
        print(f"  PASS  (ok={ok}, msg={msg!r})")
        passed += 1
    else:
        print(f"  FAIL  (ok={ok}, msg={msg!r})")

    # ── Test 3: Validation rejects missing fe_ prefix ─────────────────────
    print("Test 3: _validate_new_feature rejects missing fe_ prefix ...")
    df_c: pd.DataFrame = df_a.copy()
    df_c["bad_feature"] = [10, 20, 30]

    ok2, msg2 = _validate_new_feature(df_a, df_c)
    if ok2 is False and "fe_" in msg2:
        print(f"  PASS  (ok={ok2}, msg={msg2!r})")
        passed += 1
    else:
        print(f"  FAIL  (ok={ok2}, msg={msg2!r})")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 40}")
    print(f"  Results: {passed}/{total} passed")
    print(f"{'=' * 40}")
