"""
evaluate.py
───────────
Evaluation engine for the data-aware AutoML pipeline.

Takes a broken DataFrame and an LLM-generated fix-code string, executes the
fix safely via ``exec()``, measures ML metrics before and after, and returns
a structured result dict with a reward score bounded to [0.0, 1.0].

Downstream consumers: ``feature_engineer.py``, ``env.py``, ``visualize.py``.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from config import GlobalConfig


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def detect_task_type(df: pd.DataFrame, target_col: str) -> str:
    """Infer ``"classification"`` or ``"regression"`` from target cardinality.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset (may contain nulls in the target).
    target_col : str
        Name of the target column.

    Returns
    -------
    str
        ``"classification"`` if ``df[target_col].nunique() <= 10``,
        otherwise ``"regression"``.
    """
    return "classification" if df[target_col].nunique() <= 10 else "regression"


def strip_markdown(code: str) -> str:
    """Remove ````python … ``` `` or ```` ``` … ``` `` fences if present.

    Parameters
    ----------
    code : str
        Raw code string, potentially wrapped in Markdown fences.

    Returns
    -------
    str
        The cleaned code body.
    """
    # Match ```python ... ``` or ``` ... ``` (with optional language tag)
    pattern: str = r"^```(?:\w+)?\s*\n(.*?)```\s*$"
    match: re.Match[str] | None = re.match(pattern, code.strip(), re.DOTALL)
    return match.group(1).strip() if match else code.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Model runner
# ──────────────────────────────────────────────────────────────────────────────


def run_model(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    config: GlobalConfig,
) -> float:
    """Fit a simple model and return the evaluation metric.

    * **Regression** → ``Ridge()`` → RMSE (lower is better).
    * **Classification** → ``LogisticRegression(max_iter=1000)`` → accuracy
      (higher is better).

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to evaluate.
    target_col : str
        Name of the target column.
    task_type : str
        ``"regression"`` or ``"classification"``.
    config : GlobalConfig
        Pipeline configuration (supplies ``random_state``).

    Returns
    -------
    float
        RMSE for regression, accuracy for classification.
    """
    # Drop rows where the target is null
    df_clean: pd.DataFrame = df.dropna(subset=[target_col]).copy()

    y: pd.Series = df_clean[target_col]

    # Select numeric feature columns (exclude target)
    numeric_cols: list[str] = [
        c
        for c in df_clean.select_dtypes(include="number").columns
        if c != target_col
    ]

    # Drop columns that are 100 % NaN to prevent imputer failure
    numeric_cols = [
        c for c in numeric_cols if not df_clean[c].isna().all()
    ]

    if len(numeric_cols) == 0:
        # Fallback: return worst-case metric so reward stays at 0
        if task_type == "regression":
            return float(y.std()) if len(y) > 1 else 1.0
        return 0.0  # accuracy = 0 for classification

    X: pd.DataFrame = df_clean[numeric_cols]

    # Impute remaining NaNs with column means
    imputer: SimpleImputer = SimpleImputer(strategy="mean")
    X_imputed: np.ndarray = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=config.random_state,
    )

    if task_type == "classification":
        model: LogisticRegression = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds: np.ndarray = model.predict(X_test)
        return float(accuracy_score(y_test, preds))

    # Regression
    model_r: Ridge = Ridge()
    model_r.fit(X_train, y_train)
    preds = model_r.predict(X_test)
    rmse: float = float(np.sqrt(mean_squared_error(y_test, preds)))
    return rmse


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluator
# ──────────────────────────────────────────────────────────────────────────────


def evaluate(
    df_before: pd.DataFrame,
    fix_code: str,
    target_col: str,
    config: GlobalConfig,
) -> dict:
    """Execute *fix_code* on a copy of *df_before* and measure improvement.

    Parameters
    ----------
    df_before : pd.DataFrame
        The "broken" dataset (with impurities).
    fix_code : str
        LLM-generated Python code that mutates a variable named ``df``.
    target_col : str
        Name of the target column.
    config : GlobalConfig
        Pipeline configuration.

    Returns
    -------
    dict
        Structured evaluation result.  Schema::

            {
                "task_type":   str,
                "metric":      str,
                "before":      float,
                "after":       float,
                "reward":      float,
                "improved":    bool,
                "fix_code":    str,
                "exec_error":  str | None,
                "fixed_df":    pd.DataFrame,
            }
    """
    # ── Auto-detect task type ─────────────────────────────────────────────
    task_type: str = detect_task_type(df_before, target_col)
    metric_name: str = "rmse" if task_type == "regression" else "accuracy"

    # ── Baseline metric (before fix) ──────────────────────────────────────
    before_score: float = run_model(df_before, target_col, task_type, config)

    # ── Clean the code ────────────────────────────────────────────────────
    cleaned_code: str = strip_markdown(fix_code)

    # ── Guard: empty code string ──────────────────────────────────────────
    if not cleaned_code:
        return {
            "task_type": task_type,
            "metric": metric_name,
            "before": round(before_score, 4),
            "after": round(before_score, 4),
            "reward": 0.0,
            "improved": False,
            "fix_code": cleaned_code,
            "exec_error": "Empty fix code string",
            "fixed_df": df_before.copy(),
        }

    # ── Execute the fix ───────────────────────────────────────────────────
    context: dict = {"df": df_before.copy(), "pd": pd, "np": np}

    try:
        exec(cleaned_code, context)  # noqa: S102
        fixed_df: pd.DataFrame = context["df"]
    except KeyError:
        # The code ran but removed / renamed "df" from the namespace
        return _error_result(
            task_type, metric_name, before_score, cleaned_code,
            "KeyError: 'df' not found in exec context after execution",
            df_before,
        )
    except Exception as exc:  # noqa: BLE001
        return _error_result(
            task_type, metric_name, before_score, cleaned_code,
            str(exc), df_before,
        )

    # ── Post-fix metric ───────────────────────────────────────────────────
    after_score: float = run_model(fixed_df, target_col, task_type, config)

    # ── Reward ────────────────────────────────────────────────────────────
    if task_type == "regression":
        # RMSE: lower is better → relative decrease
        reward: float = max(0.0, (before_score - after_score) / before_score) if before_score > 0 else 0.0
    else:
        # Accuracy: higher is better → absolute increase
        reward = max(0.0, after_score - before_score)

    reward = round(reward, 4)

    return {
        "task_type": task_type,
        "metric": metric_name,
        "before": round(before_score, 4),
        "after": round(after_score, 4),
        "reward": reward,
        "improved": reward > 0,
        "fix_code": cleaned_code,
        "exec_error": None,
        "fixed_df": fixed_df,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _error_result(
    task_type: str,
    metric_name: str,
    before_score: float,
    cleaned_code: str,
    error_msg: str,
    df_original: pd.DataFrame,
) -> dict:
    """Build a standard result dict for failed ``exec()`` runs."""
    return {
        "task_type": task_type,
        "metric": metric_name,
        "before": round(before_score, 4),
        "after": round(before_score, 4),
        "reward": 0.0,
        "improved": False,
        "fix_code": cleaned_code,
        "exec_error": error_msg,
        "fixed_df": df_original.copy(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Validation block
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from synthetic_data import get_synthetic_dataset

    cfg: GlobalConfig = GlobalConfig()
    passed: int = 0
    total: int = 3

    # ── Test 1: Regression fix works ──────────────────────────────────────
    print("Test 1: Regression fix produces reward > 0 …")
    df, target = get_synthetic_dataset()
    fix: str = (
        "df['performance_score'] = pd.to_numeric(df['performance_score'], errors='coerce')\n"
        "df['satisfaction_score'] = df['satisfaction_score'].fillna(df['satisfaction_score'].median())\n"
    )
    result: dict = evaluate(df, fix, target, cfg)
    if result["reward"] > 0 and result["improved"] is True:
        print(f"  PASS  (reward={result['reward']}, before={result['before']}, after={result['after']})")
        passed += 1
    else:
        print(f"  FAIL  (reward={result['reward']}, improved={result['improved']}, "
              f"exec_error={result['exec_error']})")

    # ── Test 2: exec() failure handled ────────────────────────────────────
    print("Test 2: exec() failure returns exec_error …")
    result2: dict = evaluate(df, "df = this_does_not_exist", target, cfg)
    if result2["exec_error"] is not None and result2["reward"] == 0.0:
        print(f"  PASS  (exec_error={result2['exec_error']!r})")
        passed += 1
    else:
        print(f"  FAIL  (exec_error={result2['exec_error']}, reward={result2['reward']})")

    # ── Test 3: Classification detection ──────────────────────────────────
    print("Test 3: detect_task_type returns 'classification' …")
    dummy: pd.DataFrame = pd.DataFrame({
        "feat": np.random.rand(100),
        "label": np.random.choice([0, 1], size=100),
    })
    detected: str = detect_task_type(dummy, "label")
    if detected == "classification":
        print(f"  PASS  (detected={detected!r})")
        passed += 1
    else:
        print(f"  FAIL  (detected={detected!r})")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 40}")
    print(f"  Results: {passed}/{total} passed")
    print(f"{'=' * 40}")
