"""
evaluate.py
───────────
Evaluation engine for the Silent Data Debugger.

Contains two complementary evaluation layers:

1. **Task-specific graders** (used by SilentDebuggerEnv / RL loop):
   ``grade_scaling_task``, ``grade_attrition_task``, ``grade_skewed_task``
   Each grader sandboxes the agent's code via exec(), applies AST and
   metric-based checks, and returns ``(reward: float, log: str)``.

2. **Generic evaluator** (used by pipeline.py):
   ``evaluate(df_before, fix_code, target_col, config) → dict``
   Fits a simple baseline model before and after the fix and computes
   a normalised reward based on metric improvement.

Reward scales (task-specific)
──────────────────────────────
scaling   +0.2 exec OK  |  +0.4 scaler detected  |  +0.4 Δacc ≥ threshold
attrition +0.2 exec OK  |  +0.4 nulls cleared    |  +0.4 model trains
skewed    +0.4 log1p    |  +0.3 expm1            |  +0.3 RMSE < threshold
"""

from __future__ import annotations

import ast
import contextlib
import io
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from config import GlobalConfig

_RANDOM_STATE: int = 42
_SCALER_NAMES: frozenset[str] = frozenset(
    {"StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer"}
)


# ════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def execute_agent_code(code_str: str, context: dict) -> tuple[bool, str, dict]:
    """
    Execute *code_str* in a sandboxed ``exec()`` context.

    The caller's *context* is shallow-copied so the original is never mutated.
    Stdout is captured and returned alongside any exception message.

    Parameters
    ----------
    code_str :
        Raw code from the LLM (markdown fences are stripped automatically).
    context :
        Variables injected into the namespace
        (e.g. ``{"df": df, "X_train": X_train, "pd": pd, "np": np}``).

    Returns
    -------
    success : bool
    output  : str   — captured stdout + optional exception text
    namespace : dict — the exec() namespace after execution
    """
    code_str = re.sub(r"```(?:python)?", "", code_str).replace("```", "").strip()
    buf = io.StringIO()
    namespace = dict(context)
    try:
        with contextlib.redirect_stdout(buf):
            exec(code_str, namespace)  # noqa: S102
        return True, buf.getvalue(), namespace
    except Exception as exc:
        output = buf.getvalue() + f"\n[EXECUTION ERROR] {type(exc).__name__}: {exc}"
        return False, output, namespace


def detect_task_type(df: pd.DataFrame, target_col: str) -> str:
    """Return ``"classification"`` or ``"regression"`` based on target cardinality."""
    return "classification" if df[target_col].nunique() <= 10 else "regression"


def strip_markdown(code: str) -> str:
    """Remove ````python … ``` `` fences if present."""
    pattern = r"^```(?:\w+)?\s*\n(.*?)```\s*$"
    match = re.match(pattern, code.strip(), re.DOTALL)
    return match.group(1).strip() if match else code.strip()


def _split(df: pd.DataFrame, target_col: str) -> tuple:
    """Return ``(X_train, X_test, y_train, y_test)`` with a fixed random seed."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=_RANDOM_STATE)


def _extract_func_names(code_str: str) -> list[str] | None:
    """
    Walk the AST and return names of every function call found.
    Returns ``None`` if the code has syntax errors.
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return None
    names: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and hasattr(node, "func")):
            continue
        if isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
        elif isinstance(node.func, ast.Name):
            names.append(node.func.id)
    return names


def _build_exec_context(
    df: pd.DataFrame, target_col: str, *, include_splits: bool = True
) -> tuple[dict, tuple | None]:
    """Build the exec() context dict for *df*, optionally including train/test splits."""
    if include_splits:
        X_train, X_test, y_train, y_test = _split(df, target_col)
        ctx = {
            "df":      df.copy(),
            "X_train": X_train.copy(),
            "X_test":  X_test.copy(),
            "y_train": y_train.copy(),
            "y_test":  y_test.copy(),
            "pd": pd, "np": np,
        }
        return ctx, (X_train, X_test, y_train, y_test)
    return {"df": df.copy(), "pd": pd, "np": np}, None


# ════════════════════════════════════════════════════════════════════════════
#  TASK-SPECIFIC GRADERS  (used by SilentDebuggerEnv)
# ════════════════════════════════════════════════════════════════════════════

def grade_scaling_task(
    agent_code: str,
    df: pd.DataFrame,
    target_col: str,
    baseline_accuracy: float,
    improvement_threshold: float = 0.15,
) -> tuple[float, str]:
    """
    Grade the KNN scale-mismatch fix.

    Reward breakdown:  +0.2 exec OK | +0.4 scaler detected | +0.4 Δacc ≥ threshold
    """
    reward = 0.0
    log: list[str] = []

    context, splits = _build_exec_context(df, target_col)
    _, _, y_train, y_test = splits  # type: ignore[misc]

    success, output, namespace = execute_agent_code(agent_code, context)
    if success:
        reward += 0.2
        log.append("✅ +0.2 Code executed successfully.")
    else:
        log.append(f"❌ +0.0 Code execution failed:\n{output}")
        return round(reward, 2), "\n".join(log)

    func_names = _extract_func_names(agent_code)
    if func_names is not None:
        if _SCALER_NAMES & set(func_names):
            reward += 0.4
            log.append("✅ +0.4 Scaling function detected in agent code.")
        else:
            log.append("❌ +0.0 No recognised scaling function found in agent code.")
    else:
        log.append("❌ +0.0 AST parse failed — agent code has syntax errors.")

    try:
        X_train_s = namespace.get("X_train_scaled", namespace.get("X_train"))
        X_test_s  = namespace.get("X_test_scaled",  namespace.get("X_test"))
        model = KNeighborsClassifier()
        model.fit(X_train_s, y_train)
        new_acc = accuracy_score(y_test, model.predict(X_test_s))
        improvement = new_acc - baseline_accuracy
        log.append(
            f"📊 Baseline: {baseline_accuracy:.4f} | "
            f"New: {new_acc:.4f} | Δ: {improvement:+.4f}"
        )
        if improvement >= improvement_threshold:
            reward += 0.4
            log.append(
                f"✅ +0.4 Accuracy improved by {improvement:.2%} "
                f"(threshold: {improvement_threshold:.0%})."
            )
        else:
            log.append(
                f"❌ +0.0 Improvement {improvement:.2%} below "
                f"threshold {improvement_threshold:.0%}."
            )
    except Exception as exc:
        log.append(f"❌ +0.0 Could not evaluate model accuracy: {exc}")

    return round(min(reward, 1.0), 2), "\n".join(log)


def grade_attrition_task(
    agent_code: str,
    df: pd.DataFrame,
    target_col: str = "Attrition",
    null_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
) -> tuple[float, str]:
    """
    Grade the HR-attrition imputation/encoding fix.

    Reward breakdown:  +0.2 exec OK | +0.4 nulls cleared | +0.4 model trains
    """
    reward = 0.0
    log: list[str] = []
    null_cols = null_cols or []

    context, _ = _build_exec_context(df, target_col, include_splits=False)
    success, output, namespace = execute_agent_code(agent_code, context)

    if success:
        reward += 0.2
        log.append("✅ +0.2 Code executed successfully.")
    else:
        log.append(f"❌ +0.0 Code execution failed:\n{output}")
        return round(reward, 2), "\n".join(log)

    df_fixed: pd.DataFrame = namespace.get("df", df)

    if null_cols:
        still_null = [
            col for col in null_cols
            if col in df_fixed.columns and df_fixed[col].isnull().any()
        ]
        if not still_null:
            reward += 0.4
            log.append(f"✅ +0.4 All null columns imputed: {null_cols}")
        else:
            log.append(f"❌ +0.0 Columns still contain NaNs: {still_null}")
    else:
        if df_fixed.isnull().values.any():
            log.append("❌ +0.0 NaN values still present in DataFrame.")
        else:
            reward += 0.4
            log.append("✅ +0.4 No NaN values remain in DataFrame.")

    try:
        if target_col not in df_fixed.columns:
            log.append(f"❌ +0.0 Target column '{target_col}' missing after fix.")
            return round(min(reward, 1.0), 2), "\n".join(log)

        X = df_fixed.drop(columns=[target_col])
        remaining_cats = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if remaining_cats:
            log.append(f"❌ +0.0 Unencoded categoricals remain: {remaining_cats}")
            return round(min(reward, 1.0), 2), "\n".join(log)

        y = df_fixed[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=_RANDOM_STATE
        )
        clf = RandomForestClassifier(random_state=_RANDOM_STATE)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        reward += 0.4
        log.append(f"✅ +0.4 RandomForest trained successfully. Accuracy: {acc:.4f}")
    except Exception as exc:
        log.append(f"❌ +0.0 Model training failed: {exc}")

    return round(min(reward, 1.0), 2), "\n".join(log)


def grade_skewed_task(
    agent_code: str,
    df: pd.DataFrame,
    target_col: str,
    baseline_rmse: float,
    rmse_threshold: float,
) -> tuple[float, str]:
    """
    Grade the log-normal skew regression fix.

    Reward breakdown:  +0.4 log1p | +0.3 expm1 | +0.3 RMSE < threshold
    """
    reward = 0.0
    log: list[str] = []

    func_names = _extract_func_names(agent_code)
    if func_names is not None:
        if "log1p" in func_names:
            reward += 0.4
            log.append("✅ +0.4 np.log1p detected — log transform applied.")
        else:
            log.append("❌ +0.0 np.log1p not found in agent code.")
        if "expm1" in func_names:
            reward += 0.3
            log.append("✅ +0.3 np.expm1 detected — predictions inverse-transformed.")
        else:
            log.append("❌ +0.0 np.expm1 not found — predictions may be on log scale.")
    else:
        log.append("❌ +0.0 AST parse failed — agent code has syntax errors.")
        return round(reward, 2), "\n".join(log)

    context, splits = _build_exec_context(df, target_col)
    _, _, _, y_test = splits  # type: ignore[misc]

    success, output, namespace = execute_agent_code(agent_code, context)
    if not success:
        log.append(f"❌ +0.0 Code execution failed:\n{output}")
        return round(min(reward, 1.0), 2), "\n".join(log)

    try:
        predictions = namespace.get("predictions") or namespace.get("y_pred")
        if predictions is None:
            log.append("❌ +0.0 No 'predictions' or 'y_pred' variable produced.")
            return round(min(reward, 1.0), 2), "\n".join(log)

        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        log.append(
            f"📊 Baseline RMSE: {baseline_rmse:.2f} | "
            f"New RMSE: {rmse:.2f} | Threshold: {rmse_threshold:.2f}"
        )
        if rmse < rmse_threshold:
            reward += 0.3
            log.append(f"✅ +0.3 RMSE {rmse:.2f} is below threshold {rmse_threshold:.2f}.")
        else:
            log.append(f"❌ +0.0 RMSE {rmse:.2f} did not beat threshold {rmse_threshold:.2f}.")
    except Exception as exc:
        log.append(f"❌ +0.0 Could not compute RMSE: {exc}")

    return round(min(reward, 1.0), 2), "\n".join(log)


# ════════════════════════════════════════════════════════════════════════════
#  GENERIC EVALUATOR  (used by pipeline.py)
# ════════════════════════════════════════════════════════════════════════════

def run_model(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    config: GlobalConfig,
) -> float:
    """
    Fit a simple baseline model and return its evaluation metric.

    * **Regression**      → ``Ridge()``                   → RMSE (lower is better)
    * **Classification**  → ``LogisticRegression()``      → accuracy (higher is better)

    NaN values in features are imputed with column means before fitting.
    """
    df_clean = df.dropna(subset=[target_col]).copy()
    y = df_clean[target_col]
    numeric_cols = [
        c for c in df_clean.select_dtypes(include="number").columns
        if c != target_col and not df_clean[c].isna().all()
    ]
    if not numeric_cols:
        return float(y.std()) if task_type == "regression" and len(y) > 1 else 0.0

    X = df_clean[numeric_cols]
    X_imputed = SimpleImputer(strategy="mean").fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=config.random_state,
    )

    if task_type == "classification":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        return float(accuracy_score(y_test, model.predict(X_test)))

    model_r = Ridge()
    model_r.fit(X_train, y_train)
    return float(np.sqrt(mean_squared_error(y_test, model_r.predict(X_test))))


def evaluate(
    df_before: pd.DataFrame,
    fix_code: str,
    target_col: str,
    config: GlobalConfig,
) -> dict:
    """
    Execute *fix_code* on a copy of *df_before* and measure improvement.

    Returns a structured result dict::

        {
            "task_type":  str,
            "metric":     str,
            "before":     float,
            "after":      float,
            "reward":     float,
            "improved":   bool,
            "fix_code":   str,
            "exec_error": str | None,
            "fixed_df":   pd.DataFrame,
        }
    """
    task_type = detect_task_type(df_before, target_col)
    metric_name = "rmse" if task_type == "regression" else "accuracy"
    before_score = run_model(df_before, target_col, task_type, config)

    cleaned = strip_markdown(fix_code)
    if not cleaned:
        return _error_result(
            task_type, metric_name, before_score, cleaned,
            "Empty fix code string", df_before,
        )

    ctx: dict = {"df": df_before.copy(), "pd": pd, "np": np}
    try:
        exec(cleaned, ctx)  # noqa: S102
        fixed_df = ctx["df"]
    except KeyError:
        return _error_result(
            task_type, metric_name, before_score, cleaned,
            "KeyError: 'df' not found in exec context after execution", df_before,
        )
    except Exception as exc:
        return _error_result(task_type, metric_name, before_score, cleaned, str(exc), df_before)

    after_score = run_model(fixed_df, target_col, task_type, config)

    if task_type == "regression":
        reward = max(0.0, (before_score - after_score) / before_score) if before_score > 0 else 0.0
    else:
        reward = max(0.0, after_score - before_score)

    return {
        "task_type":  task_type,
        "metric":     metric_name,
        "before":     round(before_score, 4),
        "after":      round(after_score, 4),
        "reward":     round(reward, 4),
        "improved":   reward > 0,
        "fix_code":   cleaned,
        "exec_error": None,
        "fixed_df":   fixed_df,
    }


def _error_result(
    task_type: str,
    metric_name: str,
    before_score: float,
    cleaned_code: str,
    error_msg: str,
    df_original: pd.DataFrame,
) -> dict:
    """Build a standard result dict for failed exec() runs."""
    return {
        "task_type":  task_type,
        "metric":     metric_name,
        "before":     round(before_score, 4),
        "after":      round(before_score, 4),
        "reward":     0.0,
        "improved":   False,
        "fix_code":   cleaned_code,
        "exec_error": error_msg,
        "fixed_df":   df_original.copy(),
    }