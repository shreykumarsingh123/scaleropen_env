"""
pipeline.py
───────────
Full pipeline orchestrator for the upload-mode / Gradio UI path.

Runs the following steps in order:
  1. Health report   (state.get_health_report)
  2. LLM fix         (agent.generate_fix or mock)
  3. Evaluation      (evaluate.evaluate)
  4. Feature engineering (feature_engineer.engineer_features)
  5. Correlation filtering (correlation.filter_features)

A fallback ``generate_fix`` mock is provided when ``agent.py`` cannot be
imported (e.g. no API key configured at import time).

Returns
───────
dict with keys:
  health_report, fix_code, exec_error, task_type, metric,
  before, after, reward, improved, feature_log, drop_report,
  heatmap_fig, final_df
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import (
    AttritionTaskConfig,
    EnvConfig,
    FeatureEngineeringConfig,
    GlobalConfig,
    ScalingTaskConfig,
    SkewedTaskConfig,
)
from correlation import filter_features
from evaluate import evaluate
from feature_engineer import engineer_features
from state import get_health_report

# ── Agent import with fallback ───────────────────────────────────────────────
try:
    from agent import generate_fix  # type: ignore[import]
except ImportError:
    def generate_fix(  # type: ignore[misc]
        df: pd.DataFrame,
        target_col: str,
        task_type: str,
        health_report: dict,
        api_key: str | None = None,
    ) -> str:
        """Placeholder until agent.py is configured with an API key."""
        return "# No LLM agent configured — fix code not generated"


# ── Demo data loader ─────────────────────────────────────────────────────────

def load_demo_data(config: EnvConfig) -> tuple[pd.DataFrame, str]:
    """
    Generate a fully synthetic demo dataset from the task config.

    Parameters
    ----------
    config : EnvConfig
        Must have a ``task`` field that is one of
        ``"scaling"``, ``"attrition"``, or ``"skewed"``.

    Returns
    -------
    tuple[pd.DataFrame, str]
        ``(dataframe, target_column_name)``
    """
    task = config.task

    if task == "scaling":
        cfg: ScalingTaskConfig = config.scaling_cfg
        rng = np.random.default_rng(cfg.random_state)
        n = cfg.n_samples
        data: dict[str, np.ndarray] = {
            f"feat_{i}": rng.standard_normal(n) for i in range(cfg.n_features)
        }
        data["feat_0"] = data["feat_0"] * cfg.scale_factor
        data["target"] = rng.choice([0, 1], size=n).astype(float)
        return pd.DataFrame(data), "target"

    if task == "attrition":
        cfg_a: AttritionTaskConfig = config.attrition_cfg
        rng = np.random.default_rng(cfg_a.random_state)
        n = cfg_a.n_samples
        df = pd.DataFrame({
            "Age":    rng.integers(22, 60, size=n),
            "Salary": rng.integers(25_000, 120_000, size=n).astype(float),
            "Satisfaction":     rng.uniform(1.0, 5.0, size=n),
            "YearsAtCompany":   rng.integers(0, 30, size=n),
            "Distance":         rng.integers(1, 50, size=n).astype(float),
            "Department":       rng.choice(["Sales", "R&D", "HR", "Engineering"], size=n),
            "JobRole":          rng.choice(["Manager", "Analyst", "Engineer", "Clerk"], size=n),
            "OverTime":         rng.choice(["Yes", "No"], size=n),
            cfg_a.target_col:   rng.choice([0, 1], size=n, p=[0.84, 0.16]),
        })
        for col in cfg_a.null_cols:
            if col in df.columns:
                idx = rng.choice(n, size=int(n * cfg_a.null_rate), replace=False)
                df.loc[df.index[idx], col] = np.nan
        return df, cfg_a.target_col

    if task == "skewed":
        cfg_s: SkewedTaskConfig = config.skewed_cfg
        rng = np.random.default_rng(cfg_s.random_state)
        n = cfg_s.n_samples
        X = (rng.standard_normal((n, cfg_s.n_features))
             + rng.standard_normal((n, cfg_s.n_features)) * cfg_s.noise_level)
        y = np.exp(cfg_s.skew_param * rng.standard_normal(n))
        cols = [f"feature_{i}" for i in range(cfg_s.n_features)]
        df = pd.DataFrame(X, columns=cols)
        df["target"] = y
        return df, "target"

    raise ValueError(
        f"Unknown task '{task}'. Expected: 'scaling', 'attrition', or 'skewed'."
    )


# ── Master pipeline ───────────────────────────────────────────────────────────

def run_pipeline(
    df: pd.DataFrame | None = None,
    target_col: str | None = None,
    config: EnvConfig | None = None,
    api_key: str | None = None,
) -> dict:
    """
    Execute the full Silent Data Debugger pipeline.

    Steps
    -----
    1. Health report   → state.get_health_report
    2. LLM fix code    → agent.generate_fix
    3. Evaluate fix    → evaluate.evaluate
    4. Feature engineering → feature_engineer.engineer_features
    5. Correlation filter  → correlation.filter_features

    If any step fails the pipeline returns what it has accumulated so far,
    plus a ``"pipeline_error"`` key describing the failure.

    Parameters
    ----------
    df :
        Input dataset.  Not needed when ``config.task`` is set.
    target_col :
        Target column name.  Not needed in demo/task mode.
    config :
        Pipeline config.  Defaults to ``EnvConfig()`` (scaling task).
    api_key :
        Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.

    Returns
    -------
    dict
        Master result dictionary.
    """
    config = config or EnvConfig()
    resolved_key: str | None = api_key or os.environ.get("ANTHROPIC_API_KEY")

    result: dict = {
        "health_report": {},
        "fix_code":      "",
        "exec_error":    None,
        "task_type":     "",
        "metric":        "",
        "before":        0.0,
        "after":         0.0,
        "reward":        0.0,
        "improved":      False,
        "feature_log":   [],
        "drop_report":   {},
        "heatmap_fig":   go.Figure(),
        "final_df":      pd.DataFrame(),
    }

    try:
        # Demo task mode: generate data from config
        if config.task in ("scaling", "attrition", "skewed"):
            df, target_col = load_demo_data(config)

        if df is None or target_col is None:
            raise ValueError("df and target_col must be provided for custom datasets.")

        result["final_df"] = df

        # Step 1: health report
        task_type = "classification" if df[target_col].nunique() <= 10 else "regression"
        result["task_type"] = task_type

        health_report = get_health_report(
            df, config.global_cfg, target_col, task_type,
        )
        result["health_report"] = health_report

        # Step 2: LLM fix
        fix_code = generate_fix(df, target_col, task_type, health_report, resolved_key)
        result["fix_code"] = fix_code

        # Step 3: evaluate
        eval_result = evaluate(df, fix_code, target_col, config.global_cfg)
        result["task_type"]  = eval_result["task_type"]
        result["exec_error"] = eval_result["exec_error"]
        result["metric"]     = eval_result["metric"]
        result["before"]     = eval_result["before"]
        result["after"]      = eval_result["after"]
        result["reward"]     = eval_result["reward"]
        result["improved"]   = eval_result["improved"]

        if eval_result["exec_error"] is not None:
            result["final_df"] = df
            return result

        # Step 4: feature engineering
        fe_config = FeatureEngineeringConfig()
        engineered_df, fe_log = engineer_features(
            eval_result["fixed_df"], target_col, task_type,
            health_report, fe_config, config.global_cfg, resolved_key,
        )
        result["feature_log"] = fe_log

        # Step 5: correlation filtering
        final_df, corr_report, heatmap_fig = filter_features(
            engineered_df, target_col, config.correlation_cfg if hasattr(config, "correlation_cfg") else None,
        )
        result["drop_report"] = corr_report
        result["heatmap_fig"] = heatmap_fig
        result["final_df"]    = final_df

    except Exception as exc:
        result["pipeline_error"] = str(exc)

    return result
