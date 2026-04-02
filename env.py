"""
env.py
──────
Master pipeline orchestrator for the Silent Data Debugger.

Exposes a single public function :func:`run_pipeline` that sequentially
executes every component (health report → agent fix → evaluation →
feature engineering → correlation filtering), handles errors gracefully,
and returns a master result dictionary.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import (
    AttritionTaskConfig,
    EnvConfig,
    ScalingTaskConfig,
    SkewedTaskConfig,
)
from correlation import filter_features
from evaluate import evaluate
from feature_engineer import engineer_features
from state import get_health_report

# ──────────────────────────────────────────────────────────────────────────────
# Agent import (mock if agent.py doesn't exist yet)
# ──────────────────────────────────────────────────────────────────────────────

try:
    from agent import generate_fix
except ImportError:

    def generate_fix(
        df: pd.DataFrame,
        target_col: str,
        task_type: str,
        health_report: dict,
        api_key: str,
    ) -> str:
        """Mock ``generate_fix`` — placeholder until ``agent.py`` is built."""
        return "df['dummy_fix'] = 1"


# ──────────────────────────────────────────────────────────────────────────────
# Demo data loader
# ──────────────────────────────────────────────────────────────────────────────


def load_demo_data(config: EnvConfig) -> tuple[pd.DataFrame, str]:
    """Generate a fully synthetic dataset based on the demo task config.

    Parameters
    ----------
    config : EnvConfig
        Must have ``mode == "demo"`` and a populated ``task``.

    Returns
    -------
    tuple[pd.DataFrame, str]
        ``(dataframe, target_column_name)``

    Raises
    ------
    ValueError
        If the task config type is unrecognised.
    """
    task = config.task

    # ── ScalingTaskConfig ─────────────────────────────────────────────────
    if isinstance(task, ScalingTaskConfig):
        rng: np.random.Generator = np.random.default_rng(task.random_state)
        n: int = task.n_samples

        data: dict[str, np.ndarray] = {}
        for i in range(task.n_features):
            data[f"feat_{i}"] = rng.standard_normal(n)

        # Inject 1000x scale difference between two features
        data["feat_0"] = data["feat_0"] * 1000.0
        data["feat_1"] = data["feat_1"] * 1.0

        # Binary target
        data["target"] = rng.choice([0, 1], size=n).astype(float)

        return pd.DataFrame(data), "target"

    # ── AttritionTaskConfig ───────────────────────────────────────────────
    if isinstance(task, AttritionTaskConfig):
        rng = np.random.default_rng(task.random_state)
        n = task.n_samples

        age: np.ndarray = rng.integers(22, 60, size=n)
        salary: np.ndarray = rng.integers(25000, 120000, size=n).astype(float)
        satisfaction: np.ndarray = rng.uniform(1.0, 5.0, size=n)
        years_at_company: np.ndarray = rng.integers(0, 30, size=n)
        distance: np.ndarray = rng.integers(1, 50, size=n).astype(float)
        dept: np.ndarray = rng.choice(
            ["Sales", "R&D", "HR", "Engineering"], size=n,
        )
        rating: np.ndarray = np.round(rng.uniform(1.0, 5.0, size=n), 1)

        # Imbalanced binary target
        n_minority: int = int(n * task.imbalance_ratio)
        attrited: np.ndarray = np.zeros(n, dtype=int)
        attrited[:n_minority] = 1
        rng.shuffle(attrited)

        df: pd.DataFrame = pd.DataFrame({
            "age": age,
            "salary": salary,
            "satisfaction": satisfaction,
            "years_at_company": years_at_company,
            "distance": distance,
            "department": dept,
            "rating": rating,
            "attrition": attrited,
        })

        # Inject nulls in 3 columns
        for col, frac in [("satisfaction", 0.08), ("distance", 0.06), ("rating", 0.10)]:
            null_idx: np.ndarray = rng.choice(n, size=int(n * frac), replace=False)
            df.loc[df.index[null_idx], col] = np.nan

        # Inject wrong dtype: cast rating to string
        df["rating"] = df["rating"].astype(str)

        return df, "attrition"

    # ── SkewedTaskConfig ──────────────────────────────────────────────────
    if isinstance(task, SkewedTaskConfig):
        rng = np.random.default_rng(task.random_state)
        n = task.n_samples

        feat_a: np.ndarray = rng.standard_normal(n)
        feat_b: np.ndarray = rng.uniform(0, 10, size=n)
        feat_c: np.ndarray = rng.integers(1, 100, size=n).astype(float)

        # Log-normal target with configurable skew
        sigma: float = task.skew_factor * 0.3
        log_target: np.ndarray = rng.normal(loc=3.0, scale=sigma, size=n)
        target: np.ndarray = np.exp(log_target)

        return pd.DataFrame({
            "feat_a": feat_a,
            "feat_b": feat_b,
            "feat_c": feat_c,
            "target": np.round(target, 2),
        }), "target"

    # ── Unknown task ──────────────────────────────────────────────────────
    raise ValueError(
        f"Unrecognised task config type: {type(task).__name__}. "
        f"Expected ScalingTaskConfig, AttritionTaskConfig, or SkewedTaskConfig."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ──────────────────────────────────────────────────────────────────────────────


def run_pipeline(
    df: pd.DataFrame | None = None,
    target_col: str | None = None,
    config: EnvConfig | None = None,
    api_key: str | None = None,
) -> dict:
    """Execute the full Silent Data Debugger pipeline.

    Runs the following steps in order:

    1. Produce a data health report.
    2. Generate a fix via the LLM agent.
    3. Evaluate the fix (before / after metrics + reward).
    4. Iterative feature engineering.
    5. Correlation-based feature filtering.

    If any step fails, the pipeline returns what it has so far along with
    a ``"pipeline_error"`` key describing the failure.

    Parameters
    ----------
    df : pd.DataFrame | None
        Input dataset.  Ignored when ``config.mode == "demo"``.
    target_col : str | None
        Target column name.  Ignored when ``config.mode == "demo"``.
    config : EnvConfig | None
        Pipeline configuration.  Defaults to ``EnvConfig()`` (upload mode).
    api_key : str | None
        Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.

    Returns
    -------
    dict
        Master result dictionary (see Return Schema in module docstring).
    """
    # ── Handle defaults ───────────────────────────────────────────────────
    if config is None:
        config = EnvConfig()

    resolved_key: str | None = api_key or os.environ.get("ANTHROPIC_API_KEY")

    # ── Default result dict (safe to return early on error) ───────────────
    result: dict = {
        "health_report": {},
        "fix_code": "",
        "exec_error": None,
        "task_type": "",
        "metric": "",
        "before": 0.0,
        "after": 0.0,
        "reward": 0.0,
        "improved": False,
        "feature_log": [],
        "drop_report": {},
        "heatmap_fig": go.Figure(),
        "final_df": pd.DataFrame(),
    }

    try:
        # ── Demo mode: generate data ──────────────────────────────────────
        if config.mode == "demo":
            df, target_col = load_demo_data(config)

        # Safety check
        if df is None or target_col is None:
            raise ValueError(
                "df and target_col must be provided in 'upload' mode."
            )

        result["final_df"] = df  # fallback in case later steps fail

        # ── Step 1: Health report ─────────────────────────────────────────
        task_type: str = "classification" if df[target_col].nunique() <= 10 else "regression"
        result["task_type"] = task_type

        health_report: dict = get_health_report(
            df, config.global_cfg, target_col, task_type,
        )
        result["health_report"] = health_report

        # ── Step 2: Generate fix ──────────────────────────────────────────
        fix_code: str = generate_fix(
            df, target_col, task_type, health_report, resolved_key,
        )
        result["fix_code"] = fix_code

        # ── Step 3: Evaluate fix ──────────────────────────────────────────
        eval_result: dict = evaluate(df, fix_code, target_col, config.global_cfg)

        task_type = eval_result["task_type"]
        result["task_type"] = task_type

        result["exec_error"] = eval_result["exec_error"]
        result["metric"] = eval_result["metric"]
        result["before"] = eval_result["before"]
        result["after"] = eval_result["after"]
        result["reward"] = eval_result["reward"]
        result["improved"] = eval_result["improved"]

        # ── Error guard: skip remaining steps if exec failed ──────────────
        if eval_result["exec_error"] is not None:
            result["final_df"] = df
            return result

        # ── Step 4: Feature engineering ───────────────────────────────────
        engineered_df: pd.DataFrame
        fe_log: list[dict]
        engineered_df, fe_log = engineer_features(
            eval_result["fixed_df"],
            target_col,
            task_type,
            health_report,
            config.feature_eng_cfg,
            config.global_cfg,
            resolved_key,
        )
        result["feature_log"] = fe_log

        # ── Step 5: Correlation filtering ─────────────────────────────────
        final_df: pd.DataFrame
        corr_report: dict
        heatmap_fig: go.Figure
        final_df, corr_report, heatmap_fig = filter_features(
            engineered_df, target_col, config.correlation_cfg,
        )
        result["drop_report"] = corr_report
        result["heatmap_fig"] = heatmap_fig
        result["final_df"] = final_df

        # TODO: hyperparameter_tuning

    except Exception as e:
        result["pipeline_error"] = str(e)

    return result
