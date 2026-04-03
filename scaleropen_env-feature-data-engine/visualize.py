"""
visualize.py
────────────
Plotly figure generators for the Silent Data Debugger.

Two sets of helpers live here:

1. **Task-specific builders** (used by SilentDebuggerEnv — imported by env.py):
   ``build_scaling_figures``, ``build_attrition_figures``, ``build_skewed_figures``
   Each returns a ``{key: go.Figure}`` dict that populates ``obs["figures"]``.

2. **Generic UI figures** (used by pipeline.py / Gradio UI):
   ``plot_health_report``, ``plot_before_after``,
   ``plot_feature_engineering``, ``plot_reward_summary``

All figures use either ``plotly_dark`` template (generic) or a light-on-dark
colour palette (task builders) for visual consistency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # noqa: F401 (available for callers)


# ════════════════════════════════════════════════════════════════════════════
#  SHARED PRIMITIVES
# ════════════════════════════════════════════════════════════════════════════

def _reward_gauge(reward: float) -> go.Figure:
    """Gauge chart showing the current reward score (0.0 → 1.0)."""
    bar_color = "#4CAF50" if reward >= 0.8 else "#FF9800" if reward >= 0.4 else "#F44336"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=reward,
        number={"font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1},
            "bar": {"color": bar_color},
            "steps": [
                {"range": [0.0, 0.4], "color": "#FFEBEE"},
                {"range": [0.4, 0.8], "color": "#FFF3E0"},
                {"range": [0.8, 1.0], "color": "#E8F5E9"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": reward,
            },
        },
        title={"text": "Reward Score", "font": {"size": 18}},
    ))
    fig.update_layout(height=250, margin=dict(t=60, b=0, l=20, r=20))
    return fig


def _null_heatmap(df: pd.DataFrame, title: str) -> go.Figure:
    """Heatmap showing missing-value positions across the DataFrame."""
    fig = px.imshow(
        df.isnull().astype(int),
        color_continuous_scale=["#E8F5E9", "#F44336"],
        aspect="auto",
        title=title,
        labels={"color": "Is Null"},
    )
    fig.update_layout(height=300, margin=dict(t=50, b=20, l=20, r=20))
    fig.update_coloraxes(showscale=False)
    return fig


def _distribution_plot(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame | None,
    col: str,
) -> go.Figure:
    """Overlaid histogram comparing a column distribution before and after fix."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df_before[col], name="Before", opacity=0.6,
        marker_color="#F44336", nbinsx=40, histnorm="probability density",
    ))
    if df_after is not None:
        fig.add_trace(go.Histogram(
            x=df_after[col], name="After", opacity=0.6,
            marker_color="#4CAF50", nbinsx=40, histnorm="probability density",
        ))
    fig.update_layout(
        barmode="overlay",
        title=f"Distribution: {col}",
        xaxis_title=col,
        yaxis_title="Density",
        height=300,
        margin=dict(t=50, b=40, l=40, r=20),
        legend=dict(x=0.75, y=0.95),
    )
    return fig


def _class_balance_bar(df: pd.DataFrame, target_col: str, title: str) -> go.Figure:
    """Bar chart showing the class distribution of a target column."""
    counts = df[target_col].value_counts().reset_index()
    counts.columns = [target_col, "count"]
    fig = px.bar(
        counts, x=target_col, y="count",
        color=target_col, title=title,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=300, margin=dict(t=50, b=40, l=40, r=20), showlegend=False)
    return fig


def _scale_bar_chart(df: pd.DataFrame, title: str) -> go.Figure:
    """Bar chart of per-column ranges to visualise scale mismatch."""
    numeric_df = df.select_dtypes(include=[np.number])
    ranges = (numeric_df.max() - numeric_df.min()).reset_index()
    ranges.columns = ["column", "range"]
    fig = px.bar(
        ranges, x="column", y="range", title=title,
        color="range", color_continuous_scale="RdYlGn_r",
    )
    fig.update_layout(height=300, margin=dict(t=50, b=40, l=40, r=20))
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  TASK-SPECIFIC BUILDERS  (imported by env.py → obs["figures"])
# ════════════════════════════════════════════════════════════════════════════

def build_scaling_figures(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame | None = None,
    reward: float = 0.0,
    inflated_col: str = "feature_0",
) -> dict:
    """
    Figures for the KNN scaling task.

    Keys: ``scale_before``, ``scale_after``, ``dist_before``,
          ``dist_after``, ``reward_gauge``.
    """
    figures: dict[str, go.Figure] = {}
    figures["scale_before"] = _scale_bar_chart(df_before, "Feature Ranges — Before Fix")
    figures["dist_before"]  = _distribution_plot(df_before, None, inflated_col)
    if df_after is not None:
        figures["scale_after"] = _scale_bar_chart(df_after, "Feature Ranges — After Fix")
        figures["dist_after"]  = _distribution_plot(df_before, df_after, inflated_col)
    figures["reward_gauge"] = _reward_gauge(reward)
    return figures


def build_attrition_figures(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame | None = None,
    reward: float = 0.0,
    target_col: str = "Attrition",
    null_cols: list | None = None,
) -> dict:
    """
    Figures for the HR attrition imputation task.

    Keys: ``null_heatmap_before``, ``null_heatmap_after``,
          ``class_balance``, ``null_bar``, ``reward_gauge``.
    """
    figures: dict[str, go.Figure] = {}
    null_cols = null_cols or []

    figures["null_heatmap_before"] = _null_heatmap(df_before, "Null Map — Before Fix")
    figures["class_balance"] = _class_balance_bar(
        df_before, target_col, "Class Balance (Attrition)"
    )

    source = df_before[null_cols] if null_cols else df_before
    null_counts = source.isnull().sum().reset_index()
    null_counts.columns = ["column", "null_count"]
    null_counts = null_counts[null_counts["null_count"] > 0]
    if not null_counts.empty:
        fig_bar = px.bar(
            null_counts, x="column", y="null_count",
            title="Null Count per Column — Before Fix",
            color="null_count", color_continuous_scale="Reds",
        )
        fig_bar.update_layout(height=300, margin=dict(t=50, b=40, l=40, r=20))
        figures["null_bar"] = fig_bar

    if df_after is not None:
        figures["null_heatmap_after"] = _null_heatmap(df_after, "Null Map — After Fix")

    figures["reward_gauge"] = _reward_gauge(reward)
    return figures


def build_skewed_figures(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame | None = None,
    reward: float = 0.0,
    target_col: str = "target",
    baseline_rmse: float | None = None,
    new_rmse: float | None = None,
) -> dict:
    """
    Figures for the skewed regression task.

    Keys: ``dist_before``, ``dist_after``, ``rmse_compare``, ``reward_gauge``.
    """
    figures: dict[str, go.Figure] = {}
    figures["dist_before"] = _distribution_plot(df_before, None, target_col)
    if df_after is not None:
        figures["dist_after"] = _distribution_plot(df_before, df_after, target_col)

    if baseline_rmse is not None and new_rmse is not None:
        rmse_df = pd.DataFrame({
            "Model": ["Baseline (No Fix)", "Agent Fix"],
            "RMSE":  [baseline_rmse, new_rmse],
        })
        fig_rmse = px.bar(
            rmse_df, x="Model", y="RMSE", title="RMSE Comparison",
            color="Model",
            color_discrete_map={
                "Baseline (No Fix)": "#F44336",
                "Agent Fix":         "#4CAF50",
            },
        )
        fig_rmse.update_layout(height=300, margin=dict(t=50, b=40, l=40, r=20), showlegend=False)
        figures["rmse_compare"] = fig_rmse

    figures["reward_gauge"] = _reward_gauge(reward)
    return figures


# ════════════════════════════════════════════════════════════════════════════
#  GENERIC UI FIGURES  (used by pipeline.py / Gradio)
# ════════════════════════════════════════════════════════════════════════════

def plot_health_report(health_report: dict) -> go.Figure:
    """Bar chart summarising the number of affected columns per issue type."""
    missing      = len(health_report.get("missing_values", {}))
    skewed       = len(health_report.get("high_skew_columns", {}))
    outliers     = len(health_report.get("outliers", {}))
    wrong_dtypes = len(health_report.get("wrong_dtypes", {}))
    scale        = 1 if health_report.get("scale_mismatch", False) else 0
    imbalance    = 1 if health_report.get("class_imbalance") is not None else 0

    categories = ["Missing Values", "Skewed Columns", "Outliers",
                  "Wrong Dtypes", "Scale Mismatch", "Class Imbalance"]
    values     = [missing, skewed, outliers, wrong_dtypes, scale, imbalance]
    colors     = ["#EF553B" if v > 0 else "#00CC96" for v in values]

    fig = go.Figure(data=[go.Bar(
        x=categories, y=values,
        marker_color=colors, text=values, textposition="outside",
    )])
    fig.update_layout(
        title="Data Health Report",
        yaxis_title="Affected Columns",
        xaxis_title="Issue Type",
        template="plotly_dark",
        showlegend=False,
        bargap=0.3,
    )
    return fig


def plot_before_after(result: dict) -> go.Figure:
    """Grouped bar chart comparing the ML metric before and after the fix."""
    if not result.get("metric"):
        fig = go.Figure()
        fig.update_layout(title="No evaluation results yet", template="plotly_dark")
        return fig

    before    = result.get("before", 0.0)
    after     = result.get("after", 0.0)
    metric    = result.get("metric", "metric")
    task_type = result.get("task_type", "unknown")

    fig = go.Figure(data=[
        go.Bar(name="Before", x=["Metric"], y=[before],
               marker_color="#EF553B", text=[f"{before:.4f}"], textposition="outside"),
        go.Bar(name="After",  x=["Metric"], y=[after],
               marker_color="#00CC96", text=[f"{after:.4f}"],  textposition="outside"),
    ])
    fig.update_layout(
        title=f"Model Performance: Before vs After ({task_type.title()})",
        yaxis_title=metric,
        barmode="group",
        template="plotly_dark",
        bargap=0.3,
    )
    if result.get("improved"):
        fig.add_annotation(
            text="<b>&#10003; Improved</b>", xref="paper", yref="paper",
            x=0.95, y=0.95, showarrow=False,
            font=dict(size=16, color="#00CC96"),
            bgcolor="rgba(0,204,150,0.15)",
            bordercolor="#00CC96", borderwidth=1, borderpad=6,
        )
    return fig


def plot_feature_engineering(feature_log: list[dict]) -> go.Figure:
    """Line chart showing metric progression across feature-engineering iterations."""
    if not feature_log:
        fig = go.Figure()
        fig.update_layout(title="No features engineered", template="plotly_dark")
        return fig

    iterations     = [e["iteration"]         for e in feature_log]
    metric_before  = [e["metric_before"]     for e in feature_log]
    metric_after   = [e["metric_after"]      for e in feature_log]
    kept           = [e.get("kept", False)   for e in feature_log]
    names          = [e.get("feature_name", "") or "(failed)" for e in feature_log]
    marker_colors  = ["#00CC96" if k else "#EF553B" for k in kept]
    marker_symbols = ["circle"  if k else "x"       for k in kept]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iterations, y=metric_before,
        mode="lines+markers", name="Before",
        line=dict(color="gray", dash="dash", width=1.5),
        marker=dict(size=6, color="gray"),
        hovertext=names, hoverinfo="text+y",
    ))
    fig.add_trace(go.Scatter(
        x=iterations, y=metric_after,
        mode="lines+markers", name="After",
        line=dict(color="#636EFA", width=2),
        marker=dict(size=10, color=marker_colors, symbol=marker_symbols,
                    line=dict(width=1.5, color="white")),
        hovertext=[f"{n} ({'kept' if k else 'discarded'})" for n, k in zip(names, kept)],
        hoverinfo="text+y",
    ))
    fig.update_layout(
        title="Feature Engineering Progression",
        xaxis_title="Iteration", yaxis_title="Metric",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_reward_summary(result: dict) -> go.Figure:
    """Gauge chart displaying the agent's reward score (0.0 → 1.0)."""
    reward = result.get("reward", 0.0)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=reward,
        title={"text": "Agent Reward Score", "font": {"size": 20}},
        number={"font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 1.0], "tickwidth": 1},
            "bar": {"color": "#636EFA"},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0,   0.3], "color": "rgba(239,85,59,0.3)"},
                {"range": [0.3, 0.7], "color": "rgba(255,210,76,0.3)"},
                {"range": [0.7, 1.0], "color": "rgba(0,204,150,0.3)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": reward,
            },
        },
    ))
    fig.update_layout(template="plotly_dark", height=350)
    return fig
