import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ====================================================================== #
#  VISUALIZE.PY                                                           #
#  Generates Plotly before/after figure dicts for each task.             #
#  Called by env.py and returned inside the obs dict under "figures".    #
#                                                                         #
#  Each public function returns:                                          #
#    {                                                                    #
#      "before": { "dist": fig, "heatmap": fig, ... },                   #
#      "after":  { "dist": fig, "heatmap": fig, ... },                   #
#      "reward_gauge": fig                                                #
#    }                                                                    #
# ====================================================================== #


# ---------------------------------------------------------------------- #
#  SHARED UTILITIES                                                       #
# ---------------------------------------------------------------------- #

def _reward_gauge(reward: float) -> go.Figure:
    """Gauge chart showing current reward score (0.0 → 1.0)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=reward,
        number={"font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1},
            "bar": {"color": "#4CAF50" if reward >= 0.8 else "#FF9800" if reward >= 0.4 else "#F44336"},
            "steps": [
                {"range": [0, 0.4], "color": "#FFEBEE"},
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
    """Heatmap showing missing value positions across the DataFrame."""
    null_matrix = df.isnull().astype(int)
    fig = px.imshow(
        null_matrix,
        color_continuous_scale=["#E8F5E9", "#F44336"],
        aspect="auto",
        title=title,
        labels={"color": "Is Null"},
    )
    fig.update_layout(height=300, margin=dict(t=50, b=20, l=20, r=20))
    fig.update_coloraxes(showscale=False)
    return fig


def _distribution_plot(df_before: pd.DataFrame, df_after: pd.DataFrame, col: str) -> go.Figure:
    """Overlaid KDE-style histogram comparing a column before and after fix."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df_before[col], name="Before", opacity=0.6,
        marker_color="#F44336", nbinsx=40,
        histnorm="probability density",
    ))
    if df_after is not None:
        fig.add_trace(go.Histogram(
            x=df_after[col], name="After", opacity=0.6,
            marker_color="#4CAF50", nbinsx=40,
            histnorm="probability density",
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
    """Bar chart showing class distribution of the target column."""
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
        ranges, x="column", y="range",
        title=title,
        color="range",
        color_continuous_scale="RdYlGn_r",
    )
    fig.update_layout(height=300, margin=dict(t=50, b=40, l=40, r=20))
    return fig


# ---------------------------------------------------------------------- #
#  TASK 1 — SCALING TASK                                                  #
# ---------------------------------------------------------------------- #

def build_scaling_figures(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame | None = None,
    reward: float = 0.0,
    inflated_col: str = "feature_0",
) -> dict:
    """
    Figures for the KNN scaling task.

    Returns:
        {
          "scale_before"  : bar chart of column ranges (broken),
          "scale_after"   : bar chart of column ranges (fixed),
          "dist_before"   : distribution of inflated column (broken),
          "dist_after"    : distribution of inflated column (fixed),
          "reward_gauge"  : reward indicator,
        }
    """
    figures = {}
    figures["scale_before"] = _scale_bar_chart(df_before, "Feature Ranges — Before Fix")
    figures["dist_before"] = _distribution_plot(df_before, None, inflated_col)

    if df_after is not None:
        figures["scale_after"] = _scale_bar_chart(df_after, "Feature Ranges — After Fix")
        figures["dist_after"] = _distribution_plot(df_before, df_after, inflated_col)

    figures["reward_gauge"] = _reward_gauge(reward)
    return figures


# ---------------------------------------------------------------------- #
#  TASK 2 — ATTRITION TASK                                                #
# ---------------------------------------------------------------------- #

def build_attrition_figures(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame | None = None,
    reward: float = 0.0,
    target_col: str = "Attrition",
    null_cols: list | None = None,
) -> dict:
    """
    Figures for the HR attrition imputation task.

    Returns:
        {
          "null_heatmap_before" : missing value heatmap (broken),
          "null_heatmap_after"  : missing value heatmap (fixed),
          "class_balance"       : target class distribution,
          "null_bar"            : null count per column,
          "reward_gauge"        : reward indicator,
        }
    """
    figures = {}
    null_cols = null_cols or []

    figures["null_heatmap_before"] = _null_heatmap(df_before, "Null Map — Before Fix")
    figures["class_balance"] = _class_balance_bar(df_before, target_col, "Class Balance (Attrition)")

    # Null count bar chart
    null_counts = df_before[null_cols].isnull().sum().reset_index() if null_cols else df_before.isnull().sum().reset_index()
    null_counts.columns = ["column", "null_count"]
    null_counts = null_counts[null_counts["null_count"] > 0]
    if not null_counts.empty:
        fig_null_bar = px.bar(
            null_counts, x="column", y="null_count",
            title="Null Count per Column — Before Fix",
            color="null_count", color_continuous_scale="Reds",
        )
        fig_null_bar.update_layout(height=300, margin=dict(t=50, b=40, l=40, r=20))
        figures["null_bar"] = fig_null_bar

    if df_after is not None:
        figures["null_heatmap_after"] = _null_heatmap(df_after, "Null Map — After Fix")

    figures["reward_gauge"] = _reward_gauge(reward)
    return figures


# ---------------------------------------------------------------------- #
#  TASK 3 — SKEWED TASK                                                   #
# ---------------------------------------------------------------------- #

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

    Returns:
        {
          "dist_before"   : target distribution before fix (raw skewed),
          "dist_after"    : target distribution after fix (log-transformed),
          "rmse_compare"  : bar chart comparing baseline vs new RMSE,
          "reward_gauge"  : reward indicator,
        }
    """
    figures = {}

    figures["dist_before"] = _distribution_plot(df_before, None, target_col)

    if df_after is not None:
        figures["dist_after"] = _distribution_plot(df_before, df_after, target_col)

    # RMSE comparison bar
    if baseline_rmse is not None and new_rmse is not None:
        rmse_df = pd.DataFrame({
            "Model": ["Baseline (No Fix)", "Agent Fix"],
            "RMSE": [baseline_rmse, new_rmse],
        })
        fig_rmse = px.bar(
            rmse_df, x="Model", y="RMSE",
            title="RMSE Comparison",
            color="Model",
            color_discrete_map={
                "Baseline (No Fix)": "#F44336",
                "Agent Fix": "#4CAF50",
            },
        )
        fig_rmse.update_layout(height=300, margin=dict(t=50, b=40, l=40, r=20), showlegend=False)
        figures["rmse_compare"] = fig_rmse

    figures["reward_gauge"] = _reward_gauge(reward)
    return figures
