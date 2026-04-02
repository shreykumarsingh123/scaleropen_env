"""
visualize.py
────────────
Plotly figure generators for the Silent Data Debugger Gradio UI.

Every function consumes part of the master result dict returned by
``env.run_pipeline()`` and produces a ``go.Figure`` ready for display.

All figures use the ``plotly_dark`` template.
"""

from __future__ import annotations

import plotly.graph_objects as go


# ──────────────────────────────────────────────────────────────────────────────
# 1. Health report bar chart
# ──────────────────────────────────────────────────────────────────────────────


def plot_health_report(health_report: dict) -> go.Figure:
    """Bar chart summarising the number of affected columns per issue type.

    Parameters
    ----------
    health_report : dict
        Output of ``state.get_data_health()`` / ``state.get_health_report()``.

    Returns
    -------
    go.Figure
        A bar chart with one bar per issue category.
    """
    missing: int = len(health_report.get("missing_values", {}))
    skewed: int = len(health_report.get("high_skew_columns", {}))
    outliers: int = len(health_report.get("outliers", {}))
    wrong_dtypes: int = len(health_report.get("wrong_dtypes", {}))
    scale: int = 1 if health_report.get("scale_mismatch", False) else 0
    imbalance: int = 1 if health_report.get("class_imbalance") is not None else 0

    categories: list[str] = [
        "Missing Values",
        "Skewed Columns",
        "Outliers",
        "Wrong Dtypes",
        "Scale Mismatch",
        "Class Imbalance",
    ]
    values: list[int] = [missing, skewed, outliers, wrong_dtypes, scale, imbalance]
    colors: list[str] = [
        "#EF553B" if v > 0 else "#00CC96" for v in values
    ]

    fig: go.Figure = go.Figure(
        data=[
            go.Bar(
                x=categories,
                y=values,
                marker_color=colors,
                text=values,
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title="Data Health Report",
        yaxis_title="Affected Columns",
        xaxis_title="Issue Type",
        template="plotly_dark",
        showlegend=False,
        bargap=0.3,
    )

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 2. Before vs After grouped bar chart
# ──────────────────────────────────────────────────────────────────────────────


def plot_before_after(result: dict) -> go.Figure:
    """Grouped bar chart comparing the ML metric before and after the fix.

    Parameters
    ----------
    result : dict
        Master result dict from ``env.run_pipeline()``.

    Returns
    -------
    go.Figure
        A grouped bar chart with Before and After bars.
    """
    before: float = result.get("before", 0.0)
    after: float = result.get("after", 0.0)

    # Empty state
    if result.get("metric", "") == "":
        fig: go.Figure = go.Figure()
        fig.update_layout(
            title="No evaluation results yet",
            template="plotly_dark",
        )
        return fig

    metric: str = result.get("metric", "metric")
    task_type: str = result.get("task_type", "unknown")

    fig = go.Figure(
        data=[
            go.Bar(
                name="Before",
                x=["Metric"],
                y=[before],
                marker_color="#EF553B",
                text=[f"{before:.4f}"],
                textposition="outside",
            ),
            go.Bar(
                name="After",
                x=["Metric"],
                y=[after],
                marker_color="#00CC96",
                text=[f"{after:.4f}"],
                textposition="outside",
            ),
        ]
    )

    fig.update_layout(
        title=f"Model Performance: Before vs After Fix ({task_type.title()})",
        yaxis_title=metric,
        barmode="group",
        template="plotly_dark",
        bargap=0.3,
    )

    # Add improvement annotation
    if result.get("improved", False):
        fig.add_annotation(
            text="<b>&#10003; Improved</b>",
            xref="paper",
            yref="paper",
            x=0.95,
            y=0.95,
            showarrow=False,
            font=dict(size=16, color="#00CC96"),
            bgcolor="rgba(0,204,150,0.15)",
            bordercolor="#00CC96",
            borderwidth=1,
            borderpad=6,
        )

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 3. Feature engineering progression
# ──────────────────────────────────────────────────────────────────────────────


def plot_feature_engineering(feature_log: list[dict]) -> go.Figure:
    """Line chart showing metric progression across feature-engineering
    iterations.

    Parameters
    ----------
    feature_log : list[dict]
        The ``feature_log`` list from ``engineer_features()``.

    Returns
    -------
    go.Figure
        A line chart with before (dashed) and after (solid) metric traces.
    """
    # Empty state
    if not feature_log:
        fig: go.Figure = go.Figure()
        fig.update_layout(
            title="No features engineered",
            template="plotly_dark",
        )
        return fig

    iterations: list[int] = [e["iteration"] for e in feature_log]
    metric_before: list[float] = [e["metric_before"] for e in feature_log]
    metric_after: list[float] = [e["metric_after"] for e in feature_log]
    kept: list[bool] = [e.get("kept", False) for e in feature_log]
    names: list[str] = [e.get("feature_name", "") or "(failed)" for e in feature_log]

    # Marker colors and symbols based on kept status
    marker_colors: list[str] = [
        "#00CC96" if k else "#EF553B" for k in kept
    ]
    marker_symbols: list[str] = [
        "circle" if k else "x" for k in kept
    ]

    fig = go.Figure()

    # Baseline trace (metric_before): dashed gray
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=metric_before,
            mode="lines+markers",
            name="Before",
            line=dict(color="gray", dash="dash", width=1.5),
            marker=dict(size=6, color="gray"),
            hovertext=names,
            hoverinfo="text+y",
        )
    )

    # After trace: solid, colored markers
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=metric_after,
            mode="lines+markers",
            name="After",
            line=dict(color="#636EFA", width=2),
            marker=dict(
                size=10,
                color=marker_colors,
                symbol=marker_symbols,
                line=dict(width=1.5, color="white"),
            ),
            hovertext=[
                f"{n} ({'kept' if k else 'discarded'})"
                for n, k in zip(names, kept)
            ],
            hoverinfo="text+y",
        )
    )

    fig.update_layout(
        title="Feature Engineering Progression",
        xaxis_title="Iteration",
        yaxis_title="Metric",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 4. Reward gauge
# ──────────────────────────────────────────────────────────────────────────────


def plot_reward_summary(result: dict) -> go.Figure:
    """Gauge chart displaying the agent's reward score (0.0 → 1.0).

    Parameters
    ----------
    result : dict
        Master result dict from ``env.run_pipeline()``.

    Returns
    -------
    go.Figure
        A gauge chart with red / yellow / green colour steps.
    """
    reward: float = result.get("reward", 0.0)

    fig: go.Figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=reward,
            title={"text": "Agent Reward Score", "font": {"size": 20}},
            number={"font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 1.0], "tickwidth": 1},
                "bar": {"color": "#636EFA"},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 0.3], "color": "rgba(239,85,59,0.3)"},
                    {"range": [0.3, 0.7], "color": "rgba(255,210,76,0.3)"},
                    {"range": [0.7, 1.0], "color": "rgba(0,204,150,0.3)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.8,
                    "value": reward,
                },
            },
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=350,
    )

    return fig
