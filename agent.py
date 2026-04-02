"""
agent.py
────────
LLM agent that reads a data health report and generates pandas/numpy
fix code to repair a broken DataFrame.

The generated code operates on a variable called ``df`` and is designed
to be executed via ``exec()`` inside ``evaluate.py``.
"""

from __future__ import annotations

import json
import os
import re

import anthropic
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _strip_markdown(code: str) -> str:
    """Remove ```python … ``` or ``` … ``` fences if present.

    Parameters
    ----------
    code : str
        Raw code string, potentially wrapped in Markdown fences.

    Returns
    -------
    str
        The cleaned code body.
    """
    pattern: str = r"^```(?:\w+)?\s*\n(.*?)```\s*$"
    match: re.Match[str] | None = re.match(pattern, code.strip(), re.DOTALL)
    return match.group(1).strip() if match else code.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Main agent
# ──────────────────────────────────────────────────────────────────────────────


def generate_fix(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    health_report: dict,
    api_key: str | None = None,
) -> str:
    """Generate Python fix code for a broken DataFrame using Claude.

    The returned code string is standalone pandas/numpy code that mutates
    a variable named ``df``.  It is designed to be passed to
    ``evaluate.evaluate()`` for execution via ``exec()``.

    Parameters
    ----------
    df : pd.DataFrame
        The broken dataset (used for context, not mutated here).
    target_col : str
        Name of the target column (must not be touched by the fix).
    task_type : str
        ``"regression"`` or ``"classification"``.
    health_report : dict
        Output of ``state.get_data_health()`` / ``state.get_health_report()``.
    api_key : str | None, optional
        Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.

    Returns
    -------
    str
        Clean Python code string (markdown fences stripped).

    Raises
    ------
    ValueError
        If no API key is available.
    Exception
        If the Anthropic API call fails.
    """
    # ── Resolve API key ───────────────────────────────────────────────────
    resolved_key: str | None = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if resolved_key is None:
        raise ValueError(
            "No Anthropic API key provided.  Pass 'api_key' or set the "
            "ANTHROPIC_API_KEY environment variable."
        )

    # ── Make health_report JSON-safe ──────────────────────────────────────
    safe_report: dict = {}
    for k, v in health_report.items():
        try:
            json.dumps(v)
            safe_report[k] = v
        except (TypeError, ValueError):
            safe_report[k] = str(v)

    # ── Build prompts ─────────────────────────────────────────────────────
    system_prompt: str = (
        "You are an expert data scientist. Your task is to write clean, "
        "standalone Python code using ONLY pandas and numpy to fix data "
        "quality issues in a DataFrame. Do NOT import or use sklearn. "
        "Do NOT use any external libraries beyond pandas (pd) and numpy (np). "
        "The code must operate on a variable called `df` which is a "
        "pandas DataFrame already available in scope."
    )

    user_prompt: str = (
        f"## Task\n"
        f"Fix the data quality issues described in the health report below.\n\n"
        f"- **Target column**: `{target_col}` (do NOT modify this column)\n"
        f"- **Task type**: {task_type}\n\n"
        f"## Health Report\n"
        f"```json\n{json.dumps(safe_report, indent=2)}\n```\n\n"
        f"## Fix Instructions\n"
        f"Write a single Python code block that performs ALL applicable fixes:\n\n"
        f"1. **missing_values**: Impute with median (numeric) or mode "
        f"(categorical), or drop rows if < 5% missing. Do NOT touch "
        f"`{target_col}`.\n"
        f"2. **high_skew_columns**: Apply `np.log1p()` to right-skewed "
        f"numeric columns. Skip `{target_col}`.\n"
        f"3. **scale_mismatch**: If True, standardise numeric features "
        f"manually: `(col - col.mean()) / col.std()`. Skip `{target_col}`.\n"
        f"4. **wrong_dtypes**: Cast listed columns via "
        f"`pd.to_numeric(df[col], errors='coerce')`.\n"
        f"5. **outliers**: Clip outlier columns to the 1st–99th percentile "
        f"range. Skip `{target_col}`.\n"
        f"6. **class_imbalance**: Add a comment acknowledging it but do NOT "
        f"resample.\n\n"
        f"Output ONLY the Python code block. No explanations. "
        f"The variable `df` must remain the fixed DataFrame at the end. "
        f"`pd` and `np` are already imported."
    )

    # ── API call ──────────────────────────────────────────────────────────
    client: anthropic.Anthropic = anthropic.Anthropic(api_key=resolved_key)

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    # ── Cleanup ───────────────────────────────────────────────────────────
    raw_text: str = message.content[0].text
    cleaned: str = _strip_markdown(raw_text)

    return cleaned if cleaned else ""
