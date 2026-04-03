"""
main.py
───────
Single entry-point for the Silent Data Debugger pipeline.

Sections
────────
1. Demo Mode   — 3 synthetic broken datasets, Optuna fallback (no API key needed)
2. Generic Tuner — classification on synthetic CSV data
3. Generic Tuner — regression on synthetic CSV data
4. Prompt Layer  — shows what the LLM context looks like for any uploaded CSV

Usage
-----
    python main.py                   # all sections, 10 Optuna trials
    python main.py --trials 40       # more thorough tuning
    python main.py --section 1       # only run Demo Mode
    python main.py --provider claude --api-key sk-... --trials 5
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

warnings.filterwarnings("ignore")

# ── Console formatting helpers ───────────────────────────────────────────────
DIVIDER = "=" * 65
SECTION = "─" * 65
TICK    = "✅"
CROSS   = "❌"
PARTIAL = "⚠️ "


def header(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def subheader(title: str) -> None:
    print(f"\n  {SECTION}")
    print(f"  {title}")
    print(f"  {SECTION}")


# ════════════════════════════════════════════════════════════════════════════
#  SECTION RUNNERS
# ════════════════════════════════════════════════════════════════════════════

def run_section1(provider: str, api_key: str | None, trials: int) -> dict:
    """Demo Mode: 3 synthetic broken datasets via agent + env + evaluate + tune."""
    from agent import run_all_tasks

    header("SECTION 1 — DEMO MODE  (env → agent → evaluate → tune)")
    print(f"""
  Three synthetic \"broken\" datasets are generated automatically.
  Provider: {provider}  |  Optuna trials: {trials}

  Task       │ Synthetic problem injected
  ───────────┼────────────────────────────────────────────────────
  scaling    │ 1 feature has {'>'}1000x larger range → KNN accuracy collapses
  attrition  │ NaN values + unencoded categoricals → RF blocked
  skewed     │ log-normal target → raw RMSE is huge
""")

    t0 = time.time()
    results = run_all_tasks(
        provider=provider,
        api_key=api_key,
        n_tune_trials=trials,
        verbose=True,
    )
    elapsed = time.time() - t0

    subheader("Demo Mode Summary")
    print(f"\n  {'Task':<14} {'Reward':>7}  {'Solved':>6}  {'Steps':>5}  {'Time':>7}")
    print(f"  {'─'*14} {'─'*7}  {'─'*6}  {'─'*5}  {'─'*7}")
    for task, r in results.items():
        icon = TICK if r.done else PARTIAL
        print(f"  {task:<14} {r.final_reward:>6.2f}   {icon:>5}  {r.n_steps:>5}  {r.elapsed_sec:>6.1f}s")
    print(f"\n  Total elapsed: {elapsed:.1f}s")
    return results


def run_section2(trials: int) -> object:
    """Generic Tuner: classification on a synthetic CSV-like dataset."""
    from tune import GenericTuner

    header("SECTION 2 — GENERIC TUNER: Classification on synthetic CSV data")
    print(f"""
  Simulates a user uploading a CSV with a binary target column.
  GenericTuner auto-detects classification (≤10 unique target values)
  and searches across:  RandomForest  |  GradientBoosting  |  KNN
  Trials: {trials}
""")

    X_cls, y_cls = make_classification(
        n_samples=600, n_features=12, n_informative=8,
        n_redundant=2, random_state=42
    )
    df_cls = pd.DataFrame(X_cls, columns=[f"feature_{i}" for i in range(12)])
    df_cls["Churn"] = y_cls

    print(f"  Dataset  : {df_cls.shape[0]} rows × {df_cls.shape[1]} cols")
    print(f"  Target   : Churn  ({df_cls['Churn'].nunique()} unique values → classification)")
    print(f"  Features : {[c for c in df_cls.columns if c != 'Churn']}\n")

    result = GenericTuner(df_cls, "Churn", n_trials=trials, cv_folds=3, verbose=True).tune()

    subheader("Classification Results")
    print(result.summary())
    print("\n  Generated agent code:")
    for line in result.agent_code.strip().splitlines():
        print(f"    {line}")
    return result


def run_section3(trials: int) -> object:
    """Generic Tuner: regression on a synthetic CSV-like dataset."""
    from tune import GenericTuner

    header("SECTION 3 — GENERIC TUNER: Regression on synthetic CSV data")
    print(f"""
  Simulates a user uploading a CSV with a continuous numeric target.
  GenericTuner auto-detects regression (>10 unique target values)
  and searches across:  Ridge  |  Lasso  |  ElasticNet  |  GradientBoosting
  Trials: {trials}
""")

    X_reg, y_reg = make_regression(
        n_samples=600, n_features=10, n_informative=7,
        noise=25.0, random_state=42
    )
    df_reg = pd.DataFrame(X_reg, columns=[f"feature_{i}" for i in range(10)])
    df_reg["HousePrice"] = np.abs(y_reg) * 1000 + 50_000

    print(f"  Dataset  : {df_reg.shape[0]} rows × {df_reg.shape[1]} cols")
    print(f"  Target   : HousePrice  ({df_reg['HousePrice'].nunique()} unique → regression)")
    print(f"  Range    : {df_reg['HousePrice'].min():,.0f} – {df_reg['HousePrice'].max():,.0f}\n")

    result = GenericTuner(df_reg, "HousePrice", n_trials=trials, cv_folds=3, verbose=True).tune()

    subheader("Regression Results")
    print(result.summary())
    print("\n  Generated agent code:")
    for line in result.agent_code.strip().splitlines():
        print(f"    {line}")
    return result


def run_section4() -> None:
    """Prompt Layer: show what the LLM context looks like for any uploaded CSV."""
    from agent import DataDebuggerAgent

    header("SECTION 4 — PROMPT LAYER  (what the LLM sees for any uploaded CSV)")
    print("""
  When a user uploads any CSV, agent._build_generic_context() reads the
  Data Health JSON from state.py and auto-generates LLM instructions.
""")

    health_json = {
        "shape": {"rows": 500, "cols": 11},
        "missing_values":        {"Age": {"count": 23}, "Salary": {"count": 11}},
        "unencoded_categoricals": ["Department", "JobRole"],
        "scale_mismatch":        {"Salary": {"min": 30000, "max": 200000, "range": 170000}},
        "high_skewness":         {"Salary": {"skew": 4.7}},
        "class_imbalance":       0.08,
        "outliers":              {"YearsAtCompany": {"count": 5}},
    }
    obs = {
        "task":  "uploaded_csv",
        "state": json.dumps(health_json),
        "meta":  {"target_col": "Attrition", "mode": "upload"},
    }

    context = DataDebuggerAgent._build_generic_context(obs)
    subheader("Context sent to LLM")
    for line in context.splitlines():
        print(f"  {line}")

    subheader("Regex parser — extract code from LLM response")
    variants = [
        ("```python\ndf['Age'].fillna(df['Age'].median())\n```",   "```python ... ```"),
        ("```Python\ndf['Age'].fillna(df['Age'].median())\n```",   "```Python ... ```"),
        ("```\ndf['Age'].fillna(df['Age'].median())\n```",         "``` ... ```"),
        ("Here is the fix:\ndf['Age'].fillna(df['Age'].median())", "raw text (no fences)"),
    ]
    for raw, label in variants:
        extracted = DataDebuggerAgent._extract_code(raw)
        print(f"\n  Input  [{label}]")
        print(f"  Output → {extracted!r}")


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Silent Data Debugger — full pipeline demo\n\n"
            "Runs 4 sections:\n"
            "  1. RL demo tasks (scaling / attrition / skewed) via Optuna fallback\n"
            "  2. GenericTuner on synthetic classification CSV\n"
            "  3. GenericTuner on synthetic regression CSV\n"
            "  4. Prompt layer — shows the LLM context for any uploaded CSV\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--section", type=int, choices=[1, 2, 3, 4],
        default=None, help="Run only this section (default: all 4)",
    )
    parser.add_argument(
        "--trials", type=int, default=10,
        help="Number of Optuna trials for tuning (default: 10 for speed; use 40 for production)",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini", "claude", "fallback"],
        default="fallback",
        help="LLM provider for Section 1 (default: fallback / Optuna only)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key for the chosen provider (falls back to env vars)",
    )
    args = parser.parse_args()

    run_all = args.section is None

    # ── Section 1: Demo RL tasks ────────────────────────────────────────────
    demo_results = None
    if run_all or args.section == 1:
        demo_results = run_section1(args.provider, args.api_key, args.trials)

    # ── Section 2: Generic Tuner (classification) ───────────────────────────
    cls_result = None
    if run_all or args.section == 2:
        cls_result = run_section2(args.trials)

    # ── Section 3: Generic Tuner (regression) ──────────────────────────────
    reg_result = None
    if run_all or args.section == 3:
        reg_result = run_section3(args.trials)

    # ── Section 4: Prompt layer ─────────────────────────────────────────────
    if run_all or args.section == 4:
        run_section4()

    # ── Final summary (only when all sections ran) ──────────────────────────
    if run_all and demo_results and cls_result and reg_result:
        header("FINAL SUMMARY — All Sections Complete")
        print(f"\n  Section 1 — Demo RL Tasks\n  {'─'*52}")
        for task, r in demo_results.items():
            icon = TICK if r.done else PARTIAL
            print(f"  {icon}  {task:<12}  reward={r.final_reward:.2f}  ({r.elapsed_sec:.1f}s)")

        print(f"""
  Section 2 — GenericTuner Classification
  {'─'*52}
  {TICK}  best model  = {cls_result.best_params.get('model', 'N/A')}
  {TICK}  accuracy    = {cls_result.best_score:.4f}  (baseline {cls_result.baseline:.4f},  Δ {cls_result.improvement:+.4f})

  Section 3 — GenericTuner Regression
  {'─'*52}
  {TICK}  best model  = {reg_result.best_params.get('model', 'N/A')}
  {TICK}  RMSE        = {reg_result.best_score:,.2f}  (baseline {reg_result.baseline:,.2f},  improvement {reg_result.improvement:,.2f})

  Section 4 — Prompt Layer
  {'─'*52}
  {TICK}  Context builder: all issue types detected & formatted
  {TICK}  Regex extractor: handles python / Python / no-lang / raw text

  {'─'*52}
  Pipeline: state.py → agent.py → evaluate.py → tune.py → result
""")


if __name__ == "__main__":
    main()
