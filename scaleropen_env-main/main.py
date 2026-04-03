"""
main.py
Single entry-point that runs the entire Silent Data Debugger pipeline.

Covers:
  1. Demo Mode  — 3 built-in synthetic tasks (scaling / attrition / skewed)
                  via agent.py + env.py + evaluate.py + tune.py
  2. Generic Mode — GenericTuner on a custom synthetic classification dataset
  3. Generic Mode — GenericTuner on a custom synthetic regression dataset
  4. Prompt Layer — shows what the LLM would see for an uploaded CSV

Run:
    python main.py
"""

import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

warnings.filterwarnings("ignore")

DIVIDER  = "=" * 65
SECTION  = "─" * 65
TICK     = "✅"
CROSS    = "❌"
PARTIAL  = "⚠️ "


def header(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def subheader(title: str):
    print(f"\n  {SECTION}")
    print(f"  {title}")
    print(f"  {SECTION}")


# ══════════════════════════════════════════════════════════════
#  SECTION 1 — DEMO MODE: 3 synthetic tasks via agent.py + env
# ══════════════════════════════════════════════════════════════

header("SECTION 1 — DEMO MODE  (agent.py → env → evaluate.py → tune.py)")
print("""
  Three built-in synthetic "broken" datasets are generated automatically.
  The agent uses Optuna (no LLM / no API key) to find best hyperparameters,
  generates fix code, and submits it to the environment.

  Task  │ Synthetic problem injected
  ──────┼──────────────────────────────────────────────────────
  scaling   │ 1 feature has 1000x larger range → KNN fails
  attrition │ NaN values + unencoded categoricals → RF blocked
  skewed    │ log-normal target → raw regression RMSE is huge
""")

from agent import run_all_tasks

t0 = time.time()
demo_results = run_all_tasks(
    provider="fallback",
    n_tune_trials=40,
    verbose=True,
)
demo_elapsed = time.time() - t0

subheader("Demo Mode Summary")
print(f"\n  {'Task':<14} {'Reward':>7}  {'Solved':>6}  {'Steps':>5}  {'Time':>7}")
print(f"  {'─'*14} {'─'*7}  {'─'*6}  {'─'*5}  {'─'*7}")
for task, r in demo_results.items():
    icon = TICK if r.done else PARTIAL
    print(f"  {task:<14} {r.final_reward:>6.2f}   {icon:>5}  {r.n_steps:>5}  {r.elapsed_sec:>6.1f}s")
print(f"\n  Total elapsed: {demo_elapsed:.1f}s")


# ══════════════════════════════════════════════════════════════
#  SECTION 2 — GENERIC TUNER: classification (any uploaded CSV)
# ══════════════════════════════════════════════════════════════

header("SECTION 2 — GENERIC TUNER: Classification on synthetic CSV data")
print("""
  Simulates a user uploading a CSV with a binary target column.
  GenericTuner auto-detects classification (≤10 unique target values)
  and searches across:  RandomForest  |  GradientBoosting  |  KNN
""")

from tune import GenericTuner

np.random.seed(42)
X_cls, y_cls = make_classification(
    n_samples=600, n_features=12, n_informative=8,
    n_redundant=2, random_state=42
)
df_cls = pd.DataFrame(X_cls, columns=[f"feature_{i}" for i in range(12)])
df_cls["Churn"] = y_cls

print(f"  Dataset  : {df_cls.shape[0]} rows × {df_cls.shape[1]} cols")
print(f"  Target   : Churn  ({df_cls['Churn'].nunique()} unique values → classification)")
print(f"  Features : {[c for c in df_cls.columns if c != 'Churn']}")
print()

cls_result = GenericTuner(df_cls, "Churn", n_trials=40, cv_folds=5, verbose=True).tune()

subheader("Classification Results")
print(cls_result.summary())
print("\n  Generated agent code:")
for line in cls_result.agent_code.strip().splitlines():
    print(f"    {line}")


# ══════════════════════════════════════════════════════════════
#  SECTION 3 — GENERIC TUNER: regression (any uploaded CSV)
# ══════════════════════════════════════════════════════════════

header("SECTION 3 — GENERIC TUNER: Regression on synthetic CSV data")
print("""
  Simulates a user uploading a CSV with a continuous numeric target.
  GenericTuner auto-detects regression (>10 unique target values)
  and searches across:  Ridge  |  Lasso  |  ElasticNet  |  GradientBoosting
""")

X_reg, y_reg = make_regression(
    n_samples=600, n_features=10, n_informative=7,
    noise=25.0, random_state=42
)
df_reg = pd.DataFrame(X_reg, columns=[f"feature_{i}" for i in range(10)])
df_reg["HousePrice"] = np.abs(y_reg) * 1000 + 50000

print(f"  Dataset  : {df_reg.shape[0]} rows × {df_reg.shape[1]} cols")
print(f"  Target   : HousePrice  ({df_reg['HousePrice'].nunique()} unique values → regression)")
print(f"  Range    : {df_reg['HousePrice'].min():,.0f} – {df_reg['HousePrice'].max():,.0f}")
print()

reg_result = GenericTuner(df_reg, "HousePrice", n_trials=40, cv_folds=5, verbose=True).tune()

subheader("Regression Results")
print(reg_result.summary())
print("\n  Generated agent code:")
for line in reg_result.agent_code.strip().splitlines():
    print(f"    {line}")


# ══════════════════════════════════════════════════════════════
#  SECTION 4 — PROMPT LAYER: what the LLM sees for any CSV
# ══════════════════════════════════════════════════════════════

header("SECTION 4 — PROMPT LAYER  (agent.py generic context builder)")
print("""
  When a user uploads any CSV, agent._build_generic_context() reads the
  Data Health JSON produced by state.py and auto-generates LLM instructions.
  This is what gets sent to OpenAI / Gemini / Claude in production.
""")

from agent import DataDebuggerAgent

health_json = {
    "shape": {"rows": 500, "cols": 11},
    "missing_values": {
        "Age":    {"count": 23, "pct": 4.6},
        "Salary": {"count": 11, "pct": 2.2},
    },
    "unencoded_categoricals": ["Department", "JobRole"],
    "scale_mismatch": {
        "Salary": {"min": 30000, "max": 200000, "range": 170000}
    },
    "high_skewness": {"Salary": {"skew": 4.7}},
    "class_imbalance": 0.08,
    "outliers": {"YearsAtCompany": {"count": 5}},
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
    ("```python\ndf['Age'].fillna(df['Age'].median())\n```",     "```python ... ```"),
    ("```Python\ndf['Age'].fillna(df['Age'].median())\n```",     "```Python ... ```"),
    ("```\ndf['Age'].fillna(df['Age'].median())\n```",           "``` ... ```"),
    ("Here is the fix:\ndf['Age'].fillna(df['Age'].median())",   "raw text (no fences)"),
]
for raw, label in variants:
    extracted = DataDebuggerAgent._extract_code(raw)
    print(f"\n  Input  [{label}]")
    print(f"  Output → {extracted!r}")


# ══════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════

header("FINAL SUMMARY — All Sections Complete")

print(f"""
  Section 1 — Demo Mode (3 synthetic env tasks)
  {'─'*52}""")
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
  {TICK}  RMSE        = {reg_result.best_score:,.2f}  (baseline {reg_result.baseline:,.2f},  saved {reg_result.improvement:,.2f})

  Section 4 — Prompt Layer
  {'─'*52}
  {TICK}  Generic context builder: all issue types detected & formatted
  {TICK}  Regex extractor: handles python / Python / no-lang / raw text

  {'─'*52}
  Pipeline: state.py → agent.py → evaluate.py → tune.py → result
""")
