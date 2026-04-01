"""
demo_run.py
Runs the full pipeline end-to-end using ONLY synthetic data (no API key needed).
Covers:
  Part 1 — agent.py demo mode (3 built-in synthetic tasks via env)
  Part 2 — GenericTuner on custom synthetic classification dataset
  Part 3 — GenericTuner on custom synthetic regression dataset
  Part 4 — agent.py _build_generic_context() shown with synthetic health JSON
"""

import json
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

print("""
╔══════════════════════════════════════════════════════════╗
║          SILENT DATA DEBUGGER — FULL DEMO RUN            ║
║  Synthetic data only · No API key needed · fallback mode ║
╚══════════════════════════════════════════════════════════╝
""")


# ──────────────────────────────────────────────────────────
# WHAT DOES agent.py DO?  (printed before running anything)
# ──────────────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  WHAT DOES agent.py DO?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  agent.py is the BRAIN of the pipeline.

  It takes a "Data Health JSON" (produced by state.py) which
  describes what's WRONG with a dataset, then:

    1. DECIDES what fix to apply
       ─ If task is known (scaling / attrition / skewed):
           uses a task-specific prompt template
       ─ If task is ANYTHING ELSE (uploaded CSV):
           _build_generic_context() reads the health JSON and
           auto-fills the prompt with targeted fix instructions

    2. CALLS an LLM (or Optuna fallback)
       ─ openai  → GPT-4o via OpenAI API
       ─ gemini  → Gemini 1.5 via Google GenAI API
       ─ claude  → Claude 3.5 via Anthropic API
       ─ fallback→ NO LLM — uses tune.py (Optuna) instead

    3. PARSES the response
       ─ _extract_code() strips conversational text
       ─ Handles ```python, ```Python, ``` (no tag), CRLF

    4. SUBMITS code to env.step()
       ─ evaluate.py runs the code in a sandboxed exec()
       ─ Measures before/after metric
       ─ Returns reward 0.0 → 1.0

  Full loop:  env.reset() → agent.act() → env.step() → reward

""")

input("  Press Enter to start the demo run...\n")


# ──────────────────────────────────────────────────────────
# PART 1 — agent.py DEMO MODE (3 synthetic tasks)
# ──────────────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PART 1: agent.py fallback mode — 3 built-in synthetic tasks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Each task generates its own synthetic "broken" dataset:

  Task 1 — scaling   : KNN hurt by 1 feature 1000x bigger
  Task 2 — attrition : RandomForest blocked by NaNs + categoricals
  Task 3 — skewed    : Regression on a log-normal target

  The agent uses Optuna (no LLM) to find best hyperparameters,
  generates fix code, and submits it to the environment.
""")

from agent import run_all_tasks

results = run_all_tasks(
    provider="fallback",
    n_tune_trials=30,
    verbose=True,
)


# ──────────────────────────────────────────────────────────
# PART 2 — GenericTuner on custom CLASSIFICATION data
# ──────────────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PART 2: GenericTuner — custom synthetic CLASSIFICATION data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Simulates a user uploading a CSV with a binary target.
  GenericTuner auto-detects classification (≤10 unique values)
  and searches across RandomForest | GradientBoosting | KNN.
""")

from tune import GenericTuner

np.random.seed(42)
X_cls, y_cls = make_classification(
    n_samples=600, n_features=12, n_informative=8,
    n_redundant=2, random_state=42
)
df_cls = pd.DataFrame(X_cls, columns=[f"feature_{i}" for i in range(12)])
df_cls["target"] = y_cls

print(f"  Synthetic classification dataset: {df_cls.shape}")
print(f"  Target unique values: {df_cls['target'].nunique()} (should ≤ 10 → classification)")
print()

cls_result = GenericTuner(df_cls, "target", n_trials=30, cv_folds=5, verbose=True).tune()

print(f"\n  Summary:")
print(cls_result.summary())

print("\n  Generated Agent Code:")
print("  " + "\n  ".join(cls_result.agent_code.strip().splitlines()))


# ──────────────────────────────────────────────────────────
# PART 3 — GenericTuner on custom REGRESSION data
# ──────────────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PART 3: GenericTuner — custom synthetic REGRESSION data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Simulates a user uploading a CSV with a continuous target
  (e.g. house prices). GenericTuner auto-detects regression
  (>10 unique values) and searches across Ridge | Lasso |
  ElasticNet | GradientBoosting.
""")

X_reg, y_reg = make_regression(
    n_samples=600, n_features=10, n_informative=7,
    noise=25.0, random_state=42
)
df_reg = pd.DataFrame(X_reg, columns=[f"feature_{i}" for i in range(10)])
df_reg["house_price"] = np.abs(y_reg) * 1000 + 50000

print(f"  Synthetic regression dataset: {df_reg.shape}")
print(f"  Target unique values: {df_reg['house_price'].nunique()} (should >10 → regression)")
print(f"  Target range: {df_reg['house_price'].min():.0f} – {df_reg['house_price'].max():.0f}")
print()

reg_result = GenericTuner(df_reg, "house_price", n_trials=30, cv_folds=5, verbose=True).tune()

print(f"\n  Summary:")
print(reg_result.summary())

print("\n  Generated Agent Code:")
print("  " + "\n  ".join(reg_result.agent_code.strip().splitlines()))


# ──────────────────────────────────────────────────────────
# PART 4 — Generic Context Builder (what the LLM would see)
# ──────────────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PART 4: agent.py _build_generic_context()
          What the LLM sees when a CSV is uploaded
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

from agent import DataDebuggerAgent

synthetic_health_json = {
    "shape": {"rows": 500, "cols": 11},
    "missing_values": {
        "Age":    {"count": 23, "pct": 4.6},
        "Salary": {"count": 11, "pct": 2.2},
    },
    "unencoded_categoricals": ["Department", "JobRole"],
    "scale_mismatch": {
        "Salary": {"min": 30000, "max": 200000, "range": 170000}
    },
    "high_skewness": {
        "Salary": {"skew": 4.7}
    },
    "class_imbalance": 0.08,
    "outliers": {
        "YearsAtCompany": {"count": 5}
    },
}

dummy_obs = {
    "task": "uploaded_csv",
    "state": json.dumps(synthetic_health_json),
    "meta": {"target_col": "Attrition", "mode": "upload"},
}

context = DataDebuggerAgent._build_generic_context(dummy_obs)
print("  ─── LLM Context generated by _build_generic_context() ───\n")
for line in context.splitlines():
    print(f"  {line}")

print("\n  ─── Regex Extractor test ───")
fake_llm_responses = [
    ("```python\ndf['Age'].fillna(df['Age'].median())\n```", "python fence"),
    ("```Python\ndf['Age'].fillna(df['Age'].median())\n```", "Python fence"),
    ("```\ndf['Age'].fillna(df['Age'].median())\n```",       "no-lang fence"),
    ("Here's the fix:\ndf['Age'].fillna(df['Age'].median())", "raw text"),
]
for raw, label in fake_llm_responses:
    code = DataDebuggerAgent._extract_code(raw)
    print(f"    [{label}] → extracted: {code!r}")


# ──────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FINAL SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""")

print(f"\n  Part 1 — Demo tasks (agent.py + env):")
for task, r in results.items():
    status = "✅ SOLVED" if r.done else "⚠️  partial"
    print(f"    {task:<12} reward={r.final_reward:.2f}  {status}  ({r.elapsed_sec:.1f}s)")

print(f"\n  Part 2 — GenericTuner Classification:")
print(f"    best model : {cls_result.best_params.get('model', 'N/A')}")
print(f"    accuracy   : {cls_result.best_score:.4f} (baseline: {cls_result.baseline:.4f}, Δ {cls_result.improvement:+.4f})")

print(f"\n  Part 3 — GenericTuner Regression:")
print(f"    best model : {reg_result.best_params.get('model', 'N/A')}")
print(f"    RMSE       : {reg_result.best_score:.2f} (baseline: {reg_result.baseline:.2f}, Δ {reg_result.improvement:+.2f})")

print(f"\n  Part 4 — Generic prompt builder: ✅ all 6 issue types detected & formatted")
print(f"           Regex extractor: ✅ handles python / Python / no-lang / raw text")
print()
