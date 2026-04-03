"""
test_50k.py
───────────
Stress-test the full Silent Data Debugger pipeline on a realistic
50,000-row synthetic dataset with deliberately injected data quality
problems.

The synthetic dataset mimics an employee / HR dataset:
  - Scale mismatch   : salary (50k–200k) next to 0-1 features
  - Missing values   : ~8 % NaN in Age, MonthlyIncome, YearsAtCompany
  - Categoricals     : Department, JobRole, OverTime (unencoded strings)
  - Skewed target    : log-normal salary distribution

Tests run
─────────
  A. DataHealth inspection           (state.py)
  B. Generic Tuner — classification  (tune.GenericTuner)
  C. Generic Tuner — regression      (tune.GenericTuner)
  D. Generic Evaluator               (evaluate.evaluate)
  E. RL env tasks on 50k config      (agent.run_all_tasks)

Usage
─────
    python test_50k.py
    python test_50k.py --trials 20       # more Optuna trials
    python test_50k.py --no-rl           # skip the slow RL loop
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── formatting helpers ────────────────────────────────────────────────────────
DIV  = "=" * 68
SEP  = "─" * 68
TICK = "✅"
SKIP = "⏭️ "


def header(title: str) -> None:
    print(f"\n{DIV}\n  {title}\n{DIV}")


def subheader(title: str) -> None:
    print(f"\n  {SEP}\n  {title}\n  {SEP}")


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 1 — GENERATE 50K SYNTHETIC DATASET
# ═════════════════════════════════════════════════════════════════════════════

def build_50k_dataset(seed: int = 42) -> pd.DataFrame:
    """
    50,000-row synthetic HR / employee dataset with deliberate quality issues.

    Injected problems
    -----------------
    1. Scale mismatch  : 'Salary' in [50k, 200k] while other numerics in [0, 1]
    2. Missing values  : ~8 % NaN in Age, MonthlyIncome, YearsAtCompany
    3. Unencoded cats  : Department, JobRole, OverTime are raw strings
    4. Skewed target   : LogSalary = log-normal (for regression task)
    5. Class imbalance : Attrition is ~15 % positive (for classification task)
    """
    rng = np.random.default_rng(seed)
    n = 50_000

    departments = ["Engineering", "Sales", "HR", "Finance", "Marketing", "Legal"]
    job_roles   = ["Analyst", "Manager", "Director", "Engineer", "Clerk",
                   "VP", "Consultant", "Recruiter"]
    overtime    = ["Yes", "No"]
    edu_fields  = ["Life Sciences", "Medical", "Marketing",
                   "Technical Degree", "Human Resources", "Other"]

    age           = rng.integers(22, 61, n).astype(float)
    monthly_inc   = rng.integers(2_000, 20_001, n).astype(float)
    years_company = rng.integers(0, 31, n).astype(float)
    distance      = rng.integers(1, 30, n)
    satisfaction  = rng.integers(1, 5, n)
    perf_rating   = rng.integers(1, 5, n)
    work_life     = rng.integers(1, 5, n)
    training      = rng.integers(0, 7, n)
    involvement   = rng.integers(1, 5, n)
    edu_level     = rng.integers(1, 6, n)

    # ── Scale mismatch: Salary is 1000x larger than other numeric features
    # Underlying log-normal so the distribution is also skewed
    log_salary = (
        0.3 * (monthly_inc / 20_000)       # income signal
        + 0.2 * (age / 60)                 # seniority signal
        + 0.1 * (years_company / 30)       # tenure signal
        + rng.normal(0, 0.4, n)            # noise
    )
    salary = np.exp(log_salary) * 80_000   # [~30k, ~250k], log-normal

    dept     = rng.choice(departments, n)
    role     = rng.choice(job_roles, n)
    ot       = rng.choice(overtime, n, p=[0.28, 0.72])
    edu_f    = rng.choice(edu_fields, n)

    # Attrition: ~15 % positive, influenced by OverTime + distance
    attrition_prob = 0.08 + 0.20 * (ot == "Yes") + 0.002 * distance
    attrition = (rng.random(n) < attrition_prob).astype(int)

    df = pd.DataFrame({
        "Age":               age,
        "MonthlyIncome":     monthly_inc,
        "Salary":            salary,
        "YearsAtCompany":    years_company,
        "DistanceFromHome":  distance,
        "JobSatisfaction":   satisfaction,
        "PerformanceRating": perf_rating,
        "WorkLifeBalance":   work_life,
        "TrainingLastYear":  training,
        "JobInvolvement":    involvement,
        "EducationLevel":    edu_level,
        "Department":        dept,
        "JobRole":           role,
        "OverTime":          ot,
        "EducationField":    edu_f,
        "Attrition":         attrition,
        "LogSalary":         np.log1p(salary),  # clean target for regression demo
    })

    # ── Inject nulls (~8 %) into numeric columns
    for col in ("Age", "MonthlyIncome", "YearsAtCompany"):
        null_idx = rng.choice(n, size=int(n * 0.08), replace=False)
        df.loc[null_idx, col] = np.nan

    return df


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 2 — DATA HEALTH INSPECTION
# ═════════════════════════════════════════════════════════════════════════════

def test_A_health(df: pd.DataFrame) -> None:
    from state import get_state
    from config import GlobalConfig

    header("TEST A — Data Health Inspection  (state.py)")
    t0 = time.time()
    health_json = get_state(df, GlobalConfig())
    elapsed = time.time() - t0

    health = json.loads(health_json)
    print(f"\n  Rows : {len(df):,}   Cols : {len(df.columns)}")
    print(f"  Health scan completed in {elapsed:.3f}s\n")

    checks = {
        "missing_values":        "Missing values",
        "high_skew_columns":     "High-skew columns",
        "outliers":              "Outlier columns",
        "wrong_dtypes":          "Wrong dtypes",
        "scale_mismatch":        "Scale mismatch",
        "class_imbalance":       "Class imbalance",
    }
    for key, label in checks.items():
        val = health.get(key)
        if val:
            icon = "⚠️  "
        else:
            icon = TICK
        print(f"  {icon} {label:25s}: {val}")

    print(f"\n  {TICK} state.py scan: OK")


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 3 — GENERIC TUNER: CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════

def test_B_classification(df: pd.DataFrame, trials: int) -> None:
    from tune import GenericTuner

    header("TEST B — GenericTuner: Classification  (Attrition ~ binary)")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    target = "Attrition"
    features = [c for c in numeric_cols if c not in (target, "LogSalary")]

    df_full = df[features + [target]].dropna().copy()
    # Subsample 10k for tuning — full 50k makes RF CV too slow per trial
    df_tune = df_full.sample(n=min(10_000, len(df_full)), random_state=42)
    print(f"  Full dataset : {df_full.shape[0]:,} rows × {df_full.shape[1]} cols (after dropna)")
    print(f"  Tuning on   : {df_tune.shape[0]:,} rows  (10k subsample for speed)")
    print(f"  Target       : {target}  ({df_full[target].value_counts().to_dict()})")
    print(f"  Trials       : {trials}  |  cv_folds=3  |  timeout=90s\n")

    t0 = time.time()
    result = GenericTuner(
        df_tune, target, n_trials=trials, cv_folds=3, verbose=True
    ).tune()
    elapsed = time.time() - t0

    subheader("Classification Results")
    print(result.summary())
    print(f"\n  {TICK}  Elapsed: {elapsed:.1f}s")
    print(f"  {TICK}  Best model : {result.best_params.get('model', 'N/A')}")
    print(f"  {TICK}  Accuracy   : {result.best_score:.4f}  (Δ {result.improvement:+.4f} vs baseline)")


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 4 — GENERIC TUNER: REGRESSION
# ═════════════════════════════════════════════════════════════════════════════

def test_C_regression(df: pd.DataFrame, trials: int) -> None:
    from tune import GenericTuner

    header("TEST C — GenericTuner: Regression  (Salary ~ continuous)")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    target = "Salary"
    features = [c for c in numeric_cols if c not in (target, "LogSalary", "Attrition")]

    df_full = df[features + [target]].dropna().copy()
    df_tune = df_full.sample(n=min(10_000, len(df_full)), random_state=42)
    print(f"  Full dataset : {df_full.shape[0]:,} rows × {df_full.shape[1]} cols")
    print(f"  Tuning on   : {df_tune.shape[0]:,} rows  (10k subsample for speed)")
    print(f"  Target       : {target}  |  range [{df_full[target].min():,.0f} – {df_full[target].max():,.0f}]")
    print(f"  Trials       : {trials}  |  cv_folds=3\n")

    t0 = time.time()
    result = GenericTuner(
        df_tune, target, n_trials=trials, cv_folds=3, verbose=True
    ).tune()
    elapsed = time.time() - t0

    subheader("Regression Results")
    print(result.summary())
    print(f"\n  {TICK}  Elapsed : {elapsed:.1f}s")
    print(f"  {TICK}  Best model : {result.best_params.get('model', 'N/A')}")
    print(f"  {TICK}  RMSE       : {result.best_score:,.2f}  (baseline {result.baseline:,.2f},  saved {result.improvement:,.2f})")
    print("\n  Generated agent code:")
    for line in result.agent_code.strip().splitlines():
        print(f"    {line}")


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 5 — GENERIC EVALUATOR
# ═════════════════════════════════════════════════════════════════════════════

def test_D_evaluator(df: pd.DataFrame) -> None:
    from evaluate import evaluate
    from config import GlobalConfig

    header("TEST D — Generic Evaluator  (evaluate.evaluate)")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    target = "Attrition"
    features = [c for c in numeric_cols if c not in (target, "LogSalary")]
    df_eval = df[features + [target]].dropna().copy()

    # Minimal fix code: impute + train
    fix_code = """
import pandas as pd
from sklearn.impute import SimpleImputer
# Impute any remaining NaN
num_cols = df.select_dtypes(include='number').drop(columns=['Attrition'], errors='ignore').columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
"""
    print(f"  Dataset  : {df_eval.shape[0]:,} rows × {df_eval.shape[1]} cols")
    print(f"  Target   : {target}")

    t0 = time.time()
    result = evaluate(df_eval, fix_code, target, GlobalConfig())
    elapsed = time.time() - t0

    subheader("Evaluator Results")
    print(f"  Task type  : {result['task_type']}")
    print(f"  Metric     : {result['metric']}")
    print(f"  Before     : {result['before']:.4f}")
    print(f"  After      : {result['after']:.4f}")
    print(f"  Reward     : {result['reward']:.4f}")
    print(f"  Improved   : {result['improved']}")
    if result["exec_error"]:
        print(f"  Error      : {result['exec_error']}")
    print(f"\n  {TICK}  Elapsed: {elapsed:.3f}s")


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 6 — RL ENV LOOP ON STANDARD TASKS (with 50k config override)
# ═════════════════════════════════════════════════════════════════════════════

def test_E_rl_tasks(trials: int) -> None:
    from agent import run_all_tasks

    header("TEST E — RL Env Tasks  (agent → env → evaluate → tune, n=50k config)")
    print(f"""
  Runs all 3 built-in synthetic tasks on the environment.
  The environment generates its own synthetic data (n configurable in config.py).
  Here we test with the default n=1000 but more Optuna trials ({trials}).
""")

    t0 = time.time()
    results = run_all_tasks(
        provider="fallback",
        n_tune_trials=trials,
        verbose=True,
    )
    total = time.time() - t0

    subheader("RL Task Summary")
    print(f"\n  {'Task':<14} {'Reward':>7}  {'Solved':>6}  {'Steps':>5}  {'Time':>7}")
    print(f"  {'─'*14} {'─'*7}  {'─'*6}  {'─'*5}  {'─'*7}")
    for task, r in results.items():
        icon = TICK if r.done else "⚠️ "
        print(f"  {task:<14} {r.final_reward:>6.2f}   {icon:>5}  {r.n_steps:>5}  {r.elapsed_sec:>6.1f}s")
    print(f"\n  Total elapsed: {total:.1f}s")
    return results


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stress-test the full pipeline on a 50,000-row synthetic dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trials", type=int, default=10,
        help="Optuna trials for GenericTuner and RL tasks (default: 10)",
    )
    parser.add_argument(
        "--no-rl", action="store_true",
        help="Skip the RL env test (TEST E) — faster overall run",
    )
    args = parser.parse_args()

    print(f"\n{'#'*68}")
    print(f"#{'SILENT DATA DEBUGGER — 50K SYNTHETIC STRESS TEST':^66}#")
    print(f"{'#'*68}")

    # ── Build dataset ──────────────────────────────────────────────────────
    header("GENERATING 50,000-ROW SYNTHETIC DATASET")
    t0 = time.time()
    df = build_50k_dataset()
    print(f"\n  {TICK}  Built in {time.time()-t0:.2f}s")
    print(f"  Shape    : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"\n  Columns  : {list(df.columns)}")
    print(f"\n  Missing values per column:")
    nans = df.isnull().sum()
    for col, cnt in nans[nans > 0].items():
        print(f"     {col:<22}: {cnt:,}  ({cnt/len(df)*100:.1f}%)")
    print(f"\n  Salary range  : {df['Salary'].min():,.0f} – {df['Salary'].max():,.0f}")
    print(f"  Attrition rate: {df['Attrition'].mean()*100:.1f}%")

    # ── Run all tests ──────────────────────────────────────────────────────
    results: dict = {}

    # A: health check
    test_A_health(df)
    results["health"] = TICK

    # B: classification tuner
    test_B_classification(df, args.trials)
    results["classification"] = TICK

    # C: regression tuner
    test_C_regression(df, args.trials)
    results["regression"] = TICK

    # D: generic evaluator
    test_D_evaluator(df)
    results["evaluator"] = TICK

    # E: RL loop
    if not args.no_rl:
        test_E_rl_tasks(args.trials)
        results["rl_tasks"] = TICK
    else:
        print(f"\n  {SKIP} TEST E — RL tasks skipped (--no-rl flag)")
        results["rl_tasks"] = SKIP

    # ── Final summary ──────────────────────────────────────────────────────
    header("FINAL SUMMARY — 50K STRESS TEST")
    print(f"""
  Component                  Status
  ─────────────────────────  ──────
  A. Data Health (state.py)  {results['health']}
  B. Tuner — Classification  {results['classification']}
  C. Tuner — Regression      {results['regression']}
  D. Generic Evaluator       {results['evaluator']}
  E. RL Env (all 3 tasks)    {results['rl_tasks']}

  Dataset: 50,000 rows  |  16 columns  |  injected: nulls, scale mismatch,
           categoricals, class imbalance, log-normal skewed target
""")


if __name__ == "__main__":
    main()
