"""
synthetic_data.py
─────────────────
Generates a rich, realistic synthetic Employee Productivity & Salary dataset
with intentional impurities designed to stress-test a data-aware AutoML
pipeline's cleaning, feature-engineering, and correlation-analysis stages.

Functions
---------
generate_clean_data  – Produces the base dataset with logical relationships.
inject_impurities    – Applies nulls and dtype corruption to a clean frame.
get_synthetic_dataset – Orchestrator that returns the final (df, target) pair.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_DEPARTMENTS: list[str] = ["IT", "HR", "Finance", "Marketing", "Operations"]
_EDUCATION_LEVELS: list[str] = ["High School", "Bachelor", "Master", "PhD"]
_JOB_ROLES: list[str] = ["Junior", "Mid", "Senior", "Lead", "Manager"]
_GENDERS: list[str] = ["Male", "Female", "Non-binary"]
_EMPLOYMENT_TYPES: list[str] = ["Full-time", "Part-time", "Contract"]
_YES_NO: list[str] = ["Yes", "No"]
_LOCATIONS: list[str] = [
    "Aetherton",
    "Bluridge",
    "Crescentville",
    "Dunmore",
    "Eldenport",
]

# Education and job-role ordinal maps used for salary generation
_EDUCATION_WEIGHT: dict[str, float] = {
    "High School": 0.0,
    "Bachelor": 0.3,
    "Master": 0.6,
    "PhD": 0.9,
}

_JOB_ROLE_WEIGHT: dict[str, float] = {
    "Junior": 0.0,
    "Mid": 0.25,
    "Senior": 0.55,
    "Lead": 0.80,
    "Manager": 1.0,
}

# Null-injection specification: column → fraction of rows to nullify
_NULL_SPEC: dict[str, float] = {
    "training_hours": 0.08,
    "satisfaction_score": 0.06,
    "last_promotion_years": 0.10,
    "distance_from_office": 0.05,
}


# ──────────────────────────────────────────────────────────────────────────────
# 1. Clean data generation
# ──────────────────────────────────────────────────────────────────────────────


def generate_clean_data(
    n_samples: int = 1500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a *clean* DataFrame with 22 features + 1 target (``salary``).

    All logical relationships (e.g. ``years_experience`` correlating with
    ``age``) are baked in here.  No impurities are introduced at this stage.

    Parameters
    ----------
    n_samples : int, default 1500
        Number of rows to generate.
    random_state : int, default 42
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        A clean dataset with 23 columns (22 features + ``salary``).
    """
    rng: np.random.Generator = np.random.default_rng(random_state)

    # ── Numerical features ────────────────────────────────────────────────
    age: np.ndarray = rng.integers(25, 66, size=n_samples)  # 25–65

    # years_experience positively correlated with age
    max_exp: np.ndarray = np.clip(age - 22, 0, 40)
    years_experience: np.ndarray = np.clip(
        (max_exp * rng.uniform(0.3, 1.0, size=n_samples)).astype(int),
        0,
        40,
    )

    hours_per_week: np.ndarray = rng.integers(20, 81, size=n_samples)
    num_projects: np.ndarray = rng.integers(1, 21, size=n_samples)
    team_size: np.ndarray = rng.integers(2, 51, size=n_samples)
    performance_score: np.ndarray = np.round(
        rng.uniform(1.0, 5.0, size=n_samples), 2
    )
    training_hours: np.ndarray = rng.integers(0, 201, size=n_samples)
    distance_from_office: np.ndarray = rng.integers(1, 101, size=n_samples)
    num_reports: np.ndarray = rng.integers(0, 16, size=n_samples)
    monthly_expenses: np.ndarray = rng.integers(500, 5001, size=n_samples)
    satisfaction_score: np.ndarray = rng.integers(1, 11, size=n_samples)
    last_promotion_years: np.ndarray = rng.integers(0, 16, size=n_samples)

    # ── Categorical features ──────────────────────────────────────────────
    department: np.ndarray = rng.choice(_DEPARTMENTS, size=n_samples)
    education: np.ndarray = rng.choice(_EDUCATION_LEVELS, size=n_samples)
    job_role: np.ndarray = rng.choice(_JOB_ROLES, size=n_samples)
    gender: np.ndarray = rng.choice(_GENDERS, size=n_samples)
    employment_type: np.ndarray = rng.choice(_EMPLOYMENT_TYPES, size=n_samples)
    remote_work: np.ndarray = rng.choice(_YES_NO, size=n_samples)
    overtime: np.ndarray = rng.choice(_YES_NO, size=n_samples)
    location: np.ndarray = rng.choice(_LOCATIONS, size=n_samples)

    # ── Redundant / weak features ─────────────────────────────────────────
    # age_copy: almost identical to age (r > 0.95)
    age_copy: np.ndarray = age + rng.normal(0, 0.5, size=n_samples)

    # random_noise: pure noise, negligible correlation with target
    random_noise: np.ndarray = rng.uniform(0, 1, size=n_samples)

    # ── Target: salary (log-normal, right-skewed > 3.0) ──────────────────
    # Build a latent score from the key drivers
    edu_weights: np.ndarray = np.array(
        [_EDUCATION_WEIGHT[e] for e in education]
    )
    role_weights: np.ndarray = np.array(
        [_JOB_ROLE_WEIGHT[r] for r in job_role]
    )

    latent: np.ndarray = (
        0.25 * (years_experience / 40.0)
        + 0.30 * role_weights
        + 0.25 * edu_weights
        + 0.10 * ((performance_score - 1.0) / 4.0)
        + 0.10 * rng.uniform(0, 1, size=n_samples)  # stochastic residual
    )  # range ≈ [0, 1]

    # Map to log-normal with heavy right skew
    mu: float = 10.3          # centres salary around ~30-60k
    sigma: float = 0.85       # drives the skewness
    log_salary: np.ndarray = mu + sigma * (latent + rng.normal(0, 0.3, size=n_samples))
    salary: np.ndarray = np.exp(log_salary)

    # Ensure skewness > 3.0 by amplifying the upper tail
    # The log-normal with σ ≈ 0.85 gives skew ≈ 3.05 analytically;
    # we add a small multiplicative boost to the top 5 % to guarantee > 3.0
    # even with finite-sample variance.
    p95: float = float(np.percentile(salary, 95))
    tail_mask: np.ndarray = salary > p95
    salary[tail_mask] *= rng.uniform(1.3, 2.5, size=int(tail_mask.sum()))

    salary = np.round(salary, 2)

    # ── Assemble DataFrame ────────────────────────────────────────────────
    df: pd.DataFrame = pd.DataFrame(
        {
            # Numerical
            "age": age,
            "years_experience": years_experience,
            "hours_per_week": hours_per_week,
            "num_projects": num_projects,
            "team_size": team_size,
            "performance_score": performance_score,
            "training_hours": training_hours,
            "distance_from_office": distance_from_office,
            "num_reports": num_reports,
            "monthly_expenses": monthly_expenses,
            "satisfaction_score": satisfaction_score,
            "last_promotion_years": last_promotion_years,
            # Categorical
            "department": department,
            "education": education,
            "job_role": job_role,
            "gender": gender,
            "employment_type": employment_type,
            "remote_work": remote_work,
            "overtime": overtime,
            "location": location,
            # Redundant / weak
            "age_copy": age_copy,
            "random_noise": random_noise,
            # Target
            "salary": salary,
        }
    )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. Impurity injection
# ──────────────────────────────────────────────────────────────────────────────


def inject_impurities(
    df: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """Inject realistic data-quality issues into an already-clean DataFrame.

    Mutations applied (on a **copy** – the original is never altered):

    * **Null values** in ``training_hours`` (8 %), ``satisfaction_score``
      (6 %), ``last_promotion_years`` (10 %), ``distance_from_office`` (5 %).
    * **Wrong dtype** – ``performance_score`` is cast to ``str`` (e.g.
      ``"3.5"``), simulating a common ingest error.

    Parameters
    ----------
    df : pd.DataFrame
        A clean dataset (typically from :func:`generate_clean_data`).
    random_state : int, default 42
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        The corrupted dataset.
    """
    rng: np.random.Generator = np.random.default_rng(random_state)
    dirty: pd.DataFrame = df.copy()

    n: int = len(dirty)

    # ── Inject NaN values ─────────────────────────────────────────────────
    for col, frac in _NULL_SPEC.items():
        null_count: int = int(n * frac)
        null_idx: np.ndarray = rng.choice(n, size=null_count, replace=False)
        dirty.loc[dirty.index[null_idx], col] = np.nan

    # ── Wrong dtype: performance_score → str ──────────────────────────────
    dirty["performance_score"] = dirty["performance_score"].astype(str)

    return dirty


# ──────────────────────────────────────────────────────────────────────────────
# 3. Orchestrator
# ──────────────────────────────────────────────────────────────────────────────


def get_synthetic_dataset(
    n_samples: int = 1500,
    random_state: int = 42,
) -> tuple[pd.DataFrame, str]:
    """Generate the full synthetic dataset ready for pipeline consumption.

    This is the single entry-point intended for downstream code.  It calls
    :func:`generate_clean_data` followed by :func:`inject_impurities` and
    returns the resulting DataFrame along with the name of the target column.

    Parameters
    ----------
    n_samples : int, default 1500
        Number of rows.
    random_state : int, default 42
        Seed for reproducibility (forwarded to both sub-functions).

    Returns
    -------
    tuple[pd.DataFrame, str]
        ``(dataframe, "salary")``
    """
    clean_df: pd.DataFrame = generate_clean_data(
        n_samples=n_samples,
        random_state=random_state,
    )
    dirty_df: pd.DataFrame = inject_impurities(
        clean_df,
        random_state=random_state,
    )
    return dirty_df, "salary"


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check when run as a script
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df, target = get_synthetic_dataset()

    print("═" * 60)
    print("  Synthetic Dataset – Quick Diagnostics")
    print("═" * 60)
    print(f"  Shape           : {df.shape}")
    print(f"  Target column   : {target}")
    print(f"  Columns ({len(df.columns)})    : {list(df.columns)}")
    print()

    # Null summary
    null_counts: pd.Series = df.isnull().sum()
    nulls: pd.Series = null_counts[null_counts > 0]
    print("  Null counts:")
    for col_name, cnt in nulls.items():
        pct: float = cnt / len(df) * 100
        print(f"    {col_name:.<30s} {cnt:>5d}  ({pct:.1f}%)")
    print()

    # Dtype check
    print(f"  performance_score dtype : {df['performance_score'].dtype}")
    print()

    # Salary skewness
    salary_numeric: pd.Series = pd.to_numeric(df["salary"], errors="coerce")
    skew_val: float = float(salary_numeric.skew())
    print(f"  Salary skewness         : {skew_val:.2f}  (target > 3.0)")

    # age vs age_copy correlation
    age_corr: float = float(
        df["age"].astype(float).corr(df["age_copy"].astype(float))
    )
    print(f"  age ↔ age_copy corr     : {age_corr:.4f}  (target > 0.95)")

    # random_noise vs salary correlation
    noise_corr: float = float(
        df["random_noise"].astype(float).corr(salary_numeric)
    )
    print(f"  random_noise ↔ salary   : {abs(noise_corr):.4f}  (target < 0.05)")
    print("═" * 60)
