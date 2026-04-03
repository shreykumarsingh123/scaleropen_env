"""
tune.py — Hyperparameter Tuning Module for the Silent Data Debugger.

Uses Optuna to find optimal model hyperparameters for each task AFTER
applying the canonical data fix. Integrates with the existing
SilentDebuggerEnv loop and can generate ready-to-use agent code strings
with the discovered best parameters baked in.

Typical usage:
    from tune import HyperparameterTuner

    tuner = HyperparameterTuner(task="scaling", n_trials=50)
    result = tuner.tune()
    print(result.summary())

    # Pipe best code straight into the environment
    from env import SilentDebuggerEnv
    from config import EnvConfig

    env = SilentDebuggerEnv(EnvConfig(task="scaling"))
    obs = env.reset()
    obs = env.step(result.agent_code)
    print(obs["reward"])   # should be 1.0

CLI usage:
    python tune.py --task scaling --trials 50
    python tune.py --task attrition --trials 80
    python tune.py --task skewed --trials 100
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from config import EnvConfig
from env import SilentDebuggerEnv

# Silence Optuna's per-trial logging (we print our own summary)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


# ====================================================================== #
#  RESULT DATACLASS                                                        #
# ====================================================================== #

@dataclass
class TuneResult:
    """
    Returned by HyperparameterTuner.tune().

    Fields:
        task        : "scaling" | "attrition" | "skewed"
        best_params : dict of best hyperparameters found by Optuna
        best_score  : best cross-validated metric score (accuracy or -RMSE)
        n_trials    : number of Optuna trials actually completed
        study       : the full optuna.Study object (for plots / further analysis)
        agent_code  : ready-to-use Python code string for env.step()
        baseline    : baseline score (untuned default params)
        improvement : score improvement over baseline
    """
    task:        str
    best_params: dict
    best_score:  float
    n_trials:    int
    study:       optuna.Study
    agent_code:  str
    baseline:    float = 0.0
    improvement: float = 0.0

    def summary(self) -> str:
        """Returns a formatted human-readable summary string."""
        is_rmse = self.task == "skewed"
        metric_name = "RMSE (↓ lower is better)" if is_rmse else "Accuracy (↑ higher is better)"
        lines = [
            "",
            "=" * 60,
            f"  TUNING RESULT — Task: {self.task.upper()}",
            "=" * 60,
            f"  Trials completed : {self.n_trials}",
            f"  Metric           : {metric_name}",
            f"  Baseline         : {self.baseline:.4f}",
            f"  Best             : {self.best_score:.4f}",
            f"  Improvement      : {self.improvement:+.4f}  "
            f"({'↓ RMSE reduced' if is_rmse else '↑ accuracy gained'})",
            "",
            "  Best Hyperparameters:",
        ]
        for k, v in self.best_params.items():
            lines.append(f"    {k:<30} = {v}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ====================================================================== #
#  HYPERPARAMETER TUNER                                                    #
# ====================================================================== #

class HyperparameterTuner:
    """
    Optuna-based hyperparameter tuner for all 3 Silent Data Debugger tasks.

    Workflow per task:
        1. reset() the environment to get the canonical broken DataFrame + meta
        2. Apply the standard data fix (scaling / imputation / log-transform)
        3. Run an Optuna study with n_trials to find the best model params
        4. Compute baseline (default params) for comparison
        5. Return a TuneResult with best_params + ready-to-use agent_code

    Args:
        task         : "scaling" | "attrition" | "skewed"
        n_trials     : number of Optuna optimization trials (default: 50)
        cv_folds     : number of cross-validation folds (default: 5)
        timeout      : max seconds for the study (None = unlimited)
        config       : optional EnvConfig override (uses defaults if None)
        verbose      : if True, prints per-trial progress
    """

    def __init__(
        self,
        task:     str = "scaling",
        n_trials: int = 50,
        cv_folds: int = 5,
        timeout:  Optional[float] = None,
        config:   Optional[EnvConfig] = None,
        verbose:  bool = True,
    ):
        if task not in ("scaling", "attrition", "skewed"):
            raise ValueError(f"Unknown task '{task}'. Choose: scaling | attrition | skewed")

        self.task     = task
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.timeout  = timeout
        self.verbose  = verbose

        # Build environment with the right task set
        cfg = config or EnvConfig()
        cfg.task = task
        self.env = SilentDebuggerEnv(cfg)

        # These are populated during tune()
        self._df:   Optional[pd.DataFrame] = None
        self._meta: dict = {}

    # ------------------------------------------------------------------ #
    #  PUBLIC ENTRY POINT                                                  #
    # ------------------------------------------------------------------ #

    def tune(self) -> TuneResult:
        """
        Run the full hyperparameter search and return a TuneResult.
        """
        if self.verbose:
            print(f"\n🔧 Starting Optuna search for task='{self.task}' "
                  f"({self.n_trials} trials, {self.cv_folds}-fold CV)...")

        # Step 1: Generate broken data via env.reset()
        obs = self.env.reset()
        self._df   = self.env.df.copy()
        self._meta = self.env.meta.copy()

        if self.verbose:
            print(f"   Data shape : {self._df.shape}")
            print(f"   Task meta  : {self._meta}")

        # Step 2: Dispatch to task-specific tuner
        dispatch = {
            "scaling":   self._tune_scaling,
            "attrition": self._tune_attrition,
            "skewed":    self._tune_skewed,
        }
        return dispatch[self.task]()

    # ------------------------------------------------------------------ #
    #  TASK 1 — SCALING (KNN Classifier)                                   #
    # ------------------------------------------------------------------ #

    def _tune_scaling(self) -> TuneResult:
        """
        Tunes KNeighborsClassifier after applying StandardScaler.

        Search space:
            scaler_type   : StandardScaler | MinMaxScaler | RobustScaler
            n_neighbors   : 1 – 30
            weights       : uniform | distance
            metric        : euclidean | manhattan | chebyshev | minkowski
            p             : 1 – 4  (only for minkowski)
        """
        df         = self._df
        target_col = self._meta["target_col"]

        X = df.drop(columns=[target_col]).values
        y = df[target_col].values

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        # --- Baseline: default KNN, no scaling ---
        baseline_knn = KNeighborsClassifier()
        baseline_scores = cross_val_score(baseline_knn, X, y, cv=cv, scoring="accuracy")
        baseline = float(baseline_scores.mean())

        def objective(trial: optuna.Trial) -> float:
            # --- Scaler ---
            scaler_name = trial.suggest_categorical(
                "scaler", ["StandardScaler", "MinMaxScaler", "RobustScaler"]
            )
            scaler_cls = {
                "StandardScaler": StandardScaler,
                "MinMaxScaler":   MinMaxScaler,
                "RobustScaler":   RobustScaler,
            }[scaler_name]
            X_scaled = scaler_cls().fit_transform(X)

            # --- KNN hyperparameters ---
            n_neighbors = trial.suggest_int("n_neighbors", 1, 30)
            weights     = trial.suggest_categorical("weights", ["uniform", "distance"])
            metric      = trial.suggest_categorical(
                "metric", ["euclidean", "manhattan", "chebyshev", "minkowski"]
            )
            knn_kwargs = dict(n_neighbors=n_neighbors, weights=weights, metric=metric)
            if metric == "minkowski":
                knn_kwargs["p"] = trial.suggest_int("p", 1, 4)

            model = KNeighborsClassifier(**knn_kwargs)
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")
            return float(scores.mean())

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout,
                       show_progress_bar=self.verbose)

        best = study.best_params
        best_score = study.best_value

        if self.verbose:
            print(f"   ✅ Best accuracy  : {best_score:.4f}  (baseline: {baseline:.4f})")
            print(f"   Best params      : {best}")

        agent_code = self._generate_scaling_code(best)

        return TuneResult(
            task=self.task,
            best_params=best,
            best_score=best_score,
            n_trials=len(study.trials),
            study=study,
            agent_code=agent_code,
            baseline=baseline,
            improvement=best_score - baseline,
        )

    def _generate_scaling_code(self, params: dict) -> str:
        scaler  = params.get("scaler", "StandardScaler")
        n_nbrs  = params.get("n_neighbors", 5)
        weights = params.get("weights", "uniform")
        metric  = params.get("metric", "euclidean")
        p       = params.get("p", 2)

        metric_arg = f'metric="{metric}"'
        if metric == "minkowski":
            metric_arg += f", p={p}"

        return dedent(f"""\
            from sklearn.preprocessing import {scaler}
            from sklearn.neighbors import KNeighborsClassifier

            # --- Tuned by Optuna (tune.py) ---
            scaler = {scaler}()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled  = scaler.transform(X_test)

            model = KNeighborsClassifier(
                n_neighbors={n_nbrs},
                weights="{weights}",
                {metric_arg}
            )
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
        """)

    # ------------------------------------------------------------------ #
    #  TASK 2 — ATTRITION (RandomForest Classifier)                        #
    # ------------------------------------------------------------------ #

    def _tune_attrition(self) -> TuneResult:
        """
        Tunes RandomForestClassifier after imputing NaNs + encoding categoricals.

        Search space:
            n_estimators      : 50 – 500
            max_depth         : 3 – 25 (or None)
            min_samples_split : 2 – 20
            min_samples_leaf  : 1 – 10
            max_features      : sqrt | log2 | 0.3 – 0.9
            class_weight      : None | balanced
        """
        df   = self._df.copy()
        meta = self._meta

        # Apply the canonical fix: median impute + one-hot encode
        for col in meta["null_cols"]:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        df = pd.get_dummies(df, columns=meta["cat_cols"], drop_first=True)

        target_col = meta["target_col"]
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        # --- Baseline: default RandomForest ---
        baseline_rf = RandomForestClassifier(random_state=42)
        baseline_scores = cross_val_score(baseline_rf, X, y, cv=cv, scoring="accuracy")
        baseline = float(baseline_scores.mean())

        def objective(trial: optuna.Trial) -> float:
            n_estimators      = trial.suggest_int("n_estimators", 50, 500, step=50)
            use_max_depth     = trial.suggest_categorical("use_max_depth", [True, False])
            max_depth         = trial.suggest_int("max_depth", 3, 25) if use_max_depth else None
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 10)
            max_features      = trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7, 0.9]
            )
            class_weight      = trial.suggest_categorical("class_weight", [None, "balanced"])

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1,
            )
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            return float(scores.mean())

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout,
                       show_progress_bar=self.verbose)

        best = study.best_params
        best_score = study.best_value

        if self.verbose:
            print(f"   ✅ Best accuracy  : {best_score:.4f}  (baseline: {baseline:.4f})")
            print(f"   Best params      : {best}")

        agent_code = self._generate_attrition_code(best, meta)

        return TuneResult(
            task=self.task,
            best_params=best,
            best_score=best_score,
            n_trials=len(study.trials),
            study=study,
            agent_code=agent_code,
            baseline=baseline,
            improvement=best_score - baseline,
        )

    def _generate_attrition_code(self, params: dict, meta: dict) -> str:
        null_cols  = meta["null_cols"]
        cat_cols   = meta["cat_cols"]
        target_col = meta["target_col"]

        n_est         = params.get("n_estimators", 100)
        use_max_depth = params.get("use_max_depth", True)
        max_depth     = params.get("max_depth", None) if use_max_depth else None
        min_split     = params.get("min_samples_split", 2)
        min_leaf      = params.get("min_samples_leaf", 1)
        max_feat      = params.get("max_features", "sqrt")
        class_weight  = params.get("class_weight", None)

        max_depth_str  = str(max_depth) if max_depth else "None"
        max_feat_str   = f'"{max_feat}"' if isinstance(max_feat, str) else str(max_feat)
        class_w_str    = f'"{class_weight}"' if class_weight else "None"
        null_cols_str  = repr(null_cols)
        cat_cols_str   = repr(cat_cols)

        return dedent(f"""\
            import pandas as pd
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score

            # --- Tuned by Optuna (tune.py) ---
            # Step 1: Impute null columns with median
            for col in {null_cols_str}:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())

            # Step 2: One-hot encode categorical columns
            df = pd.get_dummies(df, columns={cat_cols_str}, drop_first=True)

            # Step 3: Split and train tuned RandomForest
            X = df.drop(columns=["{target_col}"])
            y = df["{target_col}"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            model = RandomForestClassifier(
                n_estimators={n_est},
                max_depth={max_depth_str},
                min_samples_split={min_split},
                min_samples_leaf={min_leaf},
                max_features={max_feat_str},
                class_weight={class_w_str},
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
        """)

    # ------------------------------------------------------------------ #
    #  TASK 3 — SKEWED REGRESSION (model selection + hyperparameters)      #
    # ------------------------------------------------------------------ #

    def _tune_skewed(self) -> TuneResult:
        """
        Tunes regression model + hyperparameters after log1p transform of target.

        The model family itself is a hyperparameter:
            regressor : Ridge | Lasso | ElasticNet | GradientBoostingRegressor

        Per-model search spaces:
            Ridge          : alpha (1e-4 – 1e3, log)
            Lasso          : alpha (1e-4 – 1e2, log)
            ElasticNet     : alpha (1e-4 – 1e2, log) + l1_ratio (0.0 – 1.0)
            GBR            : n_estimators, learning_rate, max_depth,
                             subsample, min_samples_leaf

        Objective: minimize RMSE on original scale (after expm1 inverse transform).
        """
        df         = self._df.copy()
        target_col = self._meta["target_col"]

        X = df.drop(columns=[target_col]).values
        y = df[target_col].values

        # Apply log1p to target (the canonical fix for this task)
        y_log = np.log1p(y)

        X_train, X_test, y_train_log, y_test = train_test_split(
            X, y_log, test_size=0.2, random_state=42
        )
        # Keep original y_test for scoring on real scale
        _, _, _, y_test_orig = train_test_split(X, y, test_size=0.2, random_state=42)

        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        # --- Baseline: LinearRegression with log-transform applied ---
        baseline_model = LinearRegression()
        baseline_model.fit(X_train, y_train_log)  # train on log-scale target
        baseline_preds = np.expm1(baseline_model.predict(X_test))  # inverse back
        baseline_rmse  = float(np.sqrt(mean_squared_error(y_test_orig, baseline_preds)))

        def objective(trial: optuna.Trial) -> float:
            regressor_name = trial.suggest_categorical(
                "regressor", ["Ridge", "Lasso", "ElasticNet", "GradientBoosting"]
            )

            if regressor_name == "Ridge":
                alpha = trial.suggest_float("alpha", 1e-4, 1e3, log=True)
                model = Ridge(alpha=alpha)

            elif regressor_name == "Lasso":
                alpha = trial.suggest_float("alpha", 1e-4, 1e2, log=True)
                model = Lasso(alpha=alpha, max_iter=5000)

            elif regressor_name == "ElasticNet":
                alpha     = trial.suggest_float("alpha", 1e-4, 1e2, log=True)
                l1_ratio  = trial.suggest_float("l1_ratio", 0.0, 1.0)
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)

            else:  # GradientBoosting
                n_estimators     = trial.suggest_int("n_estimators", 50, 400, step=50)
                learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
                max_depth        = trial.suggest_int("max_depth", 2, 8)
                subsample        = trial.suggest_float("subsample", 0.5, 1.0)
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
                model = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=subsample,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                )

            # CV on train split only (no leakage of test rows into folds)
            fold_rmses = []
            for train_idx, val_idx in cv.split(X_train):
                model.fit(X_train[train_idx], y_train_log[train_idx])
                preds_log  = model.predict(X_train[val_idx])
                preds_orig = np.expm1(np.clip(preds_log, -10, 20))  # numeric safety
                y_val_orig = np.expm1(y_train_log[val_idx])
                rmse = float(np.sqrt(mean_squared_error(y_val_orig, preds_orig)))
                fold_rmses.append(rmse)

            return -float(np.mean(fold_rmses))   # Optuna maximizes → maximize -RMSE

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout,
                       show_progress_bar=self.verbose)

        best      = study.best_params
        best_rmse = -study.best_value   # convert back to positive RMSE

        if self.verbose:
            print(f"   ✅ Best RMSE      : {best_rmse:.4f}  (baseline: {baseline_rmse:.4f})")
            print(f"   Best params      : {best}")

        agent_code = self._generate_skewed_code(best)

        # Store actual RMSE values (positive). Improvement is negative when RMSE drops.
        return TuneResult(
            task=self.task,
            best_params=best,
            best_score=best_rmse,              # lower is better for RMSE
            n_trials=len(study.trials),
            study=study,
            agent_code=agent_code,
            baseline=baseline_rmse,
            improvement=baseline_rmse - best_rmse,  # positive = improvement
        )

    def _generate_skewed_code(self, params: dict) -> str:
        regressor = params.get("regressor", "Ridge")

        if regressor == "Ridge":
            alpha   = params.get("alpha", 1.0)
            imports = "from sklearn.linear_model import Ridge"
            model   = f"Ridge(alpha={alpha:.6f})"

        elif regressor == "Lasso":
            alpha   = params.get("alpha", 1.0)
            imports = "from sklearn.linear_model import Lasso"
            model   = f"Lasso(alpha={alpha:.6f}, max_iter=5000)"

        elif regressor == "ElasticNet":
            alpha    = params.get("alpha", 1.0)
            l1_ratio = params.get("l1_ratio", 0.5)
            imports  = "from sklearn.linear_model import ElasticNet"
            model    = f"ElasticNet(alpha={alpha:.6f}, l1_ratio={l1_ratio:.4f}, max_iter=5000)"

        else:  # GradientBoosting
            n_est   = params.get("n_estimators", 100)
            lr      = params.get("learning_rate", 0.1)
            depth   = params.get("max_depth", 3)
            subs    = params.get("subsample", 1.0)
            leaf    = params.get("min_samples_leaf", 1)
            imports = "from sklearn.ensemble import GradientBoostingRegressor"
            # Build as a single line to avoid indentation issues inside the dedent block
            model   = (
                f"GradientBoostingRegressor("
                f"n_estimators={n_est}, learning_rate={lr:.6f}, "
                f"max_depth={depth}, subsample={subs:.4f}, "
                f"min_samples_leaf={leaf}, random_state=42)"
            )

        # NOTE: {imports} and {model} must not contain leading whitespace,
        # because dedent() strips the common indent from the template lines.
        return dedent(f"""\
            import numpy as np
            {imports}

            # --- Tuned by Optuna (tune.py) ---
            # Step 1: Log-transform the skewed target
            y_train_log = np.log1p(y_train)

            # Step 2: Train tuned model on log-scale target
            model = {model}
            model.fit(X_train, y_train_log)

            # Step 3: Predict and inverse-transform back to original scale
            y_pred_log  = model.predict(X_test)
            predictions = np.expm1(np.clip(y_pred_log, -10, 20))
        """)


# ====================================================================== #
#  UTILITY — Run all 3 tasks and print comparison table                   #
# ====================================================================== #

def tune_all_tasks(n_trials: int = 50, verbose: bool = True) -> dict[str, TuneResult]:
    """
    Convenience function: runs tuning for all 3 tasks and returns results dict.

    Returns:
        {"scaling": TuneResult, "attrition": TuneResult, "skewed": TuneResult}
    """
    results = {}
    for task in ("scaling", "attrition", "skewed"):
        tuner = HyperparameterTuner(task=task, n_trials=n_trials, verbose=verbose)
        results[task] = tuner.tune()

    # Print summary table
    print("\n" + "=" * 72)
    print(f"  {'TASK':<15} {'BASELINE':>12} {'BEST':>12} {'IMPROVEMENT':>14}  METRIC")
    print("=" * 72)
    for task, r in results.items():
        if task == "skewed":
            metric = "RMSE ↓"
        else:
            metric = "Acc  ↑"
        print(
            f"  {task:<15} {r.baseline:>11.4f}  {r.best_score:>11.4f}  "
            f"{r.improvement:>+13.4f}  {metric}"
        )
    print("=" * 72)
    return results


def run_tuned_env(result: TuneResult) -> dict:
    """
    Re-runs the environment using the tuned agent code from a TuneResult.
    Returns the observation dict from env.step().

    Useful to verify the tuned code actually earns reward in the env.
    """
    cfg = EnvConfig(task=result.task)
    env = SilentDebuggerEnv(cfg)
    env.reset()
    obs = env.step(result.agent_code)
    return obs


# ====================================================================== #
#  CLI ENTRY POINT                                                         #
# ====================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for the Silent Data Debugger environment."
    )
    parser.add_argument(
        "--task",
        choices=["scaling", "attrition", "skewed", "all"],
        default="all",
        help="Which task to tune (default: all).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna trials per task (default: 50).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Max seconds per task study (default: None = unlimited).",
    )
    parser.add_argument(
        "--run-env",
        action="store_true",
        help="After tuning, run env.step() with the best code and print reward.",
    )
    args = parser.parse_args()

    if args.task == "all":
        results = tune_all_tasks(n_trials=args.trials, verbose=True)
    else:
        tuner  = HyperparameterTuner(
            task=args.task,
            n_trials=args.trials,
            cv_folds=args.folds,
            timeout=args.timeout,
            verbose=True,
        )
        result  = tuner.tune()
        results = {args.task: result}
        print(result.summary())
        print("\n📋 Generated Agent Code:\n")
        print(result.agent_code)

    if args.run_env:
        print("\n🚀 Running tuned code through the environment...\n")
        for task, res in results.items():
            obs = run_tuned_env(res)
            print(f"  Task: {task:<12}  Reward: {obs['reward']:.2f}  Done: {obs['done']}")
            for line in obs["log"].split("\n"):
                print(f"    {line}")


# ====================================================================== #
#  GENERIC TUNER — works on any df + target_col (upload mode)            #
# ====================================================================== #

class GenericTuner:
    """
    Hyperparameter tuner for arbitrary user-uploaded datasets.

    Unlike HyperparameterTuner (which is tied to 3 demo tasks),
    this class accepts any clean DataFrame + target column and:
      - Auto-detects task type: classification (≤10 unique target values)
        or regression (>10 unique values)
      - Searches across multiple model families via Optuna TPE
      - Returns a TuneResult with best_params + ready-to-use agent_code

    Expected usage (called AFTER data cleaning + feature engineering):
        from tune import GenericTuner
        result = GenericTuner(df_clean, target_col="SalePrice", n_trials=60).tune()
        print(result.summary())
        print(result.agent_code)  # pipe into env.step() or execute directly
    """

    CLASSIFICATION_THRESHOLD = 10

    def __init__(
        self,
        df:         pd.DataFrame,
        target_col: str,
        n_trials:   int = 50,
        cv_folds:   int = 5,
        verbose:    bool = True,
    ):
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not found in DataFrame.")

        self.df         = df.copy()
        self.target_col = target_col
        self.n_trials   = n_trials
        self.cv_folds   = cv_folds
        self.verbose    = verbose

        n_unique = int(self.df[target_col].nunique())
        self.task_type = "classification" if n_unique <= self.CLASSIFICATION_THRESHOLD else "regression"

        if self.verbose:
            print(f"\n🔍 GenericTuner: detected task_type='{self.task_type}' "
                  f"({n_unique} unique target values)")

    def tune(self) -> TuneResult:
        X_df = self.df.drop(columns=[self.target_col])
        y    = self.df[self.target_col]

        for col in X_df.select_dtypes(include=["object", "category"]).columns:
            X_df[col] = X_df[col].astype("category").cat.codes

        X = X_df.values
        y_arr = y.values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_arr, test_size=0.2, random_state=42
        )

        if self.task_type == "classification":
            return self._tune_classification(X, y_arr, X_train, X_test, y_train, y_test)
        else:
            return self._tune_regression(X, y_arr, X_train, X_test, y_train, y_test)

    def _tune_classification(self, X, y, X_train, X_test, y_train, y_test) -> TuneResult:
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        baseline_score = float(
            cross_val_score(RandomForestClassifier(random_state=42), X, y,
                            cv=cv, scoring="accuracy").mean()
        )

        def objective(trial: optuna.Trial) -> float:
            model_name = trial.suggest_categorical(
                "model", ["RandomForest", "GradientBoosting", "KNN"]
            )

            if model_name == "RandomForest":
                n_est      = trial.suggest_int("n_estimators", 50, 500, step=50)
                use_depth  = trial.suggest_categorical("use_max_depth", [True, False])
                max_depth  = trial.suggest_int("max_depth", 3, 20) if use_depth else None
                min_split  = trial.suggest_int("min_samples_split", 2, 20)
                min_leaf   = trial.suggest_int("min_samples_leaf", 1, 10)
                max_feat   = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7])
                model = RandomForestClassifier(
                    n_estimators=n_est, max_depth=max_depth,
                    min_samples_split=min_split, min_samples_leaf=min_leaf,
                    max_features=max_feat, random_state=42, n_jobs=-1,
                )

            elif model_name == "GradientBoosting":
                n_est = trial.suggest_int("n_estimators", 50, 300, step=50)
                lr    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
                depth = trial.suggest_int("max_depth", 2, 8)
                subs  = trial.suggest_float("subsample", 0.5, 1.0)
                model = GradientBoostingClassifier(
                    n_estimators=n_est, learning_rate=lr,
                    max_depth=depth, subsample=subs, random_state=42,
                )

            else:  # KNN
                n_nbrs  = trial.suggest_int("n_neighbors", 1, 30)
                weights = trial.suggest_categorical("weights", ["uniform", "distance"])
                metric  = trial.suggest_categorical("knn_metric", ["euclidean", "manhattan"])
                model = KNeighborsClassifier(n_neighbors=n_nbrs, weights=weights, metric=metric)

            return float(
                cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()
            )

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=self.verbose)

        best       = study.best_params
        best_score = study.best_value

        if self.verbose:
            print(f"   ✅ Best accuracy : {best_score:.4f}  (baseline: {baseline_score:.4f})")
            print(f"   Best params     : {best}")

        return TuneResult(
            task="generic_classification",
            best_params=best,
            best_score=best_score,
            n_trials=len(study.trials),
            study=study,
            agent_code=self._generate_classification_code(best),
            baseline=baseline_score,
            improvement=best_score - baseline_score,
        )

    def _tune_regression(self, X, y, X_train, X_test, y_train, y_test) -> TuneResult:
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        baseline_model = Ridge()
        baseline_model.fit(X_train, y_train)
        baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_model.predict(X_test))))

        def objective(trial: optuna.Trial) -> float:
            model_name = trial.suggest_categorical(
                "model", ["Ridge", "Lasso", "ElasticNet", "GradientBoosting"]
            )

            if model_name == "Ridge":
                alpha = trial.suggest_float("alpha", 1e-4, 1e3, log=True)
                model = Ridge(alpha=alpha)

            elif model_name == "Lasso":
                alpha = trial.suggest_float("alpha", 1e-4, 1e2, log=True)
                model = Lasso(alpha=alpha, max_iter=5000)

            elif model_name == "ElasticNet":
                alpha    = trial.suggest_float("alpha", 1e-4, 1e2, log=True)
                l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)

            else:  # GradientBoosting
                n_est = trial.suggest_int("n_estimators", 50, 300, step=50)
                lr    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
                depth = trial.suggest_int("max_depth", 2, 8)
                subs  = trial.suggest_float("subsample", 0.5, 1.0)
                model = GradientBoostingRegressor(
                    n_estimators=n_est, learning_rate=lr,
                    max_depth=depth, subsample=subs, random_state=42,
                )

            fold_rmses = []
            for ti, vi in cv.split(X_train):
                model.fit(X_train[ti], y_train[ti])
                preds = model.predict(X_train[vi])
                fold_rmses.append(float(np.sqrt(mean_squared_error(y_train[vi], preds))))
            return -float(np.mean(fold_rmses))

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=self.verbose)

        best      = study.best_params
        best_rmse = -study.best_value

        if self.verbose:
            print(f"   ✅ Best RMSE     : {best_rmse:.4f}  (baseline: {baseline_rmse:.4f})")
            print(f"   Best params     : {best}")

        return TuneResult(
            task="generic_regression",
            best_params=best,
            best_score=best_rmse,
            n_trials=len(study.trials),
            study=study,
            agent_code=self._generate_regression_code(best),
            baseline=baseline_rmse,
            improvement=baseline_rmse - best_rmse,
        )

    def _generate_classification_code(self, params: dict) -> str:
        model_name = params.get("model", "RandomForest")

        if model_name == "RandomForest":
            n_est      = params.get("n_estimators", 100)
            use_depth  = params.get("use_max_depth", False)
            max_depth  = params.get("max_depth", None) if use_depth else None
            depth_str  = str(max_depth) if max_depth is not None else "None"
            min_split  = params.get("min_samples_split", 2)
            min_leaf   = params.get("min_samples_leaf", 1)
            max_feat   = params.get("max_features", "sqrt")
            feat_str   = f'"{max_feat}"' if isinstance(max_feat, str) else str(max_feat)
            return (
                f"from sklearn.ensemble import RandomForestClassifier\n"
                f"from sklearn.metrics import accuracy_score\n\n"
                f"model = RandomForestClassifier(\n"
                f"    n_estimators={n_est}, max_depth={depth_str},\n"
                f"    min_samples_split={min_split}, min_samples_leaf={min_leaf},\n"
                f"    max_features={feat_str}, random_state=42, n_jobs=-1\n"
                f")\n"
                f"model.fit(X_train, y_train)\n"
                f"predictions = model.predict(X_test)\n"
                f"score = accuracy_score(y_test, predictions)\n"
            )

        elif model_name == "GradientBoosting":
            n_est = params.get("n_estimators", 100)
            lr    = params.get("learning_rate", 0.1)
            depth = params.get("max_depth", 3)
            subs  = params.get("subsample", 1.0)
            return (
                f"from sklearn.ensemble import GradientBoostingClassifier\n"
                f"from sklearn.metrics import accuracy_score\n\n"
                f"model = GradientBoostingClassifier("
                f"n_estimators={n_est}, learning_rate={lr:.6f}, "
                f"max_depth={depth}, subsample={subs:.4f}, random_state=42)\n"
                f"model.fit(X_train, y_train)\n"
                f"predictions = model.predict(X_test)\n"
                f"score = accuracy_score(y_test, predictions)\n"
            )

        else:  # KNN
            n_nbrs  = params.get("n_neighbors", 5)
            weights = params.get("weights", "uniform")
            metric  = params.get("knn_metric", "euclidean")
            return (
                f"from sklearn.neighbors import KNeighborsClassifier\n"
                f"from sklearn.metrics import accuracy_score\n\n"
                f"model = KNeighborsClassifier("
                f"n_neighbors={n_nbrs}, weights='{weights}', metric='{metric}')\n"
                f"model.fit(X_train, y_train)\n"
                f"predictions = model.predict(X_test)\n"
                f"score = accuracy_score(y_test, predictions)\n"
            )

    def _generate_regression_code(self, params: dict) -> str:
        model_name = params.get("model", "Ridge")

        if model_name == "Ridge":
            alpha = params.get("alpha", 1.0)
            return (
                f"from sklearn.linear_model import Ridge\n"
                f"from sklearn.metrics import mean_squared_error\nimport numpy as np\n\n"
                f"model = Ridge(alpha={alpha:.6f})\n"
                f"model.fit(X_train, y_train)\n"
                f"predictions = model.predict(X_test)\n"
                f"score = float(np.sqrt(mean_squared_error(y_test, predictions)))\n"
            )

        elif model_name == "Lasso":
            alpha = params.get("alpha", 1.0)
            return (
                f"from sklearn.linear_model import Lasso\n"
                f"from sklearn.metrics import mean_squared_error\nimport numpy as np\n\n"
                f"model = Lasso(alpha={alpha:.6f}, max_iter=5000)\n"
                f"model.fit(X_train, y_train)\n"
                f"predictions = model.predict(X_test)\n"
                f"score = float(np.sqrt(mean_squared_error(y_test, predictions)))\n"
            )

        elif model_name == "ElasticNet":
            alpha    = params.get("alpha", 1.0)
            l1_ratio = params.get("l1_ratio", 0.5)
            return (
                f"from sklearn.linear_model import ElasticNet\n"
                f"from sklearn.metrics import mean_squared_error\nimport numpy as np\n\n"
                f"model = ElasticNet(alpha={alpha:.6f}, l1_ratio={l1_ratio:.4f}, max_iter=5000)\n"
                f"model.fit(X_train, y_train)\n"
                f"predictions = model.predict(X_test)\n"
                f"score = float(np.sqrt(mean_squared_error(y_test, predictions)))\n"
            )

        else:  # GradientBoosting
            n_est = params.get("n_estimators", 100)
            lr    = params.get("learning_rate", 0.1)
            depth = params.get("max_depth", 3)
            subs  = params.get("subsample", 1.0)
            return (
                f"from sklearn.ensemble import GradientBoostingRegressor\n"
                f"from sklearn.metrics import mean_squared_error\nimport numpy as np\n\n"
                f"model = GradientBoostingRegressor("
                f"n_estimators={n_est}, learning_rate={lr:.6f}, "
                f"max_depth={depth}, subsample={subs:.4f}, random_state=42)\n"
                f"model.fit(X_train, y_train)\n"
                f"predictions = model.predict(X_test)\n"
                f"score = float(np.sqrt(mean_squared_error(y_test, predictions)))\n"
            )


def generic_tune(
    df:         pd.DataFrame,
    target_col: str,
    n_trials:   int = 50,
    cv_folds:   int = 5,
    verbose:    bool = True,
) -> TuneResult:
    return GenericTuner(df, target_col, n_trials, cv_folds, verbose).tune()
