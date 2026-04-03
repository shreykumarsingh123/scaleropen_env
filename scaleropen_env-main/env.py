import optuna
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from config import EnvConfig, ScalingTaskConfig, AttritionTaskConfig, SkewedTaskConfig
from state import get_state
from evaluate import grade_scaling_task, grade_attrition_task, grade_skewed_task
from visualize import build_scaling_figures, build_attrition_figures, build_skewed_figures


# ====================================================================== #
#  DATA GENERATORS                                                        #
#  Each function builds and breaks a dataset according to its config.    #
#  Returns the broken df + any metadata needed by the grader.            #
# ====================================================================== #

def _generate_scaling_data(cfg: ScalingTaskConfig) -> tuple[pd.DataFrame, dict]:
    """
    Synthetic classification data where feature_0 is inflated by scale_factor.
    The rest of the features are in [0, 1] range.
    """
    rng = np.random.RandomState(cfg.random_state)
    X, y = make_classification(
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        n_classes=cfg.n_classes,
        n_informative=max(2, cfg.n_features - 1),
        n_redundant=0,
        random_state=cfg.random_state,
    )
    # Shift X to [0,1] range for clean scale contrast
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

    # Inject scale mismatch on feature_0
    inflated_col = "feature_0"
    X[:, 0] = X[:, 0] * cfg.scale_factor

    cols = [f"feature_{i}" for i in range(cfg.n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y

    # Compute baseline accuracy on broken data
    X_split = df.drop(columns=["target"])
    y_split = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_split, y_split, test_size=cfg.test_size, random_state=cfg.random_state
    )
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    baseline_accuracy = accuracy_score(y_test, model.predict(X_test))

    meta = {
        "target_col": "target",
        "inflated_col": inflated_col,
        "baseline_accuracy": baseline_accuracy,
        "improvement_threshold": cfg.improvement_threshold,
    }
    return df, meta


def _generate_attrition_data(cfg: AttritionTaskConfig) -> tuple[pd.DataFrame, dict]:
    """
    Simulated HR attrition dataset with injected nulls in null_cols
    and unencoded categorical columns.
    """
    rng = np.random.RandomState(cfg.random_state)
    n = cfg.n_samples

    departments = ["Sales", "HR", "R&D", "Finance", "IT"]
    job_roles = ["Manager", "Analyst", "Engineer", "Director", "Clerk"]
    overtime = ["Yes", "No"]

    df = pd.DataFrame({
        "Age":              rng.randint(22, 60, n),
        "MonthlyIncome":    rng.randint(2000, 20000, n),
        "YearsAtCompany":   rng.randint(0, 30, n),
        "DistanceFromHome": rng.randint(1, 30, n),
        "JobSatisfaction":  rng.randint(1, 5, n),
        "Department":       rng.choice(departments, n),
        "JobRole":          rng.choice(job_roles, n),
        "OverTime":         rng.choice(overtime, n),
        "Attrition":        rng.choice([0, 1], n, p=[0.84, 0.16]),
    })

    # Inject nulls into specified columns
    for col in cfg.null_cols:
        if col in df.columns:
            null_idx = rng.choice(df.index, size=int(n * cfg.null_rate), replace=False)
            df.loc[null_idx, col] = np.nan

    meta = {
        "target_col": cfg.target_col,
        "null_cols": cfg.null_cols,
        "cat_cols": cfg.cat_cols,
    }
    return df, meta


def _generate_skewed_data(cfg: SkewedTaskConfig) -> tuple[pd.DataFrame, dict]:
    """
    Regression dataset with a log-normal target that is correlated with X.

    Generation:
        X        ~ N(0, 1)
        weights  ~ N(0, 1), normalised to unit length
        y_log    = skew_param * (X @ weights) + noise   (scaled linear signal)
        y        = exp(y_log)   → log-normal, heavy right skew

    With skew_param=3, E[y]≈90 and std(y)≈8000, so raw LinearRegression
    on y produces huge RMSE.  The agent's log1p + expm1 fix reduces RMSE
    far below the 50 % threshold.
    """
    rng = np.random.RandomState(cfg.random_state)
    n = cfg.n_samples

    X = rng.randn(n, cfg.n_features)
    # Normalised weights make the log-scale signal stable across runs
    weights = rng.randn(cfg.n_features)
    weights = weights / (np.linalg.norm(weights) + 1e-8)
    # Scale by skew_param: exp(3*z) creates the heavy right-skew needed
    y_log = cfg.skew_param * (X @ weights) + rng.randn(n) * cfg.noise_level
    y = np.exp(y_log)

    cols = [f"feature_{i}" for i in range(cfg.n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y

    # Baseline: LinearRegression on raw (un-transformed) target
    X_split = df.drop(columns=["target"])
    y_split = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_split, y_split, test_size=cfg.test_size, random_state=cfg.random_state
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = np.clip(model.predict(X_test), 0, None)  # RMSLE requires non-negative
    # Use RMSLE (log-scale error) so the metric is not dominated by extreme outliers
    baseline_rmsle = float(np.sqrt(mean_squared_error(
        np.log1p(y_test), np.log1p(preds)
    )))
    rmsle_threshold = baseline_rmsle * cfg.rmse_improvement_factor

    meta = {
        "target_col":     "target",
        "baseline_rmsle": baseline_rmsle,
        "rmsle_threshold": rmsle_threshold,
    }
    return df, meta


# ====================================================================== #
#  SILENT DATA DEBUGGER ENVIRONMENT                                       #
# ====================================================================== #

class SilentDebuggerEnv:
    """
    OpenAI Gym-style RL environment for the Silent Data/Model Debugger.

    The agent receives a structured observation dict describing a broken
    dataset, injects Python code to fix it, and receives a reward based
    on how well the fix improves downstream ML performance.

    Observation dict (returned by reset() and step()):
        {
            "task"      : str,          # current task name
            "state"     : str,          # JSON health report from get_state()
            "reward"    : float,        # 0.0 at reset, updated at step()
            "done"      : bool,         # True when reward >= 1.0
            "log"       : str,          # grader feedback for the agent
            "figures"   : dict,         # Plotly figures for the UI
            "meta"      : dict,         # task metadata (baseline scores etc.)
        }

    Usage:
        env = SilentDebuggerEnv(EnvConfig(task="scaling"))
        obs = env.reset()
        obs = env.step(agent_code_string)
    """

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.df: pd.DataFrame | None = None
        self.meta: dict = {}
        self._last_reward: float = 0.0

    # ------------------------------------------------------------------ #
    #  RESET                                                              #
    # ------------------------------------------------------------------ #

    def reset(self) -> dict:
        """
        Generates a fresh broken dataset for the current task.
        Returns the initial observation (reward = 0.0).
        """
        task = self.config.task

        if task == "scaling":
            self.df, self.meta = _generate_scaling_data(self.config.scaling_cfg)
            figures = build_scaling_figures(
                df_before=self.df,
                inflated_col=self.meta["inflated_col"],
                reward=0.0,
            )

        elif task == "attrition":
            self.df, self.meta = _generate_attrition_data(self.config.attrition_cfg)
            figures = build_attrition_figures(
                df_before=self.df,
                target_col=self.meta["target_col"],
                null_cols=self.meta["null_cols"],
                reward=0.0,
            )

        elif task == "skewed":
            self.df, self.meta = _generate_skewed_data(self.config.skewed_cfg)
            figures = build_skewed_figures(
                df_before=self.df,
                target_col=self.meta["target_col"],
                reward=0.0,
            )

        else:
            raise ValueError(f"Unknown task: '{task}'. Choose from: scaling, attrition, skewed.")

        self._last_reward = 0.0

        return {
            "task":    task,
            "state":   get_state(self.df, self.config.global_cfg),
            "reward":  0.0,
            "done":    False,
            "log":     "Environment reset. Awaiting agent fix.",
            "figures": figures,
            "meta":    self.meta,
        }

    # ------------------------------------------------------------------ #
    #  STEP                                                               #
    # ------------------------------------------------------------------ #

    def step(self, agent_code: str) -> dict:
        """
        Executes the agent's code, grades the result, and returns
        a structured observation dict with updated reward and figures.

        Args:
            agent_code : Raw Python code string from the LLM agent.

        Returns:
            Observation dict (same structure as reset()).
        """
        if self.df is None:
            raise RuntimeError("Call reset() before step().")

        task = self.config.task

        # --- Grade based on task ---
        if task == "scaling":
            reward, log = grade_scaling_task(
                agent_code=agent_code,
                df=self.df,
                target_col=self.meta["target_col"],
                baseline_accuracy=self.meta["baseline_accuracy"],
                improvement_threshold=self.meta["improvement_threshold"],
            )
            # Try to get the fixed df from agent namespace for after-figures
            df_after = self._try_get_fixed_df(agent_code, task)
            figures = build_scaling_figures(
                df_before=self.df,
                df_after=df_after,
                inflated_col=self.meta["inflated_col"],
                reward=reward,
            )

        elif task == "attrition":
            reward, log = grade_attrition_task(
                agent_code=agent_code,
                df=self.df,
                target_col=self.meta["target_col"],
                null_cols=self.meta["null_cols"],
                cat_cols=self.meta["cat_cols"],
            )
            df_after = self._try_get_fixed_df(agent_code, task)
            figures = build_attrition_figures(
                df_before=self.df,
                df_after=df_after,
                target_col=self.meta["target_col"],
                null_cols=self.meta["null_cols"],
                reward=reward,
            )

        elif task == "skewed":
            reward, log = grade_skewed_task(
                agent_code=agent_code,
                df=self.df,
                target_col=self.meta["target_col"],
                baseline_rmsle=self.meta["baseline_rmsle"],
                rmsle_threshold=self.meta["rmsle_threshold"],
            )
            df_after = self._try_get_fixed_df(agent_code, task)
            # Extract new RMSLE from log for the comparison chart
            new_rmsle = self._parse_rmsle_from_log(log)
            figures = build_skewed_figures(
                df_before=self.df,
                df_after=df_after,
                target_col=self.meta["target_col"],
                baseline_rmse=self.meta["baseline_rmsle"],
                new_rmse=new_rmsle,
                reward=reward,
            )

        else:
            raise ValueError(f"Unknown task: '{task}'.")

        self._last_reward = reward
        done = reward >= 1.0

        return {
            "task":    task,
            "state":   get_state(self.df, self.config.global_cfg),
            "reward":  reward,
            "done":    done,
            "log":     log,
            "figures": figures,
            "meta":    self.meta,
        }

    # ------------------------------------------------------------------ #
    #  HELPERS                                                            #
    # ------------------------------------------------------------------ #

    def _try_get_fixed_df(self, agent_code: str, task: str) -> pd.DataFrame | None:
        """
        Re-runs agent code silently and returns a 'fixed' DataFrame
        for use in after-fix visualizations. Fails gracefully.

        For scaling: applies StandardScaler to the inflated column directly
        so the after-distribution is meaningful for the chart.
        For attrition/skewed: extracts the modified df from agent namespace.
        """
        try:
            import pandas as pd_
            import numpy as np_

            if task == "scaling":
                # Build a clean scaled copy of df for visualization purposes
                from sklearn.preprocessing import StandardScaler
                df_vis = self.df.copy()
                scaler = StandardScaler()
                num_cols = df_vis.select_dtypes(include=[np_.number]).columns.tolist()
                df_vis[num_cols] = scaler.fit_transform(df_vis[num_cols])
                return df_vis

            elif task == "skewed":
                # Agent only modifies y_train, never df — so build the
                # log-transformed df explicitly for a meaningful after-chart
                target_col = self.meta.get("target_col", "target")
                df_vis = self.df.copy()
                df_vis[target_col] = np_.log1p(df_vis[target_col])
                return df_vis

            else:
                # For attrition and skewed, run agent code and grab modified df
                from evaluate import execute_agent_code
                target_col = self.meta.get("target_col", "target")
                X = self.df.drop(columns=[target_col])
                y = self.df[target_col]
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                context = {
                    "df": self.df.copy(),
                    "X_train": X_train.copy(),
                    "X_test": X_test.copy(),
                    "y_train": y_train.copy(),
                    "y_test": y_test.copy(),
                    "pd": pd_,
                    "np": np_,
                }
                success, _, namespace = execute_agent_code(agent_code, context)
                if success:
                    return namespace.get("df")
        except Exception:
            pass
        return None

    def _parse_rmsle_from_log(self, log: str) -> float | None:
        """Extracts new RMSLE value from grader log string for the chart."""
        import re
        match = re.search(r"New RMSLE:\s*([\d.]+)", log)
        if match:
            return float(match.group(1))
        return None