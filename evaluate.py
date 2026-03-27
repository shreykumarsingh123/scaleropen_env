import io
import re
import ast
import contextlib
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split


# ====================================================================== #
#  CODE EXECUTOR                                                          #
#  This is the core of step(). Takes a raw string from the LLM,         #
#  strips markdown fences, runs it inside a safe exec() context,         #
#  and returns (success: bool, output: str, modified_namespace: dict).   #
# ====================================================================== #

def execute_agent_code(code_str: str, context: dict) -> tuple[bool, str, dict]:
    """
    Executes a string of Python code injected by the LLM agent.

    Args:
        code_str : Raw code string from the LLM (may have markdown fences).
        context  : A dict of variables available to the executed code
                   e.g. {"df": df, "X_train": X_train, ...}

    Returns:
        (success, output, namespace)
        success   : True if code ran without exceptions.
        output    : Captured stdout + any exception message.
        namespace : The exec() local namespace after execution.
                    Use this to retrieve modified variables (e.g. namespace["df"]).
    """
    # Strip markdown code fences if the LLM wrapped its response
    code_str = re.sub(r"```(?:python)?", "", code_str).replace("```", "").strip()

    stdout_capture = io.StringIO()
    namespace = dict(context)  # shallow copy so we don't mutate caller's dict

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code_str, namespace)  # noqa: S102
        output = stdout_capture.getvalue()
        return True, output, namespace
    except Exception as exc:
        output = stdout_capture.getvalue()
        output += f"\n[EXECUTION ERROR] {type(exc).__name__}: {exc}"
        return False, output, namespace


# ====================================================================== #
#  TASK 1 GRADER — The Scaling Failure (KNN)                             #
# ====================================================================== #

def grade_scaling_task(
    agent_code: str,
    df: pd.DataFrame,
    target_col: str,
    baseline_accuracy: float,
    improvement_threshold: float = 0.15,
) -> tuple[float, str]:
    """
    Grades the agent's fix for the KNN scale mismatch task.

    Reward breakdown:
        +0.2  Code executes without errors.
        +0.4  A scaling function is detected in the injected code (AST check).
        +0.4  Model accuracy improves by > improvement_threshold over baseline.

    Args:
        agent_code            : Raw code string from the LLM.
        df                    : The original broken DataFrame.
        target_col            : Name of the target/label column.
        baseline_accuracy     : Accuracy of the unscaled KNN (captured at reset()).
        improvement_threshold : Minimum required accuracy gain (default 15%).

    Returns:
        (reward, log_message)
    """
    reward = 0.0
    log = []

    # --- Step 1: Execute the agent's code ---
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    context = {
        "df": df.copy(),
        "X_train": X_train.copy(),
        "X_test": X_test.copy(),
        "y_train": y_train.copy(),
        "y_test": y_test.copy(),
        "pd": pd,
        "np": np,
    }

    success, output, namespace = execute_agent_code(agent_code, context)

    if success:
        reward += 0.2
        log.append("✅ +0.2 Code executed successfully.")
    else:
        log.append(f"❌ +0.0 Code execution failed:\n{output}")
        return round(reward, 2), "\n".join(log)

    # --- Step 2: AST check for a scaling function ---
    scaling_keywords = {"StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer"}
    try:
        tree = ast.parse(agent_code)
        found_scaler = any(
            (isinstance(node, ast.Name) and node.id in scaling_keywords)
            or (isinstance(node, ast.Attribute) and node.attr in scaling_keywords)
            for node in ast.walk(tree)
        )
        if found_scaler:
            reward += 0.4
            log.append("✅ +0.4 Scaling function detected in agent code.")
        else:
            log.append("❌ +0.0 No scaling function found in agent code.")
    except SyntaxError:
        log.append("❌ +0.0 AST parse failed — agent code has syntax errors.")

    # --- Step 3: Check accuracy improvement ---
    try:
        # Retrieve scaled arrays from namespace if the agent produced them
        X_train_scaled = namespace.get("X_train_scaled", namespace.get("X_train"))
        X_test_scaled = namespace.get("X_test_scaled", namespace.get("X_test"))

        model = KNeighborsClassifier()
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        new_accuracy = accuracy_score(y_test, preds)

        improvement = new_accuracy - baseline_accuracy
        log.append(
            f"📊 Baseline accuracy: {baseline_accuracy:.4f} | "
            f"New accuracy: {new_accuracy:.4f} | "
            f"Improvement: {improvement:+.4f}"
        )

        if improvement >= improvement_threshold:
            reward += 0.4
            log.append(f"✅ +0.4 Accuracy improved by {improvement:.2%} (threshold: {improvement_threshold:.0%}).")
        else:
            log.append(f"❌ +0.0 Accuracy improvement {improvement:.2%} below threshold {improvement_threshold:.0%}.")

    except Exception as exc:
        log.append(f"❌ +0.0 Could not evaluate model accuracy: {exc}")

    return round(min(reward, 1.0), 2), "\n".join(log)


# ====================================================================== #
#  TASK 2 GRADER — The Attrition Imputation Trap                         #
# ====================================================================== #

def grade_attrition_task(
    agent_code: str,
    df: pd.DataFrame,
    target_col: str = "Attrition",
    null_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
) -> tuple[float, str]:
    """
    Grades the agent's fix for the HR attrition task.

    Reward breakdown:
        +0.2  Code executes without errors.
        +0.4  Null columns are fully imputed (no NaNs remain in null_cols).
        +0.4  Categorical columns are encoded and model trains successfully.

    Args:
        agent_code  : Raw code string from the LLM.
        df          : The original broken DataFrame.
        target_col  : Target column name (default "Attrition").
        null_cols   : Columns that were injected with NaNs at reset().
                      Passed in dynamically — no hardcoded column names.
        cat_cols    : Columns that are unencoded categoricals.
    """
    reward = 0.0
    log = []
    null_cols = null_cols or []
    cat_cols = cat_cols or []

    # --- Step 1: Execute the agent's code ---
    context = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
    }

    success, output, namespace = execute_agent_code(agent_code, context)

    if success:
        reward += 0.2
        log.append("✅ +0.2 Code executed successfully.")
    else:
        log.append(f"❌ +0.0 Code execution failed:\n{output}")
        return round(reward, 2), "\n".join(log)

    df_fixed = namespace.get("df", df)

    # --- Step 2: Check NaN imputation in the specified null columns ---
    if null_cols:
        cols_still_null = [
            col for col in null_cols
            if col in df_fixed.columns and df_fixed[col].isnull().sum() > 0
        ]
        if not cols_still_null:
            reward += 0.4
            log.append(f"✅ +0.4 All null columns imputed: {null_cols}")
        else:
            log.append(f"❌ +0.0 These columns still have NaNs: {cols_still_null}")
    else:
        # Fallback: check globally
        if df_fixed.isnull().sum().sum() == 0:
            reward += 0.4
            log.append("✅ +0.4 No NaN values remain in DataFrame.")
        else:
            log.append("❌ +0.0 NaN values still present in DataFrame.")

    # --- Step 3: Check categorical encoding + model trainability ---
    try:
        if target_col not in df_fixed.columns:
            log.append(f"❌ +0.0 Target column '{target_col}' missing from fixed DataFrame.")
            return round(min(reward, 1.0), 2), "\n".join(log)

        X = df_fixed.drop(columns=[target_col])
        y = df_fixed[target_col]

        # Check that all object columns are gone (encoded)
        remaining_cats = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if remaining_cats:
            log.append(f"❌ +0.0 Unencoded categorical columns remain: {remaining_cats}")
            return round(min(reward, 1.0), 2), "\n".join(log)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))

        reward += 0.4
        log.append(f"✅ +0.4 Model trained successfully. Accuracy: {acc:.4f}")

    except Exception as exc:
        log.append(f"❌ +0.0 Model training failed: {exc}")

    return round(min(reward, 1.0), 2), "\n".join(log)


# ====================================================================== #
#  TASK 3 GRADER — The Skewed Forecasting Nightmare                      #
# ====================================================================== #

def grade_skewed_task(
    agent_code: str,
    df: pd.DataFrame,
    target_col: str,
    baseline_rmse: float,
    rmse_threshold: float,
) -> tuple[float, str]:
    """
    Grades the agent's fix for the heavily skewed regression task.

    Reward breakdown:
        +0.4  Log transform (np.log1p) applied to target in injected code (AST).
        +0.3  Inverse transform (np.expm1) applied to predictions (AST).
        +0.3  Final RMSE (on original scale) drops below rmse_threshold.

    Args:
        agent_code      : Raw code string from the LLM.
        df              : The original broken DataFrame.
        target_col      : Name of the skewed target column.
        baseline_rmse   : RMSE of the unmodified linear model (captured at reset()).
        rmse_threshold  : RMSE value the agent must beat to earn the final reward.
    """
    reward = 0.0
    log = []

    # --- AST checks first (don't need execution to verify these) ---
    try:
        tree = ast.parse(agent_code)
        node_calls = [
            node.func.attr if isinstance(node.func, ast.Attribute) else
            (node.func.id if isinstance(node.func, ast.Name) else "")
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and hasattr(node, "func")
        ]

        if "log1p" in node_calls:
            reward += 0.4
            log.append("✅ +0.4 np.log1p detected — log transform applied to target.")
        else:
            log.append("❌ +0.0 np.log1p not found in agent code.")

        if "expm1" in node_calls:
            reward += 0.3
            log.append("✅ +0.3 np.expm1 detected — inverse transform applied to predictions.")
        else:
            log.append("❌ +0.0 np.expm1 not found — predictions not inverse-transformed.")

    except SyntaxError:
        log.append("❌ +0.0 AST parse failed — agent code has syntax errors.")
        return round(reward, 2), "\n".join(log)

    # --- Execute the code and measure RMSE on original scale ---
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    context = {
        "df": df.copy(),
        "X_train": X_train.copy(),
        "X_test": X_test.copy(),
        "y_train": y_train.copy(),
        "y_test": y_test.copy(),
        "pd": pd,
        "np": np,
    }

    success, output, namespace = execute_agent_code(agent_code, context)

    if not success:
        log.append(f"❌ +0.0 Code execution failed:\n{output}")
        return round(min(reward, 1.0), 2), "\n".join(log)

    try:
        predictions = namespace.get("predictions", namespace.get("y_pred"))
        if predictions is None:
            log.append("❌ +0.0 Agent code did not produce a 'predictions' or 'y_pred' variable.")
            return round(min(reward, 1.0), 2), "\n".join(log)

        # Predictions must be on original scale (after expm1)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        log.append(
            f"📊 Baseline RMSE: {baseline_rmse:.2f} | "
            f"New RMSE: {rmse:.2f} | "
            f"Threshold: {rmse_threshold:.2f}"
        )

        if rmse < rmse_threshold:
            reward += 0.3
            log.append(f"✅ +0.3 RMSE {rmse:.2f} is below threshold {rmse_threshold:.2f}.")
        else:
            log.append(f"❌ +0.0 RMSE {rmse:.2f} did not beat threshold {rmse_threshold:.2f}.")

    except Exception as exc:
        log.append(f"❌ +0.0 Could not compute RMSE: {exc}")

    return round(min(reward, 1.0), 2), "\n".join(log)