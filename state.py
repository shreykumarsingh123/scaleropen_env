import pandas as pd
import numpy as np
import json

def get_state(df):
    """
    Analyzes a Pandas DataFrame and returns a Data Health JSON string.
    This acts as the 'Observation' for the AI Agent in the OpenEnv loop.
    """
    state_dict = {}

    # 1. Detect Missing Values (Task 1)
    missing_counts = df.isnull().sum()
    missing_dict = missing_counts[missing_counts > 0].to_dict()
    if missing_dict:
        state_dict["missing_values"] = missing_dict

    # 2. Detect Unencoded Categoricals (Task 2)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        state_dict["unencoded_categoricals"] = cat_cols

    # 3. Detect Heavy Skewness (Task 3)
    numeric_cols = df.select_dtypes(include=[np.number])
    if not numeric_cols.empty:
        skewness = numeric_cols.skew().dropna()
        # Threshold for extreme skewness is generally > 3.0
        high_skew = skewness[(skewness > 3.0) | (skewness < -3.0)].to_dict()
        if high_skew:
            state_dict["high_skewness"] = high_skew

    # 4. Detect Scale Mismatches (Task 4)
    if not numeric_cols.empty:
        max_vals = numeric_cols.max().dropna()
        if not max_vals.empty and (max_vals.max() / (max_vals.min() + 1e-5) > 1000):
            state_dict["scale_mismatch_detected"] = True
        else:
            state_dict["scale_mismatch_detected"] = False

    return json.dumps(state_dict, indent=4)