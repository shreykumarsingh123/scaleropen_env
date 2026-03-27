import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def grade_attrition_task(df_fixed):
    """
    Evaluates the AI's fixed dataframe and returns a reward from 0.0 to 1.0.
    """
    reward = 0.0

    # Check if Attrition exists (the target)
    if 'Attrition' not in df_fixed.columns:
        return 0.0

    # 1. Partial Reward: Check for Missing Values (+0.3)
    # We check the specific column 'Age' which was broken
    if 'Age' in df_fixed.columns and df_fixed['Age'].isnull().sum() == 0:
        reward += 0.3

    # 2. Partial Reward: Check for Categorical Encoding (+0.3)
    # Checks if 'Department' is now numeric or has been One-Hot Encoded away
    if 'Department' not in df_fixed.columns or pd.api.types.is_numeric_dtype(df_fixed['Department']):
        reward += 0.3

    # 3. Final Reward: Model Training (+0.4)
    try:
        X = df_fixed.drop('Attrition', axis=1)
        y = df_fixed['Attrition']

        # Simple split to verify the data is 'trainable'
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # If it reaches here without crashing, the data is fixed!
        reward += 0.4
    except Exception:
        # If it crashes (due to strings or NaNs), reward stays at current level
        pass

    return round(min(reward, 1.0), 2)