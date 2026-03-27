# 🕵️‍♂️ The Silent Data Debugger (OpenEnv)

## 📌 The Concept
Most AI coding assistants (like Copilot) are only *code-aware*—they fix syntax errors and typos. **The Silent Data Debugger** is an Automated Machine Learning (AutoML) agent that is *data-aware*. 

Operating within an OpenEnv reinforcement learning loop, this agent looks at the actual statistical health of a dataset in memory (missing values, extreme skewness, unscaled magnitudes) and automatically injects the correct Python code to fix failing ML pipelines.

## ⚙️ How It Works (The RL Loop)
1. **State:** The environment generates a "Data Health JSON" from a broken dataset (e.g., `{"null_values": 50, "skewness": 4.2}`).
2. **Agent:** An LLM reads the state and script, identifies the data science flaw, and writes the specific code to fix it (e.g., injecting `StandardScaler` or `np.log1p`).
3. **Reward:** The environment executes the injected code. If the model's RMSE or Accuracy improves beyond the baseline, the agent receives a reward.

## 🗂️ The 3 Tasks (Curriculum)
* **Task 1 (Easy):** The Scaling Failure (KNN scale mismatch)
* **Task 2 (Medium):** The Attrition Trap (Unencoded categoricals & NaNs)
* **Task 3 (Hard):** The Skewed Forecasting Nightmare (Log transforms for heavy skew)

## 💻 Tech Stack
* **Core Logic:** Python, Pandas, NumPy, Scikit-learn
* **Environment:** OpenEnv Specification
* **Agent:** LLM API
* **Deployment:** Docker & Gradio