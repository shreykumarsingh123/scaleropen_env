# The System Workflow (How the code runs)

**1. Initialization ➡️ 2. Observation ➡️ 3. Action ➡️ 4. Evaluation**

Here is exactly what each step does and the technologies you will use to build it:

### Step 1: Initialization (env.py)
* **Description:** The system loads a broken Python ML script and a messy CSV dataset into a virtual sandbox.
* **Technology:** Python, Pandas (to load the CSV), `subprocess` or `exec()` (to run the script safely).

### Step 2: Observation (state.py)
* **Description:** The system scans the dataset and generates the "Data Health JSON" (e.g., counting nulls, calculating skewness). This is the "State" that gets sent to the AI.
* **Technology:** Pandas (`.describe()`, `.isna()`, `.skew()`), Python `json` library.

### Step 3: Action (agent.py)
* **Description:** The LLM reads the Data Health JSON and the broken script, figures out the fix, and outputs a specific Python code snippet (e.g., `StandardScaler()`).
* **Technology:** LLM API (OpenAI, Gemini, or Claude), Python `re` (Regex to extract the code from the LLM's text response).

### Step 4: Evaluation (evaluate.py)
* **Description:** The system injects the LLM's code into the script, runs it, and checks if the machine learning metric (like RMSE or Accuracy) improved. If it did, it assigns a Reward score (0.0 to 1.0).
* **Technology:** Scikit-learn (to calculate metrics like `mean_squared_error` or `accuracy_score`), Python `ast` (to verify the agent actually wrote the correct functions).