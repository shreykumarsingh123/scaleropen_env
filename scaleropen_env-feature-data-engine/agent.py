from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional

from config import EnvConfig
from env import SilentDebuggerEnv
from tune import HyperparameterTuner, GenericTuner, TuneResult


_SYSTEM_PROMPT = (
    "You are a senior data scientist and ML debugging expert.\n"
    "Your ONLY job is to write a Python code snippet that fixes a broken ML pipeline.\n"
    "You will be given a Data Health JSON describing the exact problems in the dataset.\n\n"
    "Rules:\n"
    " - Output ONLY valid Python code — no explanations, no markdown prose.\n"
    " - Wrap your code in ```python ... ``` fences.\n"
    " - Use ONLY the variables already in scope (listed below per task).\n"
    " - Do NOT import os, sys, subprocess, or any IO library.\n"
    " - Keep the fix minimal and targeted to the exact issues reported."
)

_TASK_CONTEXT = {
    "scaling": (
        "TASK: KNN Classification — Scale Mismatch Fix\n"
        "==============================================\n"
        "The dataset has one feature with a range 1000x larger than others,\n"
        "destroying Euclidean distance-based KNN classification.\n\n"
        "Variables available in scope:\n"
        "  df        : pd.DataFrame  (full dataset with target column)\n"
        "  X_train   : np.ndarray    (train features, unscaled)\n"
        "  X_test    : np.ndarray    (test features, unscaled)\n"
        "  y_train   : pd.Series     (train labels)\n"
        "  y_test    : pd.Series     (test labels)\n"
        "  pd, np    : libraries\n\n"
        "Your fix MUST:\n"
        "  1. Apply a scaler (StandardScaler / MinMaxScaler / RobustScaler) to X_train and X_test.\n"
        "  2. Store scaled arrays as X_train_scaled and X_test_scaled.\n\n"
        "Example:\n"
        "  from sklearn.preprocessing import StandardScaler\n"
        "  scaler = StandardScaler()\n"
        "  X_train_scaled = scaler.fit_transform(X_train)\n"
        "  X_test_scaled  = scaler.transform(X_test)"
    ),
    "attrition": (
        "TASK: HR Attrition — Missing Values & Categorical Encoding Fix\n"
        "===============================================================\n"
        "The HR dataset has injected NaN values in numeric columns and\n"
        "unencoded categorical string columns that prevent model training.\n\n"
        "Variables available in scope:\n"
        "  df     : pd.DataFrame  (the full HR dataset, modify this directly)\n"
        "  pd, np : libraries\n\n"
        "Your fix MUST:\n"
        "  1. Impute ALL NaN values in numeric columns (use median or mean).\n"
        "  2. One-hot encode ALL object/category dtype columns EXCEPT the target.\n"
        "  3. The final df must have NO NaN values and NO object/category columns.\n\n"
        "Example:\n"
        "  for col in df.select_dtypes(include='number').columns:\n"
        "      df[col] = df[col].fillna(df[col].median())\n"
        "  df = pd.get_dummies(df, columns=[...cat cols...], drop_first=True)"
    ),
    "skewed": (
        "TASK: Regression — Heavy Skew Fix via Log Transform\n"
        "====================================================\n"
        "The regression target follows a log-normal distribution (extreme right skew).\n"
        "A linear model on the raw target produces very high RMSE.\n\n"
        "Variables available in scope:\n"
        "  df        : pd.DataFrame  (full dataset with target column)\n"
        "  X_train   : np.ndarray    (train features)\n"
        "  X_test    : np.ndarray    (test features)\n"
        "  y_train   : pd.Series     (train target, heavily skewed)\n"
        "  y_test    : pd.Series     (test target, heavily skewed)\n"
        "  pd, np    : libraries\n\n"
        "Your fix MUST:\n"
        "  1. Apply np.log1p() to y_train before model fitting.\n"
        "  2. Train a regression model on the log-transformed target.\n"
        "  3. Apply np.expm1() to inverse-transform predictions back to original scale.\n"
        "  4. Store final predictions in a variable named predictions.\n\n"
        "Example:\n"
        "  from sklearn.linear_model import Ridge\n"
        "  y_train_log = np.log1p(y_train)\n"
        "  model = Ridge(alpha=1.0)\n"
        "  model.fit(X_train, y_train_log)\n"
        "  predictions = np.expm1(model.predict(X_test))"
    ),
}


@dataclass
class AgentResult:
    task:         str
    provider:     str
    n_steps:      int
    final_reward: float
    done:         bool
    log:          str
    agent_code:   str
    elapsed_sec:  float

    def summary(self) -> str:
        lines = [
            "",
            "=" * 60,
            f"  AGENT RESULT — Task: {self.task.upper()}",
            "=" * 60,
            f"  Provider      : {self.provider}",
            f"  Steps taken   : {self.n_steps}",
            f"  Final reward  : {self.final_reward:.2f}",
            f"  Done (solved) : {self.done}",
            f"  Elapsed       : {self.elapsed_sec:.1f}s",
            "",
            "  Grader log:",
        ]
        for line in self.log.split("\n"):
            lines.append(f"    {line}")
        lines.append("=" * 60)
        return "\n".join(lines)


class DataDebuggerAgent:

    _DEFAULT_MODELS = {
        "openai": "gpt-4o",
        "gemini": "gemini-1.5-flash",
        "claude": "claude-3-5-haiku-20241022",
    }

    def __init__(
        self,
        provider:      str = "fallback",
        model:         Optional[str] = None,
        api_key:       Optional[str] = None,
        use_tuner:     bool = False,
        n_tune_trials: int = 40,
        temperature:   float = 0.2,
        max_tokens:    int = 1024,
        verbose:       bool = True,
    ):
        valid = ("openai", "gemini", "claude", "fallback")
        if provider not in valid:
            raise ValueError(f"Unknown provider '{provider}'. Choose: {valid}")

        self.provider      = provider
        self.model         = model or self._DEFAULT_MODELS.get(provider, "")
        self.api_key       = api_key
        self.use_tuner     = use_tuner
        self.n_tune_trials = n_tune_trials
        self.temperature   = temperature
        self.max_tokens    = max_tokens
        self.verbose       = verbose

        if self.provider == "openai" and not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        elif self.provider == "gemini" and not self.api_key:
            self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        elif self.provider == "claude" and not self.api_key:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")

        if self.provider != "fallback" and not self.api_key:
            raise ValueError(
                f"No API key for provider '{provider}'. "
                f"Set the env var or pass api_key= explicitly.\n"
                f"  openai → OPENAI_API_KEY\n"
                f"  gemini → GEMINI_API_KEY\n"
                f"  claude → ANTHROPIC_API_KEY"
            )

        self._tune_cache: dict[str, TuneResult] = {}

    def act(self, obs: dict) -> str:
        task = obs["task"]

        tune_hint = ""
        if self.use_tuner:
            tune_hint = self._get_tune_hint(task, obs)

        if self.provider == "fallback":
            return self._fallback_act(task, obs)

        context = (
            self._build_generic_context(obs)
            if task not in _TASK_CONTEXT
            else _TASK_CONTEXT[task]
        )
        prompt = self._build_prompt(obs, context, tune_hint)

        if self.verbose:
            print(f"\n{'─' * 55}")
            print(f"  📨 Sending prompt to {self.provider} ({self.model})...")
            print(f"{'─' * 55}")

        raw_response = self._call_llm(prompt)

        if self.verbose:
            print(f"  📩 Response received ({len(raw_response)} chars)")

        code = self._extract_code(raw_response)

        if self.verbose:
            print(f"\n  🔧 Extracted code ({len(code.splitlines())} lines):\n")
            for line in code.splitlines():
                print(f"    {line}")

        return code

    @staticmethod
    def _build_generic_context(obs: dict) -> str:
        health = json.loads(obs["state"])
        meta   = obs.get("meta", {})
        target = meta.get("target_col", "target")

        issues = []

        mv = health.get("missing_values") or health.get("null_cols")
        if mv:
            cols = list(mv.keys()) if isinstance(mv, dict) else mv
            issues.append(
                f"MISSING VALUES in {cols}. "
                f"Fix: df[col].fillna(df[col].median()) for numeric cols."
            )

        cats = health.get("unencoded_categoricals") or health.get("cat_cols")
        if cats:
            issues.append(
                f"UNENCODED CATEGORICALS: {cats}. "
                f"Fix: pd.get_dummies(df, columns={cats}, drop_first=True)."
            )

        sm = health.get("scale_mismatch")
        if sm:
            mismatched = list(sm.keys()) if isinstance(sm, dict) else sm
            issues.append(
                f"SCALE MISMATCH in {mismatched}. "
                f"Fix: StandardScaler on X_train / X_test. "
                f"Store as X_train_scaled, X_test_scaled."
            )

        sk = health.get("high_skewness") or health.get("skewed_cols")
        if sk:
            skewed = list(sk.keys()) if isinstance(sk, dict) else sk
            issues.append(
                f"HIGH SKEWNESS in {skewed}. "
                f"If target is skewed: y_train_log = np.log1p(y_train), "
                f"predictions = np.expm1(model.predict(X_test))."
            )

        outs = health.get("outliers")
        if outs:
            issues.append(
                f"OUTLIERS in {list(outs.keys())}. "
                f"Fix: clip with df[col].clip(lower=Q1-1.5*IQR, upper=Q3+1.5*IQR)."
            )

        imb = health.get("class_imbalance")
        if imb:
            issues.append(
                f"CLASS IMBALANCE detected (ratio={imb}). "
                f"Fix: use class_weight='balanced' in the classifier."
            )

        issues_str = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(issues)) if issues else "  No critical issues detected."

        return (
            f"TASK: Generic ML Pipeline Fix\n"
            f"Target column: {target}\n"
            f"Data shape: {health.get('shape', 'unknown')}\n\n"
            f"DETECTED ISSUES:\n{issues_str}\n\n"
            f"Variables in scope:\n"
            f"  df       : pd.DataFrame  (full dataset including target)\n"
            f"  X_train  : np.ndarray    (train features)\n"
            f"  X_test   : np.ndarray    (test features)\n"
            f"  y_train  : pd.Series     (train target)\n"
            f"  y_test   : pd.Series     (test target)\n"
            f"  pd, np   : libraries\n\n"
            f"Fix ALL detected issues in the correct order:\n"
            f"  1. Impute nulls on df  →  2. Encode categoricals on df  "
            f"→  3. Scale features (if scale mismatch)  →  4. Log-transform target (if skewed)"
        )

    def _build_prompt(self, obs: dict, context: str, tune_hint: str = "") -> str:
        state     = json.loads(obs["state"])
        meta      = obs.get("meta", {})
        state_str = json.dumps(state, indent=2)
        meta_lines = "\n".join(
            f"  {k}: {v}" for k, v in meta.items()
            if k not in ("null_cols", "cat_cols")
        )
        tune_section = (
            f"\n\nOPTUNA HINT (suggested hyperparameters):\n{tune_hint}\n"
            if tune_hint else ""
        )
        return (
            f"{_SYSTEM_PROMPT}\n\n"
            f"{'=' * 55}\n"
            f"{context}\n\n"
            f"DATA HEALTH JSON:\n"
            f"```json\n{state_str}\n```\n\n"
            f"TASK METADATA:\n{meta_lines}"
            f"{tune_section}\n\n"
            f"Now write the Python fix code:\n"
        )

    def _call_llm(self, prompt: str) -> str:
        if self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "gemini":
            return self._call_gemini(prompt)
        elif self.provider == "claude":
            return self._call_claude(prompt)
        raise RuntimeError(f"Unexpected provider: {self.provider}")

    def _call_openai(self, prompt: str) -> str:
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI SDK not installed. Run: pip install openai")

        client = openai.OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def _call_gemini(self, prompt: str) -> str:
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError:
            raise ImportError("Google GenAI SDK not installed. Run: pip install google-genai")

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        return response.text or ""

    def _call_claude(self, prompt: str) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic SDK not installed. Run: pip install anthropic")

        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.content[0].text if response.content else ""

    @staticmethod
    def _extract_code(text: str) -> str:
        patterns = [
            r"```[Pp]ython\s*\r?\n(.*?)```",
            r"```\s*\r?\n(.*?)```",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        return re.sub(r"```(?:\w+)?", "", text).replace("```", "").strip()

    def _fallback_act(self, task: str, obs: dict) -> str:
        if self.verbose:
            print(f"\n  🔧 Fallback mode: running Optuna tuner for task='{task}'...")

        result = self._get_or_run_tuner(task, obs)

        if self.verbose:
            print(f"  ✅ Tuner done. Best score: {result.best_score:.4f}")
            print(f"  📋 Best params: {result.best_params}")

        return result.agent_code

    def _get_tune_hint(self, task: str, obs: dict) -> str:
        result = self._get_or_run_tuner(task, obs)
        return "\n".join(f"  {k} = {v}" for k, v in result.best_params.items())

    def _get_or_run_tuner(self, task: str, obs: dict) -> TuneResult:
        if task not in self._tune_cache:
            tuner = HyperparameterTuner(
                task=task,
                n_trials=self.n_tune_trials,
                cv_folds=5,
                verbose=self.verbose,
            )
            self._tune_cache[task] = tuner.tune()
        return self._tune_cache[task]


def run_loop(
    task:          str,
    provider:      str = "fallback",
    model:         Optional[str] = None,
    api_key:       Optional[str] = None,
    max_steps:     int = 3,
    use_tuner:     bool = False,
    n_tune_trials: int = 40,
    temperature:   float = 0.2,
    verbose:       bool = True,
) -> AgentResult:
    divider = "=" * 60

    if verbose:
        print(f"\n{divider}")
        print(f"  🚀 Starting episode  |  task={task}  |  provider={provider}")
        print(divider)

    t_start = time.time()

    env = SilentDebuggerEnv(EnvConfig(task=task))
    agent = DataDebuggerAgent(
        provider=provider,
        model=model,
        api_key=api_key,
        use_tuner=use_tuner,
        n_tune_trials=n_tune_trials,
        temperature=temperature,
        verbose=verbose,
    )

    obs = env.reset()

    if verbose:
        state_preview = json.loads(obs["state"])
        print(f"\n📋 Initial Data Health State:")
        for k, v in state_preview.items():
            print(f"   {k}: {v}")
        print()

    agent_code   = ""
    final_reward = 0.0
    final_log    = ""

    for step in range(1, max_steps + 1):
        if verbose:
            print(f"\n{'─' * 60}")
            print(f"  Step {step}/{max_steps}")
            print(f"{'─' * 60}")

        agent_code = agent.act(obs)
        obs        = env.step(agent_code)

        final_reward = obs["reward"]
        final_log    = obs["log"]

        if verbose:
            print(f"\n📝 Grader Feedback:")
            for line in obs["log"].split("\n"):
                print(f"   {line}")
            print(f"\n🏆 Reward: {final_reward:.2f}  |  Done: {obs['done']}")

        if obs["done"]:
            if verbose:
                print(f"\n✅ Task SOLVED in {step} step(s)!")
            break

    result = AgentResult(
        task=task,
        provider=provider,
        n_steps=step,
        final_reward=final_reward,
        done=obs["done"],
        log=final_log,
        agent_code=agent_code,
        elapsed_sec=time.time() - t_start,
    )

    if verbose:
        print(result.summary())

    return result


def run_all_tasks(
    provider:      str = "fallback",
    model:         Optional[str] = None,
    api_key:       Optional[str] = None,
    max_steps:     int = 3,
    use_tuner:     bool = False,
    n_tune_trials: int = 40,
    verbose:       bool = True,
) -> dict[str, AgentResult]:
    results = {}
    for task in ("scaling", "attrition", "skewed"):
        results[task] = run_loop(
            task=task,
            provider=provider,
            model=model,
            api_key=api_key,
            max_steps=max_steps,
            use_tuner=use_tuner,
            n_tune_trials=n_tune_trials,
            verbose=verbose,
        )

    print(f"\n{'=' * 70}")
    print(f"  {'TASK':<15} {'REWARD':>8} {'DONE':>6} {'STEPS':>6} {'TIME(s)':>9}  PROVIDER")
    print(f"{'=' * 70}")
    for task, r in results.items():
        print(
            f"  {task:<15} {r.final_reward:>7.2f}  "
            f"{'✅' if r.done else '❌':>5}  "
            f"{r.n_steps:>5}  {r.elapsed_sec:>8.1f}  {r.provider}"
        )
    print(f"{'=' * 70}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Silent Data Debugger — LLM Agent\n"
            "Runs the full RL loop: env.reset() -> agent.act() -> env.step()\n\n"
            "Providers:\n"
            "  fallback  No LLM — uses Optuna (tune.py) offline\n"
            "  openai    GPT-4o / GPT-4-turbo  (needs OPENAI_API_KEY)\n"
            "  gemini    Gemini 1.5 Pro/Flash   (needs GEMINI_API_KEY)\n"
            "  claude    Claude 3.5              (needs ANTHROPIC_API_KEY)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task",        choices=["scaling", "attrition", "skewed", "all"], default="all")
    parser.add_argument("--provider",    choices=["openai", "gemini", "claude", "fallback"], default="fallback")
    parser.add_argument("--model",       default=None,  help="Model name override.")
    parser.add_argument("--api-key",     default=None,  help="Explicit API key.")
    parser.add_argument("--max-steps",   type=int,   default=3,    help="Max steps per episode.")
    parser.add_argument("--use-tuner",   action="store_true",      help="Inject Optuna hints into LLM prompt.")
    parser.add_argument("--tune-trials", type=int,   default=40,   help="Optuna trials (default: 40).")
    parser.add_argument("--temperature", type=float, default=0.2,  help="LLM temperature (default: 0.2).")
    parser.add_argument("--quiet",       action="store_true",      help="Suppress verbose output.")
    args = parser.parse_args()

    verbose = not args.quiet

    try:
        if args.task == "all":
            run_all_tasks(
                provider=args.provider,
                model=args.model,
                api_key=args.api_key,
                max_steps=args.max_steps,
                use_tuner=args.use_tuner,
                n_tune_trials=args.tune_trials,
                verbose=verbose,
            )
        else:
            result = run_loop(
                task=args.task,
                provider=args.provider,
                model=args.model,
                api_key=args.api_key,
                max_steps=args.max_steps,
                use_tuner=args.use_tuner,
                n_tune_trials=args.tune_trials,
                temperature=args.temperature,
                verbose=verbose,
            )
            if not verbose:
                print(result.summary())

    except ValueError as e:
        print(f"\n❌ Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted.")
        sys.exit(0)
