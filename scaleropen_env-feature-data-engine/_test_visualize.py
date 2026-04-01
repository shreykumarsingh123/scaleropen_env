"""
Test runner for visualize.py.
Feeds real env data into each figure-builder and saves HTML outputs.
Run: python _test_visualize.py
"""

import os
import sys
import numpy as np
import pandas as pd

print("=" * 60)
print("  VISUALIZE.PY — Test Runner")
print("=" * 60)

# ── 1. Import check ──────────────────────────────────────────
print("\n[1/6] Checking imports...")
try:
    import plotly
    from visualize import (
        build_scaling_figures,
        build_attrition_figures,
        build_skewed_figures,
        _reward_gauge,
        _null_heatmap,
        _distribution_plot,
        _class_balance_bar,
        _scale_bar_chart,
    )
    print(f"  ✅ plotly {plotly.__version__}  ·  all functions imported")
except ImportError as e:
    print(f"  ❌ Import failed: {e}")
    sys.exit(1)

OUT_DIR = "viz_output"
os.makedirs(OUT_DIR, exist_ok=True)
saved = []

def save(fig, filename: str, label: str):
    path = os.path.join(OUT_DIR, filename)
    fig.write_html(path)
    saved.append((label, path))
    print(f"  ✅ Saved → {path}")


# ── 2. Shared utility figures ─────────────────────────────────
print("\n[2/6] Testing shared utilities...")

for reward_val in [0.0, 0.5, 0.7, 1.0]:
    fig = _reward_gauge(reward_val)
    save(fig, f"gauge_{int(reward_val*100)}.html", f"Reward gauge ({reward_val})")

df_with_nulls = pd.DataFrame({
    "A": [1.0, None, 3.0, None, 5.0],
    "B": [None, 2.0, 3.0, 4.0, None],
    "C": [1.0, 2.0, 3.0, 4.0, 5.0],
})
save(_null_heatmap(df_with_nulls, "Null Heatmap Test"), "null_heatmap.html", "Null heatmap")

df_dist = pd.DataFrame({"x": np.random.randn(200), "y": np.random.randn(200) + 2})
save(_distribution_plot(df_dist, None, "x"), "dist_before.html",     "Distribution (before only)")
save(_distribution_plot(df_dist, df_dist, "x"), "dist_overlay.html", "Distribution (before+after)")

df_cls = pd.DataFrame({"label": ["Yes"] * 30 + ["No"] * 70})
save(_class_balance_bar(df_cls, "label", "Class Balance"), "class_balance.html", "Class balance bar")

df_scale = pd.DataFrame({
    "small": np.random.randn(100),
    "medium": np.random.randn(100) * 10,
    "huge": np.random.randn(100) * 10000,
})
save(_scale_bar_chart(df_scale, "Scale Mismatch"), "scale_bar.html", "Scale bar chart")


# ── 3. Task 1 — Scaling figures ──────────────────────────────
print("\n[3/6] Testing build_scaling_figures() ...")

from config import EnvConfig
from env import SilentDebuggerEnv

env_scale = SilentDebuggerEnv(EnvConfig(task="scaling"))
obs_scale  = env_scale.reset()
df_broken  = env_scale.df.copy()
meta_scale = env_scale.meta

# build before-only
figs_before = build_scaling_figures(
    df_before=df_broken,
    df_after=None,
    reward=0.0,
    inflated_col=meta_scale.get("inflated_col", "feature_0"),
)
print(f"  before-only keys: {list(figs_before.keys())}")
for name, fig in figs_before.items():
    save(fig, f"scaling_{name}_before.html", f"Scaling {name} (before)")

# run agent to get "after" state
from agent import DataDebuggerAgent
from tune import HyperparameterTuner

tuner = HyperparameterTuner(task="scaling", n_trials=20, verbose=False)
result = tuner.tune()
obs_after = env_scale.step(result.agent_code)

# approximate "after" df by applying scaler manually for vis
from sklearn.preprocessing import StandardScaler
df_after_scale = df_broken.copy()
feat_cols = [c for c in df_after_scale.columns if c != meta_scale["target_col"]]
df_after_scale[feat_cols] = StandardScaler().fit_transform(df_after_scale[feat_cols])

figs_after = build_scaling_figures(
    df_before=df_broken,
    df_after=df_after_scale,
    reward=obs_after["reward"],
    inflated_col=meta_scale.get("inflated_col", "feature_0"),
)
print(f"  after keys: {list(figs_after.keys())}")
for name, fig in figs_after.items():
    save(fig, f"scaling_{name}_after.html", f"Scaling {name} (after)")


# ── 4. Task 2 — Attrition figures ────────────────────────────
print("\n[4/6] Testing build_attrition_figures() ...")

env_attr  = SilentDebuggerEnv(EnvConfig(task="attrition"))
obs_attr  = env_attr.reset()
df_attr   = env_attr.df.copy()
meta_attr = env_attr.meta

figs_attr_before = build_attrition_figures(
    df_before=df_attr,
    df_after=None,
    reward=0.0,
    target_col=meta_attr["target_col"],
    null_cols=meta_attr.get("null_cols", []),
)
print(f"  before-only keys: {list(figs_attr_before.keys())}")
for name, fig in figs_attr_before.items():
    save(fig, f"attrition_{name}_before.html", f"Attrition {name} (before)")

tuner_a = HyperparameterTuner(task="attrition", n_trials=20, verbose=False)
result_a = tuner_a.tune()
obs_attr_after = env_attr.step(result_a.agent_code)

df_attr_fixed = df_attr.copy()
for col in meta_attr.get("null_cols", []):
    if col in df_attr_fixed.columns:
        df_attr_fixed[col] = df_attr_fixed[col].fillna(df_attr_fixed[col].median())

figs_attr_after = build_attrition_figures(
    df_before=df_attr,
    df_after=df_attr_fixed,
    reward=obs_attr_after["reward"],
    target_col=meta_attr["target_col"],
    null_cols=meta_attr.get("null_cols", []),
)
print(f"  after keys: {list(figs_attr_after.keys())}")
for name, fig in figs_attr_after.items():
    save(fig, f"attrition_{name}_after.html", f"Attrition {name} (after)")


# ── 5. Task 3 — Skewed figures ───────────────────────────────
print("\n[5/6] Testing build_skewed_figures() ...")

env_sk  = SilentDebuggerEnv(EnvConfig(task="skewed"))
obs_sk  = env_sk.reset()
df_sk   = env_sk.df.copy()
meta_sk = env_sk.meta

figs_sk_before = build_skewed_figures(
    df_before=df_sk,
    df_after=None,
    reward=0.0,
    target_col=meta_sk["target_col"],
)
print(f"  before-only keys: {list(figs_sk_before.keys())}")
for name, fig in figs_sk_before.items():
    save(fig, f"skewed_{name}_before.html", f"Skewed {name} (before)")

tuner_sk = HyperparameterTuner(task="skewed", n_trials=20, verbose=False)
result_sk = tuner_sk.tune()
obs_sk_after = env_sk.step(result_sk.agent_code)

df_sk_fixed = df_sk.copy()
df_sk_fixed[meta_sk["target_col"]] = np.log1p(df_sk_fixed[meta_sk["target_col"]])

figs_sk_after = build_skewed_figures(
    df_before=df_sk,
    df_after=df_sk_fixed,
    reward=obs_sk_after["reward"],
    target_col=meta_sk["target_col"],
    baseline_rmse=160.0,
    new_rmse=obs_sk_after.get("rmse", 120.0),
)
print(f"  after keys: {list(figs_sk_after.keys())}")
for name, fig in figs_sk_after.items():
    save(fig, f"skewed_{name}_after.html", f"Skewed {name} (after)")


# ── 6. Summary ───────────────────────────────────────────────
print(f"\n[6/6] Summary")
print("=" * 60)
print(f"  {len(saved)} HTML figures saved to ./{OUT_DIR}/")
print()
for label, path in saved:
    print(f"  ✅  {label}")
        
print(f"""
  Open any .html file in your browser to view interactive Plotly charts.
  Example:
    start {OUT_DIR}\\scaling_reward_gauge_after.html
""")
