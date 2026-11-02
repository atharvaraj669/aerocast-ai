"""
AeroCast-AI Version 2.0 - Configuration Module
================================================
Centralized path resolution and hyperparameters for the AeroCast-AI ML pipeline.

PATH SAFETY GUARANTEE:
  All paths use pathlib.Path and forward-slash operators (/), never os.sep,
  backslash strings, or hardcoded drive letters. This module is fully
  cross-platform and resolves correctly on Windows, macOS (M-series ARM64),
  and Linux without any modification.

  Mechanism:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
  This anchor is derived at import time from the location of this file itself,
  so the absolute path is always correct regardless of where Python is invoked
  or which OS is running.
"""

from pathlib import Path

# ─── Base Paths (cross-platform, derived at import time) ─────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # d:/.../  or  /Users/.../

# ─── Data Paths ───────────────────────────────────────────────────────────────
DATA_DIR            = PROJECT_ROOT / "data" / "gefcom2012"
TRAIN_DATA_PATH     = DATA_DIR / "train.csv"
FORECAST_DATA_PATH  = DATA_DIR / "windforecasts_wf1.csv"

# ─── Output Paths ─────────────────────────────────────────────────────────────
OUTPUT_DIR          = PROJECT_ROOT / "ml_pipeline" / "output"
MODELS_DIR          = OUTPUT_DIR / "models"
PREDICTIONS_PATH    = OUTPUT_DIR / "predictions.csv"

# ─── Pipeline Parameters ──────────────────────────────────────────────────────
RANDOM_STATE = 42
TOTAL_DAYS   = 30
TEST_DAYS    = 6        # Last 6 days (144 hours) reserved for test evaluation
QUANTILES    = [0.10, 0.50, 0.90]

# ─── LightGBM Quantile Regressor Hyperparameters ──────────────────────────────
# n_jobs=-1 auto-detects core count on both x86-64 and Apple Silicon (M-series).
LGBM_PARAMS = {
    "n_estimators":  120,
    "learning_rate": 0.05,
    "num_leaves":    31,
    "random_state":  RANDOM_STATE,
    "verbose":       -1,
    "n_jobs":        -1,    # safe on macOS M5 — LightGBM uses native ARM64 wheels
}


def ensure_directories() -> None:
    """Create data and output directories if they do not already exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
