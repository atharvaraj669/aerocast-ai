"""
AeroCast-AI Version 2.0 - Feature Engineering Module
=====================================================
Implements safe, leakage-free causal feature engineering:
  - Phase 1: 24-hour Autoregressive Target Lag (target_lag_24h)
  - Missing History Flag (missing_history_flag)
  - Atmospheric wind vector decomposition (u, v)
  - Kinetic energy flux (ws_cubed)
  - Diurnal cyclical temporal harmonics (sin_hour, cos_hour)
"""

from typing import List
import numpy as np
import pandas as pd

# Official Version 2.0 Feature Columns List
FEATURE_COLS: List[str] = [
    "ws",
    "wd",
    "hors",
    "u",
    "v",
    "ws_cubed",
    "hour",
    "sin_hour",
    "cos_hour",
    "day_of_week",
    "target_lag_24h",
    "missing_history_flag",
]


def engineer_features(df: pd.DataFrame, target_col: str = "wp1") -> pd.DataFrame:
    """
    Engineer physical, temporal, and autoregressive lag features without data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing valid_time, ws, wd, hors, and target_power (or wp1).
    target_col : str, default 'wp1'
        Name of the target generation column.

    Returns
    -------
    pd.DataFrame
        Dataframe with engineered features, sorted chronologically.
    """
    df_out = df.copy()

    # Ensure target_power column is accessible
    if "target_power" not in df_out.columns:
        if target_col in df_out.columns:
            df_out["target_power"] = df_out[target_col]
        elif "wp1" in df_out.columns:
            df_out["target_power"] = df_out["wp1"]
        else:
            raise KeyError(f"Target column '{target_col}' or 'target_power' not found in DataFrame.")

    # CRITICAL LEAKAGE RULE: Explicitly sort chronologically by valid_time before any shifting
    if "valid_time" in df_out.columns:
        df_out["valid_time"] = pd.to_datetime(df_out["valid_time"])
        df_out = df_out.sort_values("valid_time", kind="mergesort").reset_index(drop=True)

    # 1. Atmospheric Wind Dynamics
    # Wind vector components (u: eastward, v: northward)
    wd_rad = np.deg2rad(df_out["wd"])
    df_out["u"] = -df_out["ws"] * np.sin(wd_rad)
    df_out["v"] = -df_out["ws"] * np.cos(wd_rad)

    # Kinetic Energy Flux (power proportional to v^3)
    df_out["ws_cubed"] = df_out["ws"] ** 3

    # 2. Diurnal & Seasonal Temporal Harmonics
    hours = df_out["valid_time"].dt.hour
    df_out["hour"] = hours
    df_out["sin_hour"] = np.sin(2 * np.pi * hours / 24.0)
    df_out["cos_hour"] = np.cos(2 * np.pi * hours / 24.0)

    try:
        df_out["day_of_week"] = df_out["valid_time"].dt.day_of_week
    except AttributeError:
        df_out["day_of_week"] = df_out["valid_time"].dt.dayofweek

    # 3. PHASE 1: Safe Autoregressive Target Lag (Exactly 24h Prior)
    # 1 row = 1 hour, shift by 24 rows
    raw_lag_24h = df_out["target_power"].shift(24)

    # DO NOT use .bfill() or .fillna(method='bfill')
    # If history does not exist for the first 24 hours:
    # Fill with 0.0 and create binary indicator missing_history_flag (1 if missing, 0 if present)
    missing_history = raw_lag_24h.isna()
    df_out["missing_history_flag"] = missing_history.astype(int)
    df_out["target_lag_24h"] = raw_lag_24h.fillna(0.0)

    return df_out
