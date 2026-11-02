"""
AeroCast-AI Version 1.5 - Synthetic GEFCom 2012 Generator
===========================================================
Generates realistic local test datasets conforming to the GEFCom 2012 Wind
Track schema (train.csv and windforecasts_wf1.csv) in data/gefcom2012/.

Physics Dynamics Simulated:
  - Atmospheric synoptic pressure wave (5-day cycle)
  - Diurnal thermal boundary layer oscillation (24-hour cycle)
  - Autoregressive AR(1) turbulent wind gusts
  - Non-linear aerodynamic cubic power curve with cut-in, rated, and cut-out speeds
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_pipeline.config import (
    DATA_DIR,
    TRAIN_DATA_PATH,
    FORECAST_DATA_PATH,
    TOTAL_DAYS,
    RANDOM_STATE,
    ensure_directories,
)


def generate_gefcom_data(n_days: int = TOTAL_DAYS, seed: int = RANDOM_STATE):
    """
    Generate realistic dummy GEFCom 2012 train.csv and windforecasts_wf1.csv.
    """
    np.random.seed(seed)
    ensure_directories()

    total_hours = n_days * 24
    start_date = pd.Timestamp("2012-01-01 00:00:00")

    print("=" * 75)
    print("AEROCAST-AI v1.5: GENERATING SYNTHETIC GEFCOM 2012 DATASET")
    print(f"Target Directory: {DATA_DIR}")
    print(f"Duration:         {n_days} days ({total_hours} hourly timesteps)")
    print("=" * 75)

    # 1. Atmospheric Wind Dynamics (Time series across 720 hours)
    t = np.arange(total_hours)
    hour_of_day = t % 24

    # Synoptic front cycle (~6 days)
    synoptic_wave = 3.2 * np.sin(2 * np.pi * t / (24 * 6))
    # Diurnal oscillation (peak in afternoon)
    diurnal_wave = 2.4 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)

    # Autoregressive turbulence noise AR(1)
    ar_noise = np.zeros(total_hours)
    noise_innovations = np.random.normal(0, 0.95, size=total_hours)
    for i in range(1, total_hours):
        ar_noise[i] = 0.82 * ar_noise[i - 1] + noise_innovations[i]

    # Combined Wind Speed (0 - 25 m/s)
    base_ws = 8.5
    raw_ws = base_ws + synoptic_wave + diurnal_wave + ar_noise
    ws_actual = np.clip(raw_ws, 0.5, 24.5)

    # Wind Direction (0 - 360 degrees)
    wd_base = (190.0 + 85.0 * np.sin(2 * np.pi * t / (24 * 10)) + np.random.normal(0, 12, size=total_hours)) % 360.0

    # 2. Aerodynamic Cubic Power Curve Simulation (wp1 in [0.0, 1.0])
    # Cut-in: 3.0 m/s, Rated: 12.5 m/s, Cut-out: 25.0 m/s
    v_cutin = 3.0
    v_rated = 12.5
    v_cutout = 25.0

    wp1 = np.zeros(total_hours)
    cubic_denom = (v_rated**3 - v_cutin**3)

    for i in range(total_hours):
        v = ws_actual[i]
        if v < v_cutin or v >= v_cutout:
            power = 0.0
        elif v >= v_rated:
            power = 1.0
        else:
            power = (v**3 - v_cutin**3) / cubic_denom

        # Add turbulence, mechanical conversion losses, and measurement variance
        mech_noise = np.random.normal(0, 0.025)
        wp1[i] = np.clip(power + mech_noise, 0.0, 1.0)

    # 3. Create Weather Forecast Dataset (windforecasts_wf1.csv)
    # Forecasts issued daily at 00:00 for horizons hors=1..24
    forecast_rows = []
    for day in range(n_days):
        issue_time = start_date + pd.Timedelta(days=day)
        for h in range(1, 25):
            idx = day * 24 + (h - 1)
            # Weather forecast contains slight forecast error relative to actual atmosphere
            forecast_error_ws = np.random.normal(0, 0.35 + 0.02 * h)
            forecast_error_wd = np.random.normal(0, 4.0 + 0.2 * h)

            fc_ws = np.clip(ws_actual[idx] + forecast_error_ws, 0.2, 24.8)
            fc_wd = (wd_base[idx] + forecast_error_wd) % 360.0

            forecast_rows.append({
                "date": issue_time.strftime("%Y-%m-%d %H:%M:%S"),
                "hors": h,
                "ws": np.round(fc_ws, 3),
                "wd": np.round(fc_wd, 1),
            })

    forecasts_df = pd.DataFrame(forecast_rows)
    forecasts_df.to_csv(FORECAST_DATA_PATH, index=False)
    print(f"[OK] Generated {len(forecasts_df):,} weather forecast records -> {FORECAST_DATA_PATH}")

    # 4. Create Generation Dataset (train.csv)
    # Valid time matches each hourly observation
    train_rows = []
    for day in range(n_days):
        issue_time = start_date + pd.Timedelta(days=day)
        for h in range(1, 25):
            idx = day * 24 + (h - 1)
            valid_time = issue_time + pd.Timedelta(hours=h)
            train_rows.append({
                "date": valid_time.strftime("%Y-%m-%d %H:%M:%S"),
                "wp1": np.round(wp1[idx], 4),
            })

    train_df = pd.DataFrame(train_rows)
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    print(f"[OK] Generated {len(train_df):,} target power records     -> {TRAIN_DATA_PATH}")

    print("\nDataset Summary Statistics:")
    print("-" * 50)
    print("train.csv (wp1):")
    print(train_df["wp1"].describe().to_string())
    print("\nwindforecasts_wf1.csv (ws):")
    print(forecasts_df["ws"].describe().to_string())
    print("=" * 75)
    print("Successfully generated dummy GEFCom 2012 dataset for local testing.\n")


if __name__ == "__main__":
    generate_gefcom_data()
