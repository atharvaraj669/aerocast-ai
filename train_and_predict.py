"""
================================================================================
AEROCAST-AI: TURNKEY MASTER RUNNER & COMPETITION CLI (REPAIRED)
AI Sansad 2026 | Department of AI & ML, Bharati Vidyapeeth's College of Engineering
Team: Prompt Diplomats | Team ID: UYU672B3 | Leader: Atharva Sandeep Patil
================================================================================

Assigned Problem Statement:
"Forecast hourly wind power output using historical wind-farm generation and available wind forecasts."

Core Research Contribution:
"Physics-Informed AI with Probabilistic Bounds"

EXECUTION COMMANDS:
1. Dataset Inspection & Audit:
       python train_and_predict.py --train data.csv --inspect

2. Competition Fast Mode (Sub-Minute P10/P50/P90 Submission):
       python train_and_predict.py --train train.csv --test test.csv --mode fast

3. Full Experiment Mode (Multi-Model Benchmark, Rolling-Origin CV & Ensembling):
       python train_and_predict.py --train train.csv --test test.csv --mode full

4. Synthetic Benchmark Demo (Explicit demonstration option only):
       python train_and_predict.py --demo
================================================================================
"""

import os
import sys
import time
import argparse
from typing import Optional, Tuple, Dict, List, Any
import numpy as np
import pandas as pd

from aerocast_physics import TurbineSpecs
from aerocast_loader import AdaptiveDataLoader, DatasetConfig
from aerocast_engine import AeroCastAIEngine
from aerocast_models import PhysicsResidualForecaster, DirectQuantileForecaster, PersistenceBaseline


def generate_realistic_benchmark(n_hours: int = 4000, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Explicit benchmark generator for synthetic demonstration and testing ONLY.
    Simulates diurnal breeze cycles, atmospheric turbulence, and high-wind curtailments.
    """
    np.random.seed(seed)
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=n_hours, freq="h")

    # Diurnal cycle + Weibull wind speed
    diurnal = 2.8 * np.sin(2 * np.pi * (timestamps.hour - 3) / 24.0)
    try:
        doy = timestamps.day_of_year
    except AttributeError:
        doy = timestamps.dayofyear
    seasonal = 1.5 * np.cos(2 * np.pi * doy / 365.25)
    weibull_base = np.random.weibull(2.15, n_hours) * 5.8
    raw_ws = np.clip(weibull_base + diurnal + seasonal, 0.4, 28.5)

    direction = (np.random.normal(210, 45, n_hours) + np.sin(np.arange(n_hours) / 100.0) * 40.0) % 360.0
    temperature = 18.0 + 9.0 * np.sin(2 * np.pi * (timestamps.hour - 9) / 24.0) + np.random.normal(0, 1.2, n_hours)
    pressure = 1013.25 + 6.0 * np.cos(2 * np.pi * doy / 365.25) + np.random.normal(0, 2.5, n_hours)

    # Synthetic 100 MW farm baseline
    v_hub = raw_ws * ((120.0 / 10.0) ** 0.143)
    p_ideal = np.zeros(n_hours)
    ramp_mask = (v_hub >= 3.0) & (v_hub <= 12.5)
    plateau_mask = (v_hub > 12.5) & (v_hub <= 25.0)

    ratio = (v_hub[ramp_mask] - 3.0) / (12.5 - 3.0)
    p_ideal[ramp_mask] = (ratio ** 3) * 100.0
    p_ideal[plateau_mask] = 100.0

    p_actual = p_ideal * (1.0 - np.random.uniform(0.02, 0.08, n_hours)) + np.random.normal(0, 2.0, n_hours)
    p_actual = np.clip(p_actual, 0.0, 100.0)

    # Realistic curtailment: ~2% high-wind zero output
    curtailment_indices = np.random.choice(np.where(v_hub > 7.0)[0], size=int(n_hours * 0.02), replace=False)
    p_actual[curtailment_indices] = 0.0

    df_full = pd.DataFrame({
        "timestamp": timestamps,
        "wind_speed_10m": np.round(raw_ws, 2),
        "wind_direction_deg": np.round(direction, 1),
        "temperature_c": np.round(temperature, 1),
        "surface_pressure_hpa": np.round(pressure, 1),
        "active_power_mw": np.round(p_actual, 2)
    })

    split_pt = int(n_hours * 0.80)
    df_train = df_full.iloc[:split_pt].copy().reset_index(drop=True)
    df_test = df_full.iloc[split_pt:].copy().reset_index(drop=True)
    return df_train, df_test


def run_pipeline(
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    output_path: str = "submission.csv",
    mode: str = "fast",
    inspect_only: bool = False,
    is_demo: bool = False,
    config: Optional[DatasetConfig] = None,
    col_p10: str = "P10",
    col_p50: str = "P50_optimal",
    col_p90: str = "P90"
):
    start_total = time.time()
    print("=" * 75)
    print("AI SANSAD 2026 - DEPARTMENT OF AI & ML, BVCOE DELHI")
    print("AEROCAST-AI: PHYSICS-INFORMED HYBRID PROBABILISTIC FORECASTER")
    print("Team: Prompt Diplomats | Team ID: UYU672B3 | Leader: Atharva Sandeep Patil")
    print("=" * 75)

    config = config or DatasetConfig()
    loader = AdaptiveDataLoader(config=config)

    # Step 1: Ingest Datasets (Strictly enforce existence - no silent fallbacks!)
    if is_demo:
        print("[*] Running in explicit SYNTHETIC BENCHMARK DEMO mode (--demo).")
        df_train, df_test = generate_realistic_benchmark(n_hours=4000)
    else:
        if not train_path:
            raise ValueError(
                "No training dataset specified. Provide '--train <path>' or use '--demo' for synthetic benchmark."
            )
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training dataset file does not exist at: '{train_path}'")

        print(f"[*] Ingesting Training Dataset: {train_path}")
        df_train = loader.read_file(train_path)

        if test_path:
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"Test dataset file does not exist at: '{test_path}'")
            print(f"[*] Ingesting Test Dataset: {test_path}")
            df_test = loader.read_file(test_path)
        else:
            print("[*] No test dataset provided. Using 20% holdout split of training data for evaluation.")
            split_idx = int(len(df_train) * 0.80)
            df_test = df_train.iloc[split_idx:].copy().reset_index(drop=True)
            df_train = df_train.iloc[:split_idx].copy().reset_index(drop=True)

    # Step 2: Inspection Mode
    if inspect_only:
        report = loader.inspect_dataset(df_train)
        loader.print_inspection_report(report)
        print("\n[OK] Inspection complete. To train models, re-run without '--inspect'.")
        return

    # Step 3: Configure Turbine Specifications
    # Unknown specs remain None - strictly NO default 100 MW or 120m assumptions!
    rated_capacity = config.rated_power_mw
    specs = TurbineSpecs(
        rated_power_mw=config.rated_power_mw,
        hub_height_m=config.hub_height_m,
        mast_height_m=config.mast_height_m,
        v_cut_in_ms=config.v_cut_in_ms,
        v_rated_ms=config.v_rated_ms,
        v_cut_out_ms=config.v_cut_out_ms,
        shear_alpha=config.shear_alpha
    )

    # In synthetic demo mode only, populate synthetic parameters if unsupplied
    if is_demo:
        if specs.rated_power_mw is None:
            specs.rated_power_mw = 100.0
            rated_capacity = 100.0
        if specs.hub_height_m is None:
            specs.hub_height_m = 120.0
        if specs.mast_height_m is None:
            specs.mast_height_m = 10.0
        if specs.v_cut_in_ms is None:
            specs.v_cut_in_ms = 3.0
        if specs.v_rated_ms is None:
            specs.v_rated_ms = 12.5
        if specs.v_cut_out_ms is None:
            specs.v_cut_out_ms = 25.0
        if specs.shear_alpha is None:
            specs.shear_alpha = 0.143

    # Step 4: Preprocess and Activate Physics
    engine = AeroCastAIEngine(config=config, specs=specs, random_state=42)
    df_train_proc, col_map = engine.preprocess_dataset(df_train, require_confident_target=True)

    target_col = col_map.get("target_power")
    if not target_col or target_col not in df_train_proc:
        raise ValueError(
            f"Target power column could not be identified with certainty. "
            f"Available columns: {list(df_train.columns)}. Specify via '--target-col <name>'."
        )

    # Chronological Validation Split (80% Train, 20% Tuning Val)
    split_idx = int(len(df_train_proc) * 0.80)
    train_fold = df_train_proc.iloc[:split_idx].copy().reset_index(drop=True)
    val_fold = df_train_proc.iloc[split_idx:].copy().reset_index(drop=True)

    # Step 5: Execute Selected Mode
    if mode.lower() == "fast":
        train_res = engine.train_fast_mode(train_fold, val_fold, rated_capacity)
    else:
        train_res = engine.train_full_experiment_mode(train_fold, val_fold, rated_capacity)

        # Run Chronological Rolling-Origin Cross-Validation
        print("\n[+] Executing Chronological Rolling-Origin Walk-Forward CV (3 Expanding Folds)...")

        def cv_fn(X_tr, y_tr, X_v):
            if engine.active_path == "PATH_1_RESIDUAL" and "theoretical_power_physics" in df_train_proc:
                m = PhysicsResidualForecaster(base_engine="lightgbm", rated_power_mw=rated_capacity, random_state=engine.random_state)
                p_phys_tr = df_train_proc.loc[X_tr.index, "theoretical_power_physics"]
                p_phys_v = df_train_proc.loc[X_v.index, "theoretical_power_physics"]
                m.fit(X_tr, y_tr, p_phys_tr)
                return m.predict(X_v, p_phys_v)
            else:
                m = DirectQuantileForecaster(base_engine="lightgbm", rated_power_mw=rated_capacity, random_state=engine.random_state)
                m.fit(X_tr, y_tr)
                return m.predict(X_v)

        pb_base = PersistenceBaseline(diurnal_lag=engine.config.forecast_horizon_hours)
        def base_fn(y_tr, n):
            pb_base.fit(y_tr.values)
            return pb_base.predict(y_tr.values, n)

        cv_res = engine.validator.evaluate_model_rolling(
            df_train_proc,
            engine.feature_columns,
            target_col,
            rated_capacity,
            train_and_predict_fn=cv_fn,
            baseline_predict_fn=base_fn
        )
        engine.validator.print_validation_report(cv_res, model_name=train_res["champion_name"])

    # Step 6: Refit Champion on 100% of Training Data Before Test Prediction
    engine.refit_champion_on_full_training(df_train_proc, rated_capacity)

    # Step 7: Predict Test Dataset and Export Submission
    submission_df = engine.predict_test(
        df_test,
        rated_capacity=rated_capacity,
        historical_df=df_train_proc,
        col_p10=col_p10,
        col_p50=col_p50,
        col_p90=col_p90
    )
    submission_df.to_csv(output_path, index=False)

    total_time = time.time() - start_total
    print("\n" + "=" * 75)
    print(f"[OK] SUBMISSION EXPORTED: '{output_path}'")
    print(f"[OK] ROWS: {len(submission_df):,} | COLUMNS: {list(submission_df.columns)}")
    print(f"[OK] SAMPLE PREDICTIONS (First 5 Rows):")
    print(submission_df.head(5).to_string(index=False))
    print(f"[OK] TOTAL END-TO-END EXECUTION TIME: {total_time:.2f} seconds")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(
        description="AeroCast-AI: Physics-Informed Wind Power Forecasting Runner"
    )
    parser.add_argument("--train", type=str, default=None, help="Path to training dataset (CSV, TSV, Parquet, Excel)")
    parser.add_argument("--test", type=str, default=None, help="Path to test dataset for final predictions")
    parser.add_argument("--output", type=str, default="submission.csv", help="Path for output submission CSV")
    parser.add_argument("--mode", type=str, choices=["fast", "full"], default="fast", help="Execution mode ('fast' or 'full')")
    parser.add_argument("--inspect", action="store_true", help="Inspect dataset schema and audit curtailment without training")
    parser.add_argument("--demo", action="store_true", help="Run explicit self-contained synthetic benchmark demo")

    # Explicit schema overrides
    parser.add_argument("--target-col", type=str, default=None, help="Override target power column name")
    parser.add_argument("--ws-col", type=str, default=None, help="Override wind speed column name")
    parser.add_argument("--wd-col", type=str, default=None, help="Override wind direction column name")
    parser.add_argument("--temp-col", type=str, default=None, help="Override ambient temperature column name")
    parser.add_argument("--press-col", type=str, default=None, help="Override barometric pressure column name")
    parser.add_argument("--farm-col", type=str, default=None, help="Override farm ID column name")
    parser.add_argument("--turbine-col", type=str, default=None, help="Override turbine ID column name")
    parser.add_argument("--status-col", type=str, default=None, help="Override curtailment/status column name")

    # Explicit turbine specs (only if verified)
    parser.add_argument("--rated-mw", type=float, default=None, help="Verified turbine / farm rated capacity in MW")
    parser.add_argument("--hub-height", type=float, default=None, help="Verified rotor hub height in meters")
    parser.add_argument("--mast-height", type=float, default=None, help="Verified weather mast height in meters")
    parser.add_argument("--cut-in", type=float, default=None, help="Verified cut-in wind speed in m/s")
    parser.add_argument("--rated-speed", type=float, default=None, help="Verified rated wind speed in m/s")
    parser.add_argument("--cut-out", type=float, default=None, help="Verified cut-out wind speed in m/s")
    parser.add_argument("--shear-alpha", type=float, default=None, help="Verified wind shear alpha exponent")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon in hours (default: 24)")

    # Configurable submission column names
    parser.add_argument("--col-p10", type=str, default="P10", help="Column name for P10 prediction in submission")
    parser.add_argument("--col-p50", type=str, default="P50_optimal", help="Column name for P50 prediction in submission")
    parser.add_argument("--col-p90", type=str, default="P90", help="Column name for P90 prediction in submission")

    args = parser.parse_args()

    cfg = DatasetConfig(
        target_col=args.target_col,
        wind_speed_col=args.ws_col,
        wind_direction_col=args.wd_col,
        temperature_col=args.temp_col,
        pressure_col=args.press_col,
        farm_id_col=args.farm_col,
        turbine_id_col=args.turbine_col,
        curtailment_col=args.status_col,
        rated_power_mw=args.rated_mw,
        hub_height_m=args.hub_height,
        mast_height_m=args.mast_height,
        v_cut_in_ms=args.cut_in,
        v_rated_ms=args.rated_speed,
        v_cut_out_ms=args.cut_out,
        shear_alpha=args.shear_alpha,
        forecast_horizon_hours=args.horizon
    )

    run_pipeline(
        train_path=args.train,
        test_path=args.test,
        output_path=args.output,
        mode=args.mode,
        inspect_only=args.inspect,
        is_demo=args.demo,
        config=cfg,
        col_p10=args.col_p10,
        col_p50=args.col_p50,
        col_p90=args.col_p90
    )


if __name__ == "__main__":
    main()
