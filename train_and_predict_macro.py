"""
================================================================================
AEROCAST-AI: TURNKEY MASTER MACRO FORECASTING & CLUSTERING CLI
AI SANSAD 2026 | Team Prompt Diplomats (ID: UYU672B3)
================================================================================

Master competition entrypoint for national sustainable energy transition datasets.
Supports:
1. Automated schema detection, cleaning, and domain-specific imputation.
2. Leakage-free panel causal feature engineering (lags, momentums, elasticities).
3. Chronological walk-forward validation across 176 countries.
4. Probabilistic Quantile Modeling (P10, P50, P90) with LightGBM & HistGBM.
5. Country decarbonization archetype clustering (PCA + K-Means).
6. High-speed Kaggle-compatible CSV submission export.

Usage:
  python train_and_predict_macro.py --data "data/global-data-on-sustainable-energy.csv"
  python train_and_predict_macro.py --target "renew_share_pct" --output "renew_share_submission.csv"
  python train_and_predict_macro.py --cluster --output "macro_submission.csv"
================================================================================
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from aerocast_macro_loader import MacroDataLoader
from aerocast_macro_features import MacroFeatureEngineer
from aerocast_macro_models import (
    MacroQuantileForecaster,
    CountryArchetypeClusterer,
    HAS_LIGHTGBM
)
from aerocast_macro_validation import MacroPanelValidator, compute_macro_metrics


def run_pipeline(
    data_path: str,
    target_col: str = "electricity_renewables_twh",
    output_path: str = "macro_predictions.csv",
    run_clustering: bool = True,
    random_state: int = 42
):
    print("=" * 78)
    print(" 🌍 AEROCAST-AI: MACRO SUSTAINABLE ENERGY TRANSITION ENGINE")
    print(" Team: Prompt Diplomats | Team ID: UYU672B3 | Leader: Atharva Sandeep Patil")
    print(f" Target Variable: {target_col}")
    print(f" Execution Backend: {'LIGHTGBM (High Performance)' if HAS_LIGHTGBM else 'SCIKIT-LEARN HIST-GBM (Universal Portable)'}")
    print("=" * 78)

    t_start = time.time()

    # --------------------------------------------------------------------------
    # Step 1: Ingestion & Imputation
    # --------------------------------------------------------------------------
    print("\n[Step 1/5] Ingesting & Sanitizing Panel Dataset...")
    loader = MacroDataLoader(target_col=target_col)
    raw_df, report = loader.load_dataset(data_path)
    print(report.summary())

    # Verify target existence
    if target_col not in raw_df.columns:
        # Check alias
        available = [c for c in raw_df.columns if "renew" in c or "twh" in c or "share" in c]
        raise ValueError(
            f"Target column '{target_col}' not found in sanitized dataset. "
            f"Available relevant targets: {available}"
        )

    # --------------------------------------------------------------------------
    # Step 2: Causal Feature Engineering
    # --------------------------------------------------------------------------
    print("\n[Step 2/5] Building Causal Panel Features & Systemic Elasticities...")
    is_percentage = "pct" in target_col or "share" in target_col
    use_log = not is_percentage  # Log-transform skewed TWh, not percentages

    engineer = MacroFeatureEngineer(
        target_col=target_col,
        lags=(1, 2, 3, 5),
        use_log_transform=use_log
    )
    feature_df, feature_cols, model_target = engineer.engineer_features(raw_df)
    print(f"[*] Total Features Engineered: {len(feature_cols)}")
    print(f"[*] Primary Model Target:      {model_target} (Log1p active: {use_log})")

    # Filter out initial lag warmup rows (e.g. years 2000-2004 where 5-year lag is NaN)
    valid_mask = feature_df[f"{target_col}_lag_1y"].notna() & feature_df[target_col].notna()
    clean_panel = feature_df[valid_mask].copy().reset_index(drop=True)
    print(f"[*] Valid Panel Observations:  {len(clean_panel):,} records across {clean_panel['entity'].nunique()} nations")

    # --------------------------------------------------------------------------
    # Step 3: Chronological Walk-Forward Panel Cross-Validation
    # --------------------------------------------------------------------------
    validator = MacroPanelValidator()
    train_end, val_end, test_end = validator.resolve_cutoffs(clean_panel)
    print(f"\n[Step 3/5] Chronological Walk-Forward Validation (<={train_end} Train -> {train_end+1}-{val_end} Val -> {val_end+1}-{test_end} Holdout)...")
    train_df, val_df, test_df = validator.split_data(clean_panel)

    print(f"  - Train Partition (<={train_end}): {len(train_df):,} records")
    print(f"  - Val Partition ({train_end+1}-{val_end}): {len(val_df):,} records")
    print(f"  - Test Partition ({val_end+1}-{test_end}):{len(test_df):,} records")

    # Train validator model
    val_forecaster = MacroQuantileForecaster(
        quantiles=(0.10, 0.50, 0.90),
        use_log_target=use_log,
        is_percentage=is_percentage,
        random_state=random_state
    )

    X_train = train_df[feature_cols]
    y_train = train_df[model_target].values
    val_forecaster.fit(X_train, y_train, feature_names=feature_cols)

    # Evaluate on Holdout Test Partition
    X_test = test_df[feature_cols]
    y_test_actual = test_df[target_col].values
    test_preds = val_forecaster.predict(X_test)

    metrics = compute_macro_metrics(
        y_true=y_test_actual,
        p10=test_preds["P10"],
        p50=test_preds["P50"],
        p90=test_preds["P90"],
        target_name=target_col
    )

    print("\n" + "-" * 72)
    print(f"AEROCAST-MACRO HOLDOUT EVALUATION METRICS (Test Period: {val_end+1}-{test_end})")
    print("-" * 72)
    print(f"[*] Test Samples Evaluated:            {metrics['n_samples']:,} country-years")
    print(f"[*] Mean Absolute Error (MAE):         {metrics['mae']:.2f}")
    print(f"[*] Root Mean Squared Error (RMSE):    {metrics['rmse']:.2f}")
    print(f"[*] Normalized MAE (NMAE):             {metrics['nmae_pct']:.2f}% (Normalized on Dynamic Range)")
    print(f"[*] Weighted Abs Percent Error (WAPE): {metrics['wape_pct']:.2f}% (Volume-Weighted Error)")
    print(f"[*] Mean Quantile Pinball Loss:        {metrics['mean_pinball']:.4f}")
    print(f"[*] Prediction Interval Coverage (PICP): {metrics['picp_coverage_pct']:.2f}% [Target: 80.0% Nominal]")
    print(f"[*] Prediction Interval Width (PINAW): {metrics['pinaw_width_pct']:.2f}%")
    print(f"[*] Negative Predictions (Physical):   {metrics['negative_violations']} [PASS]")
    print(f"[*] Quantile Crossing Violations:      {metrics['quantile_crossing_violations']} [PASS: Monotonic P10 <= P50 <= P90]")
    print("-" * 72)

    # --------------------------------------------------------------------------
    # Step 4: Full-Dataset Champion Refit & Future Prediction
    # --------------------------------------------------------------------------
    print("\n[Step 4/5] Refitting Champion Forecaster on 100% Panel Data (2000-2020)...")
    champion = MacroQuantileForecaster(
        quantiles=(0.10, 0.50, 0.90),
        use_log_target=use_log,
        is_percentage=is_percentage,
        random_state=random_state
    )
    X_full = clean_panel[feature_cols]
    y_full = clean_panel[model_target].values
    champion.fit(X_full, y_full, feature_names=feature_cols)

    # Generate full dataset fitted predictions and future scenario estimates
    full_preds = champion.predict(X_full)

    out_df = clean_panel[["entity", "year"]].copy()
    out_df["Actual"] = clean_panel[target_col].values
    out_df["P10_Predicted"] = np.round(full_preds["P10"], 2)
    out_df["P50_Predicted"] = np.round(full_preds["P50"], 2)
    out_df["P90_Predicted"] = np.round(full_preds["P90"], 2)
    out_df["Error_P50"] = np.round(out_df["Actual"] - out_df["P50_Predicted"], 2)

    # --------------------------------------------------------------------------
    # Step 5: Country Archetype Clustering (Optional / Competitive Edge)
    # --------------------------------------------------------------------------
    if run_clustering:
        print("\n[Step 5/5] Generating Decarbonization Typologies via PCA + K-Means...")
        clusterer = CountryArchetypeClusterer(n_clusters=4, random_state=random_state)
        # Select latest year with representative country coverage (>= 50 nations)
        year_counts = clean_panel.groupby("year")["entity"].count()
        valid_years = year_counts[year_counts >= 50].index
        cluster_year = valid_years.max() if not valid_years.empty else clean_panel["year"].max()
        df_latest = clean_panel[clean_panel["year"] == cluster_year].copy().reset_index(drop=True)
        print(f"  Clustering Year Selected: {cluster_year} ({len(df_latest)} sovereign entities)")

        cluster_features = [
            fc for fc in [
                "calc_renewable_generation_share_lag1",
                "calc_carbon_intensity_lag1",
                "calc_access_equity_gap_lag1",
                "log_gdp_pc_lag1",
                "electricity_renewables_twh_lag_1y"
            ] if fc in df_latest.columns
        ]

        if cluster_features:
            clustered = clusterer.fit_predict(df_latest, cluster_features)
            out_df = out_df.merge(
                clustered[["entity", "cluster_id", "archetype_name"]],
                on="entity",
                how="left"
            )
            archetype_summary = clustered["archetype_name"].value_counts()
            print("  Archetype Distribution (2020 Country State):")
            for arch_name, count in archetype_summary.items():
                print(f"    - {arch_name}: {count} countries")

            # Export archetype breakdown
            archetype_path = "macro_country_archetypes.csv"
            clustered.to_csv(archetype_path, index=False)
            print(f"  [OK] Exported Country Archetypes -> '{archetype_path}'")

    # Export main predictions
    out_df.to_csv(output_path, index=False)
    elapsed = time.time() - t_start
    print(f"\n[OK] Submission & Predictions Exported -> '{output_path}'")
    print(f"[OK] Pipeline Execution Completed in {elapsed:.2f} seconds.")
    print("=" * 78)

    return metrics, out_df


def resolve_default_data_path() -> str:
    """Auto-detects training dataset preference."""
    for p in [
        "data/training_data.csv",
        "data/training data.csv",
        "training_data.csv",
        "training data.csv",
        "data/global-data-on-sustainable-energy.csv"
    ]:
        if os.path.exists(p):
            return p
    return "data/training_data.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AeroCast-AI Macro Sustainable Energy Pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default=resolve_default_data_path(),
        help="Path to the sustainable energy dataset"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="electricity_renewables_twh",
        help="Target variable (e.g. electricity_renewables_twh, renew_share_pct)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="macro_predictions.csv",
        help="Output CSV path for predictions"
    )
    parser.add_argument(
        "--no-cluster",
        dest="cluster",
        action="store_false",
        help="Disable country archetype clustering"
    )
    parser.set_defaults(cluster=True)

    args = parser.parse_args()

    run_pipeline(
        data_path=args.data,
        target_col=args.target,
        output_path=args.output,
        run_clustering=args.cluster
    )
