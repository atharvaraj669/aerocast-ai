"""
AeroCast-AI Version 2.0 - Probabilistic Quantile Training Pipeline
===================================================================
Upgrades from v1.5:
  - Phase 1: Safe 24h autoregressive target lag (via ml_pipeline/features.py)
  - Phase 2: Persistence Baseline tracked & printed before LightGBM evaluation
  - Phase 3: Physical Governors (zero-negative clamp & quantile monotonicity)
"""

import sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_pipeline.config import (
    TRAIN_DATA_PATH,
    FORECAST_DATA_PATH,
    PREDICTIONS_PATH,
    OUTPUT_DIR,
    LGBM_PARAMS,
    TEST_DAYS,
    ensure_directories,
)
from ml_pipeline.features import engineer_features, FEATURE_COLS


def load_raw_data() -> pd.DataFrame:
    """
    Load and merge train.csv + windforecasts_wf1.csv on valid_time.
    Returns a single chronologically-sorted DataFrame. Feature engineering
    is deferred to engineer_features() so lag shifts see the full dataset.
    """
    if not TRAIN_DATA_PATH.exists() or not FORECAST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing GEFCom 2012 dataset files in '{TRAIN_DATA_PATH.parent}'.\n"
            "Please run: python ml_pipeline/generate_dummy_gefcom.py first."
        )

    print("[*] Loading GEFCom 2012 datasets...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    fc_df = pd.read_csv(FORECAST_DATA_PATH)

    # Parse timestamps
    fc_df["issue_time"] = pd.to_datetime(fc_df["date"])
    fc_df["valid_time"] = fc_df["issue_time"] + pd.to_timedelta(fc_df["hors"], unit="h")
    train_df["valid_time"] = pd.to_datetime(train_df["date"])

    # In case of overlapping horizons, retain the freshest issue_time per valid_time
    fc_sorted = fc_df.sort_values(["valid_time", "issue_time"], ascending=[True, False])
    fc_dedup = fc_sorted.drop_duplicates(subset=["valid_time"], keep="first").copy()

    # Merge and sort chronologically
    merged = pd.merge(
        train_df[["valid_time", "wp1"]],
        fc_dedup[["valid_time", "hors", "ws", "wd"]],
        on="valid_time",
        how="inner",
    ).sort_values("valid_time", kind="mergesort").reset_index(drop=True)

    print(f"[OK] Aligned {len(merged):,} hourly records with matching weather forecasts and generation targets.")
    return merged


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/test split. Adaptive for short datasets.
    """
    total_hours = len(df)
    if total_hours <= (TEST_DAYS + 1) * 24:
        split_idx = int(total_hours * 0.8)
        train_split = df.iloc[:split_idx].copy()
        test_split = df.iloc[split_idx:].copy()
        split_time = test_split["valid_time"].min()
    else:
        split_time = df["valid_time"].max() - pd.Timedelta(days=TEST_DAYS)
        train_split = df[df["valid_time"] < split_time].copy()
        test_split = df[df["valid_time"] >= split_time].copy()

    print(f"[*] Chronological Split at {split_time}:")
    print(f"    - Training Set: {len(train_split):,} hours "
          f"({train_split['valid_time'].min()} to {train_split['valid_time'].max()})")
    print(f"    - Test Set:     {len(test_split):,} hours "
          f"({test_split['valid_time'].min()} to {test_split['valid_time'].max()})")
    return train_split, test_split


def compute_persistence_baseline(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = "wp1"
) -> np.ndarray:
    """
    Naive 24h diurnal persistence baseline:
    For each test hour h, predict = actual power at the same hour from 24 hours prior.

    Uses the last 24 hours of the training set to bootstrap the first test predictions.
    Returns an array of persistence predictions aligned to test_df rows.
    """
    # Build a lookup series indexed by valid_time
    all_power = pd.concat([
        train_df[["valid_time", target_col]],
        test_df[["valid_time", target_col]]
    ]).sort_values("valid_time").set_index("valid_time")[target_col]

    persistence_preds = []
    for vt in test_df["valid_time"]:
        prior_time = vt - pd.Timedelta(hours=24)
        if prior_time in all_power.index:
            persistence_preds.append(all_power[prior_time])
        else:
            # If no prior observation exists, predict the training mean
            persistence_preds.append(train_df[target_col].mean())

    return np.array(persistence_preds)


def train_probabilistic_models(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Dict[str, lgb.LGBMRegressor]:
    """
    PHASE 2: Train three separate LightGBM quantile regressors for P10, P50, and P90.
    """
    quantiles = {"P10": 0.10, "P50": 0.50, "P90": 0.90}
    models = {}

    print("\n" + "=" * 75)
    print("PHASE 2: TRAINING PROBABILISTIC QUANTILE FORECASTERS (LIGHTGBM v2.0)")
    print(f"         Feature Set: {list(X_train.columns)}")
    print("=" * 75)

    for label, alpha in quantiles.items():
        print(f"[*] Training Model {label} (objective='quantile', alpha={alpha:.2f})...")
        params = LGBM_PARAMS.copy()
        params["objective"] = "quantile"
        params["alpha"] = alpha

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        models[label] = model
        print(f"    [OK] Model {label} trained successfully.")

    return models


def apply_physical_governors(
    p10_raw: np.ndarray, p50_raw: np.ndarray, p90_raw: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PHASE 3: AeroCast Physical Governors and Monotonicity.
      1. Zero Negative Generation: clamp all predictions < 0.0 to 0.0.
      2. Quantile Monotonicity: guarantee P10 <= P50 <= P90 for every row.
    """
    print("\n" + "=" * 75)
    print("PHASE 3: APPLYING AEROCAST PHYSICAL GOVERNORS & MONOTONICITY")
    print("=" * 75)

    total_neg = (p10_raw < 0.0).sum() + (p50_raw < 0.0).sum() + (p90_raw < 0.0).sum()
    print(f"[*] Zero Negative Clamp: Found {total_neg} sub-zero raw values. Forcing to 0.0.")

    p10 = np.maximum(p10_raw, 0.0)
    p50 = np.maximum(p50_raw, 0.0)
    p90 = np.maximum(p90_raw, 0.0)

    cross_10_50 = (p10 > p50).sum()
    cross_50_90 = (p90 < p50).sum()
    print(f"[*] Quantile Monotonicity: Detected {cross_10_50} (P10 > P50) and {cross_50_90} (P90 < P50) crossings.")

    p10 = np.minimum(p10, p50)
    p90 = np.maximum(p90, p50)

    assert (p10 >= 0.0).all(), "Fatal: negative P10 after clamping!"
    assert (p10 <= p50).all(), "Fatal: P10 > P50 after monotonicity!"
    assert (p50 <= p90).all(), "Fatal: P50 > P90 after monotonicity!"

    print("[OK] Physical governors applied. Invariant P10 <= P50 <= P90 guaranteed across 100% of rows.")
    return p10, p50, p90


def run_training_pipeline():
    """
    Version 2.0 end-to-end training pipeline.
    """
    ensure_directories()

    # 1. Load raw data (no features yet)
    raw_df = load_raw_data()

    # 2. Apply engineer_features on the FULL dataset BEFORE splitting
    #    This ensures the 24h lag sees contiguous history across the split boundary
    #    and the lag is correct. The feature module sorts chronologically internally.
    print("\n[*] Engineering features (Version 2.0 — includes target_lag_24h)...")
    full_featured = engineer_features(raw_df, target_col="wp1")

    n_lag_missing = int(full_featured["missing_history_flag"].sum())
    print(f"    [OK] target_lag_24h computed. First {n_lag_missing} rows flagged as missing history (filled 0.0).")
    print(f"    [OK] Total feature columns: {FEATURE_COLS}")

    # 3. Chronological split AFTER feature engineering
    train_df, test_df = split_train_test(full_featured)

    target_col = "wp1"
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[target_col]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[target_col]

    # 4. PHASE 2: Persistence Baseline (always printed first)
    print("\n" + "=" * 75)
    print("PHASE 2A: PERSISTENCE BASELINE (24h Diurnal Repeat)")
    print("=" * 75)
    persistence_preds = compute_persistence_baseline(train_df, test_df, target_col)
    persist_mae = float(np.mean(np.abs(y_test.values - persistence_preds)))
    persist_rmse = float(np.sqrt(np.mean((y_test.values - persistence_preds) ** 2)))
    print(f"    - Persistence MAE  (P50 equivalent): {persist_mae:.4f}")
    print(f"    - Persistence RMSE (P50 equivalent): {persist_rmse:.4f}")

    # 5. Train quantile models
    models = train_probabilistic_models(X_train, y_train)

    p10_raw = models["P10"].predict(X_test)
    p50_raw = models["P50"].predict(X_test)
    p90_raw = models["P90"].predict(X_test)

    # 6. Apply Physical Governors
    p10_clean, p50_clean, p90_clean = apply_physical_governors(p10_raw, p50_raw, p90_raw)

    # 7. Quick LightGBM MAE/RMSE vs baseline (before full evaluate.py run)
    lgbm_mae = float(np.mean(np.abs(y_test.values - p50_clean)))
    lgbm_rmse = float(np.sqrt(np.mean((y_test.values - p50_clean) ** 2)))
    skill = (1.0 - lgbm_mae / persist_mae) * 100.0 if persist_mae > 0 else 0.0

    print("\n" + "=" * 75)
    print("COMPARISON: PERSISTENCE BASELINE vs LIGHTGBM v2.0")
    print("=" * 75)
    print(f"  {'Model':<35} {'MAE':>8}  {'RMSE':>8}")
    print(f"  {'-'*52}")
    print(f"  {'Persistence Baseline (24h Diurnal)':<35} {persist_mae:>8.4f}  {persist_rmse:>8.4f}")
    print(f"  {'LightGBM v2.0 (P50, +target_lag_24h)':<35} {lgbm_mae:>8.4f}  {lgbm_rmse:>8.4f}")
    print(f"  {'Skill Score (MAE reduction)':<35} {skill:>7.2f}%")
    print("=" * 75)

    # 8. Export predictions
    predictions_df = pd.DataFrame({
        "valid_time": test_df["valid_time"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "actual_power": np.round(y_test.values, 4),
        "P10": np.round(p10_clean, 4),
        "P50": np.round(p50_clean, 4),
        "P90": np.round(p90_clean, 4),
        "persistence": np.round(persistence_preds, 4),
    })
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    print(f"\n[OK] Predictions exported to: '{PREDICTIONS_PATH}'")
    print(f"     Columns: {list(predictions_df.columns)} | Rows: {len(predictions_df):,}")
    print("\nSample Probabilistic Forecasts (First 5 Test Hours):")
    print("-" * 75)
    print(predictions_df[["valid_time", "actual_power", "P10", "P50", "P90"]].head(5).to_string(index=False))
    print("=" * 75)
    print("Training pipeline v2.0 complete. Run 'python ml_pipeline/evaluate.py' to evaluate accuracy.\n")


if __name__ == "__main__":
    run_training_pipeline()
