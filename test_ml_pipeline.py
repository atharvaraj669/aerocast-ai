"""
AeroCast-AI Version 2.0 - Automated Test Suite for ml_pipeline
==============================================================
Tests:
  1. Config path resolution
  2. Dummy GEFCom 2012 generation schema and value validity
  3. Data alignment and feature presence
  4. Physical Governors (zero-negative and monotonicity)
  5. Phase 3 (v2.0): target_lag_24h causal correctness
     - lag at index 24 == target_power at index 0
     - lag at index 0 does NOT contain future data (must be 0.0, missing_history_flag=1)
  6. Full end-to-end training and evaluation
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ml_pipeline.config import (
    PROJECT_ROOT,
    DATA_DIR,
    TRAIN_DATA_PATH,
    FORECAST_DATA_PATH,
    PREDICTIONS_PATH,
    OUTPUT_DIR,
)
from ml_pipeline.generate_dummy_gefcom import generate_gefcom_data
from ml_pipeline.features import engineer_features, FEATURE_COLS
from ml_pipeline.train import (
    load_raw_data,
    split_train_test,
    apply_physical_governors,
    train_probabilistic_models,
    compute_persistence_baseline,
)
from ml_pipeline.evaluate import evaluate_probabilistic_predictions


def test_config_paths_resolve():
    """Verify that all config paths are anchored to project root."""
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "ml_pipeline").exists()
    assert DATA_DIR == PROJECT_ROOT / "data" / "gefcom2012"
    assert OUTPUT_DIR == PROJECT_ROOT / "ml_pipeline" / "output"


def test_dummy_gefcom_generation():
    """Verify dummy data generation produces exact GEFCom 2012 columns and reasonable value bounds."""
    generate_gefcom_data(n_days=5, seed=123)

    assert TRAIN_DATA_PATH.exists()
    assert FORECAST_DATA_PATH.exists()

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    fc_df = pd.read_csv(FORECAST_DATA_PATH)

    assert list(train_df.columns) == ["date", "wp1"]
    assert list(fc_df.columns) == ["date", "hors", "ws", "wd"]
    assert len(train_df) == 120            # 5 days * 24 hours
    assert len(fc_df) == 120

    assert (train_df["wp1"] >= 0.0).all()
    assert (train_df["wp1"] <= 1.0).all()
    assert (fc_df["ws"] >= 0.0).all()
    assert (fc_df["ws"] <= 25.0).all()
    assert (fc_df["wd"] >= 0.0).all()
    assert (fc_df["wd"] <= 360.0).all()
    assert set(fc_df["hors"].unique()) == set(range(1, 25))


def test_data_alignment_and_features():
    """Verify load_raw_data + engineer_features produces all required feature columns."""
    generate_gefcom_data(n_days=5, seed=123)
    raw = load_raw_data()
    df = engineer_features(raw)

    assert len(df) > 0
    for col in FEATURE_COLS:
        assert col in df.columns, f"Missing feature: {col}"
        assert df[col].isna().sum() == 0, f"NaN in feature: {col}"


def test_chronological_split_is_leakage_free():
    """Verify that max(train valid_time) < min(test valid_time)."""
    generate_gefcom_data(n_days=30, seed=42)
    raw = load_raw_data()
    df = engineer_features(raw)
    train_df, test_df = split_train_test(df)

    assert len(train_df) > 0
    assert len(test_df) > 0
    assert train_df["valid_time"].max() < test_df["valid_time"].min()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 (v2.0) — Autoregressive Lag Causal Correctness Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_target_lag_24h_index_24_equals_target_at_index_0():
    """
    CRITICAL: target_lag_24h at row index 24 must exactly equal
    target_power at row index 0 (the first observation, 24 hours earlier).
    """
    generate_gefcom_data(n_days=30, seed=42)
    raw = load_raw_data()
    df = engineer_features(raw, target_col="wp1")

    # After sorting and reset_index(drop=True) in engineer_features:
    target_at_0 = df["target_power"].iloc[0]
    lag_at_24 = df["target_lag_24h"].iloc[24]

    assert lag_at_24 == pytest.approx(target_at_0, abs=1e-9), (
        f"target_lag_24h[24] ({lag_at_24:.6f}) != target_power[0] ({target_at_0:.6f}). "
        "The 24h lag is not correctly aligned — possible leakage or off-by-one error."
    )


def test_target_lag_24h_index_0_does_not_contain_future_data():
    """
    CRITICAL: target_lag_24h at index 0 must be 0.0 (missing history placeholder),
    and missing_history_flag must be 1. It must NOT contain any future observation.
    """
    generate_gefcom_data(n_days=30, seed=42)
    raw = load_raw_data()
    df = engineer_features(raw, target_col="wp1")

    lag_at_0 = df["target_lag_24h"].iloc[0]
    flag_at_0 = df["missing_history_flag"].iloc[0]

    assert lag_at_0 == pytest.approx(0.0, abs=1e-9), (
        f"target_lag_24h[0] should be 0.0 (no history), got {lag_at_0:.6f}. "
        "This may indicate .bfill() or forward-fill was applied, causing leakage."
    )
    assert flag_at_0 == 1, (
        f"missing_history_flag[0] should be 1 (missing), got {flag_at_0}."
    )


def test_first_24_rows_all_have_missing_history_flag():
    """The first 24 rows must all be marked as missing history (no 24h prior exists)."""
    generate_gefcom_data(n_days=30, seed=42)
    raw = load_raw_data()
    df = engineer_features(raw, target_col="wp1")

    first_24_flags = df["missing_history_flag"].iloc[:24].values
    assert (first_24_flags == 1).all(), (
        f"Expected all first 24 rows to have missing_history_flag=1. "
        f"Got: {first_24_flags}"
    )

    # From row 24 onwards, history is present → flag must be 0
    subsequent_flags = df["missing_history_flag"].iloc[24:].values
    assert (subsequent_flags == 0).all(), (
        "Rows 24+ should have missing_history_flag=0 (history is available)."
    )


def test_physical_governors_zero_negative_and_monotonicity():
    """Verify governors enforce P >= 0 and P10 <= P50 <= P90 under extreme crossings."""
    p10_raw = np.array([-0.05, 0.40, 0.80, -0.10, 0.50])
    p50_raw = np.array([ 0.10, 0.35, 0.70,  0.20, 0.50])
    p90_raw = np.array([ 0.25, 0.30, 0.90,  0.15, 0.45])

    p10, p50, p90 = apply_physical_governors(p10_raw, p50_raw, p90_raw)

    assert (p10 >= 0.0).all()
    assert (p50 >= 0.0).all()
    assert (p90 >= 0.0).all()
    assert (p10 <= p50).all()
    assert (p50 <= p90).all()

    # Row 0: p10 -0.05 → 0.0 (zero clamp); p50=0.10 > 0.0 → stays
    assert p10[0] == pytest.approx(0.0)

    # Row 1: p10=0.40 > p50=0.35 → clamp p10 to 0.35; p90=0.30 < p50=0.35 → clamp to 0.35
    assert p10[1] == pytest.approx(0.35)
    assert p90[1] == pytest.approx(0.35)

    # Row 2: p10=0.80 > p50=0.70 → p10 clamped to 0.70
    assert p10[2] == pytest.approx(0.70)
    assert p90[2] == pytest.approx(0.90)


def test_full_pipeline_v2_end_to_end():
    """Full end-to-end: train + evaluate v2.0. Verifies 0 violations and valid metric ranges."""
    generate_gefcom_data(n_days=30, seed=42)
    from ml_pipeline.train import run_training_pipeline
    run_training_pipeline()

    metrics = evaluate_probabilistic_predictions(PREDICTIONS_PATH)

    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0
    assert 0.0 <= metrics["picp_pct"] <= 100.0
    assert metrics["neg_violations"] == 0
    assert metrics["monotonicity_violations"] == 0
    # Persistence baseline should be present in v2.0
    assert metrics["persist_mae"] is not None
    assert metrics["persist_mae"] > 0
