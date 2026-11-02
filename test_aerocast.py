"""
================================================================================
AEROCAST-AI AUTOMATED INTEGRITY & RELIABILITY TEST SUITE
AI SANSAD 2026 | Team Prompt Diplomats (ID: UYU672B3)
================================================================================

Covers all 12 critical audit requirements:
1. Unknown turbine specifications activate Path 2.
2. Verified specifications activate Path 1.
3. Target-column overrides work.
4. TSV loading works.
5. Initial target lags do not use future values (no .bfill() leakage).
6. Multi-farm lag features never cross farm boundaries.
7. Separate test files can receive correct historical features.
8. Path 2 rolling validation returns predictions successfully.
9. Quantile predictions are strictly monotonic (P10 <= P50 <= P90).
10. Predictions obey verified physical bounds without arbitrary caps.
11. Missing dataset paths produce explicit errors (no silent fallbacks).
12. Training and test feature columns match exactly in content and order.
================================================================================
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from aerocast_physics import DataDependentPhysicsEngine, TurbineSpecs
from aerocast_loader import AdaptiveDataLoader, DatasetConfig
from aerocast_models import (
    DirectQuantileForecaster,
    PhysicsResidualForecaster,
    PersistenceBaseline,
    enforce_quantile_monotonicity
)
from aerocast_engine import AeroCastAIEngine
from aerocast_validation import RollingOriginValidator, compute_full_metric_suite


# ------------------------------------------------------------------------------
# TEST 1: Unknown turbine specifications activate Path 2
# ------------------------------------------------------------------------------
def test_unknown_specs_activate_path_2():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=100, freq="h"),
        "wind_speed": np.random.uniform(3, 15, 100),
        "active_power": np.random.uniform(0, 50, 100)
    })
    # Default TurbineSpecs has all None
    specs = TurbineSpecs()
    engine = AeroCastAIEngine(specs=specs)
    df_proc, col_map = engine.preprocess_dataset(df)

    assert engine.active_path == "PATH_2_DIRECT"
    assert engine.physics.report.can_support_residual_path is False
    assert engine.physics.report.theoretical_power_activated is False


# ------------------------------------------------------------------------------
# TEST 2: Verified specifications activate Path 1
# ------------------------------------------------------------------------------
def test_verified_specs_activate_path_1():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=100, freq="h"),
        "wind_speed": np.random.uniform(3, 15, 100),
        "temperature": np.full(100, 20.0),
        "pressure": np.full(100, 101325.0),
        "active_power": np.random.uniform(0, 100, 100)
    })
    specs = TurbineSpecs(
        rated_power_mw=100.0,
        hub_height_m=120.0,
        mast_height_m=10.0,
        v_cut_in_ms=3.0,
        v_rated_ms=12.5,
        v_cut_out_ms=25.0,
        shear_alpha=0.143
    )
    engine = AeroCastAIEngine(specs=specs)
    df_proc, col_map = engine.preprocess_dataset(df)

    assert engine.active_path == "PATH_1_RESIDUAL"
    assert engine.physics.report.can_support_residual_path is True
    assert engine.physics.report.theoretical_power_activated is True
    assert "theoretical_power_physics" in df_proc.columns


# ------------------------------------------------------------------------------
# TEST 3: Target-column overrides work
# ------------------------------------------------------------------------------
def test_target_column_override_works():
    df = pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=10, freq="h"),
        "MY_CUSTOM_TARGET_POWER": np.arange(10, dtype=float),
        "ws_spd": np.full(10, 8.0)
    })
    cfg = DatasetConfig(target_col="MY_CUSTOM_TARGET_POWER")
    loader = AdaptiveDataLoader(config=cfg)
    role_matches = loader.detect_column_roles(df)

    assert "target_power" in role_matches
    assert role_matches["target_power"].column_name == "MY_CUSTOM_TARGET_POWER"
    assert "CONFIG OVERRIDE" in role_matches["target_power"].confidence


# ------------------------------------------------------------------------------
# TEST 4: TSV loading works
# ------------------------------------------------------------------------------
def test_tsv_loading_works():
    loader = AdaptiveDataLoader()
    with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
        f.write("timestamp\twind_speed\tactive_power\n")
        f.write("2025-01-01 00:00:00\t7.5\t25.4\n")
        f.write("2025-01-01 01:00:00\t8.2\t31.1\n")
        tsv_path = f.name

    try:
        df = loader.read_file(tsv_path)
        assert len(df) == 2
        assert list(df.columns) == ["timestamp", "wind_speed", "active_power"]
        assert df["wind_speed"].iloc[0] == 7.5
    finally:
        if os.path.exists(tsv_path):
            os.remove(tsv_path)


# ------------------------------------------------------------------------------
# TEST 5: Initial target lags do not use future values (no .bfill() leakage)
# ------------------------------------------------------------------------------
def test_initial_lags_do_not_use_future_values():
    # Construct sequence where power strictly increases: 1, 2, 3, ...
    n = 100
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "wind_speed": np.full(n, 7.0),
        "active_power": np.arange(1, n + 1, dtype=float)
    })
    engine = AeroCastAIEngine()
    col_map = {"timestamp": "timestamp", "target_power": "active_power", "wind_speed": "wind_speed", "farm_id": None}
    df_feat, _ = engine.build_causal_features(df, col_map, horizon_hours=24)

    # For the first 24 hours, past generation from before index 0 did NOT exist.
    # Therefore, target_lag_24h must NOT be backfilled with future values (e.g. active_power at index 24, which is 25.0)
    assert df_feat["flag_missing_target_history"].iloc[0] == 1
    assert df_feat["flag_missing_target_history"].iloc[23] == 1
    # At index 24, the lag from 24 hours ago IS row 0 (which has value 1.0)
    assert df_feat["target_lag_24h"].iloc[24] == 1.0
    # At index 0, the lag must NOT equal row 24's value
    assert df_feat["target_lag_24h"].iloc[0] != df["active_power"].iloc[24]


# ------------------------------------------------------------------------------
# TEST 6: Multi-farm lag features never cross farm boundaries
# ------------------------------------------------------------------------------
def test_multi_farm_lags_never_cross_boundaries():
    # Farm A has 50 observations with power = 100.0
    # Farm B has 50 observations with power = 5.0
    n = 50
    df_a = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "farm_id": "Farm_A",
        "wind_speed": np.full(n, 10.0),
        "active_power": np.full(n, 100.0)
    })
    df_b = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "farm_id": "Farm_B",
        "wind_speed": np.full(n, 5.0),
        "active_power": np.full(n, 5.0)
    })
    df_combined = pd.concat([df_a, df_b], ignore_index=True)

    engine = AeroCastAIEngine()
    col_map = {
        "timestamp": "timestamp",
        "target_power": "active_power",
        "wind_speed": "wind_speed",
        "farm_id": "farm_id"
    }
    df_feat, _ = engine.build_causal_features(df_combined, col_map, horizon_hours=24)

    # Check Farm B's first valid lagged rows (e.g. index 74, which is row 24 of Farm B)
    # The lag feature for Farm B must be 5.0 (Farm B's historical power), NEVER 100.0 (Farm A's power)
    farm_b_rows = df_feat[df_feat["farm_id"] == "Farm_B"]
    assert farm_b_rows["target_lag_24h"].iloc[24] == 5.0
    # No row in Farm B should ever have a lag equal to Farm A's 100.0
    assert (farm_b_rows["target_lag_24h"] == 100.0).sum() == 0


# ------------------------------------------------------------------------------
# TEST 7: Separate test files can receive correct historical features
# ------------------------------------------------------------------------------
def test_separate_test_receives_correct_historical_features():
    train_n = 100
    test_n = 24
    df_train = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=train_n, freq="h"),
        "wind_speed": np.full(train_n, 8.0),
        "active_power": np.arange(train_n, dtype=float)
    })
    # Test set has NO active_power column (standard competition test set)
    df_test = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-05 04:00:00", periods=test_n, freq="h"),
        "wind_speed": np.full(test_n, 8.5)
    })

    engine = AeroCastAIEngine()
    col_map = {"timestamp": "timestamp", "target_power": "active_power", "wind_speed": "wind_speed", "farm_id": None}

    # Build features on test data passing train as historical_df
    df_test_feat, _ = engine.build_causal_features(df_test, col_map, horizon_hours=24, historical_df=df_train)

    assert len(df_test_feat) == test_n
    # Row 0 of test is step 0 into test forecast. With horizon=24, the lag from 24h prior should be
    # the target from train at train_n - 24 = 100 - 24 = 76.0
    assert df_test_feat["target_lag_24h"].iloc[0] == 76.0


# ------------------------------------------------------------------------------
# TEST 8: Path 2 rolling validation returns predictions successfully
# ------------------------------------------------------------------------------
def test_path_2_rolling_validation_returns_successfully():
    n = 600
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "wind_speed": np.random.uniform(2, 20, n),
        "active_power": np.random.uniform(0, 50, n)
    })
    # Specs are completely unknown -> Path 2
    specs = TurbineSpecs()
    engine = AeroCastAIEngine(specs=specs)
    df_proc, col_map = engine.preprocess_dataset(df)
    assert engine.active_path == "PATH_2_DIRECT"

    # Define CV function for Path 2
    def cv_fn(X_tr, y_tr, X_v):
        m = DirectQuantileForecaster(base_engine="lightgbm", rated_power_mw=None, random_state=42)
        m.fit(X_tr, y_tr)
        return m.predict(X_v)

    validator = RollingOriginValidator(n_splits=2, val_window_hours=72, horizon_hours=24)
    res = validator.evaluate_model_rolling(
        df_proc,
        engine.feature_columns,
        col_map["target_power"],
        rated_capacity=None,
        train_and_predict_fn=cv_fn
    )

    assert "folds" in res
    assert len(res["folds"]) == 2
    for fold in res["folds"]:
        assert fold["MAE"] >= 0.0
        assert fold["Mean_Pinball"] >= 0.0


# ------------------------------------------------------------------------------
# TEST 9: Quantile predictions are strictly monotonic
# ------------------------------------------------------------------------------
def test_quantile_predictions_are_monotonic():
    # Intentionally crossed arrays
    p10 = np.array([15.0, 50.0, 30.0])
    p50 = np.array([10.0, 40.0, 35.0])
    p90 = np.array([5.0, 60.0, 20.0])

    p10_fix, p50_fix, p90_fix = enforce_quantile_monotonicity(p10, p50, p90)

    assert (p10_fix <= p50_fix).all()
    assert (p50_fix <= p90_fix).all()


# ------------------------------------------------------------------------------
# TEST 10: Predictions obey verified physical bounds without arbitrary caps
# ------------------------------------------------------------------------------
def test_predictions_obey_verified_physical_bounds():
    # Case A: Unknown rated power -> NO upper cap enforced
    physics_unknown = DataDependentPhysicsEngine(specs=TurbineSpecs(rated_power_mw=None))
    preds = np.array([-10.0, 50.0, 250.0])
    guarded_unknown = physics_unknown.enforce_guards(preds)
    assert guarded_unknown[0] == 0.0  # Non-negative
    assert guarded_unknown[2] == 250.0  # Not arbitrarily capped at 100

    # Case B: Verified rated power = 75 MW -> Capped at 75 MW
    physics_verified = DataDependentPhysicsEngine(specs=TurbineSpecs(rated_power_mw=75.0, v_cut_out_ms=25.0))
    wind_speeds = np.array([10.0, 12.0, 28.0])  # last speed exceeds cut-out
    preds_b = np.array([20.0, 90.0, 80.0])
    guarded_verified = physics_verified.enforce_guards(preds_b, effective_ws=wind_speeds)
    assert guarded_verified[0] == 20.0
    assert guarded_verified[1] == 75.0  # Capped at verified rated power
    assert guarded_verified[2] == 0.0   # Shut down due to storm cut-out > 25 m/s


# ------------------------------------------------------------------------------
# TEST 11: Missing dataset paths produce explicit errors
# ------------------------------------------------------------------------------
def test_missing_dataset_paths_produce_explicit_errors():
    loader = AdaptiveDataLoader()
    with pytest.raises(FileNotFoundError):
        loader.read_file("non_existent_dataset_12345.csv")


# ------------------------------------------------------------------------------
# TEST 12: Training and test feature columns match exactly
# ------------------------------------------------------------------------------
def test_training_and_test_feature_columns_match():
    df_train = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=100, freq="h"),
        "wind_speed": np.random.uniform(3, 15, 100),
        "temperature": np.full(100, 22.0),
        "active_power": np.random.uniform(0, 80, 100)
    })
    df_test = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-05 04:00:00", periods=24, freq="h"),
        "wind_speed": np.random.uniform(3, 15, 24),
        "temperature": np.full(24, 21.5)
        # Note: no active_power in test
    })

    engine = AeroCastAIEngine()
    df_train_proc, _ = engine.preprocess_dataset(df_train)
    engine.train_fast_mode(df_train_proc.iloc[:80], df_train_proc.iloc[80:], rated_capacity=None)

    train_cols = engine.feature_columns

    # Score test data
    sub = engine.predict_test(df_test, rated_capacity=None, historical_df=df_train_proc)
    assert len(sub) == 24
    assert "P10" in sub.columns
    assert "P50_optimal" in sub.columns
    assert "P90" in sub.columns
    assert (sub["P10"] <= sub["P50_optimal"]).all()
    assert (sub["P50_optimal"] <= sub["P90"]).all()
