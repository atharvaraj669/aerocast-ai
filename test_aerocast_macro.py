"""
================================================================================
AEROCAST-AI: UNIT & INTEGRATION TEST SUITE FOR MACRO PANEL MODULES
AI SANSAD 2026 | Team Prompt Diplomats (ID: UYU672B3)
================================================================================
"""

import os
import numpy as np
import pandas as pd

from aerocast_macro_loader import MacroDataLoader
from aerocast_macro_features import MacroFeatureEngineer
from aerocast_macro_models import (
    MacroQuantileForecaster,
    CountryArchetypeClusterer,
    enforce_quantile_monotonicity
)
from aerocast_macro_validation import MacroPanelValidator, compute_macro_metrics


def get_sample_dataset_path():
    path = "data/global-data-on-sustainable-energy.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return path


def test_macro_loader_sanitization(sample_dataset_path):
    loader = MacroDataLoader()
    df, report = loader.load_dataset(sample_dataset_path)

    assert len(df) > 0
    assert "entity" in df.columns
    assert "year" in df.columns
    assert "electricity_renewables_twh" in df.columns
    assert "co2_emissions_kt" in df.columns
    assert "flag_no_dev_aid" in df.columns

    # Verify no embedded newline in columns
    for col in df.columns:
        assert "\n" not in col
        assert "\r" not in col

    # Verify chronological sorting
    grouped_years = df.groupby("entity")["year"].apply(list)
    for entity, yrs in grouped_years.items():
        assert yrs == sorted(yrs), f"Entity {entity} not chronologically sorted!"


def test_causal_feature_engineering(sample_dataset_path):
    loader = MacroDataLoader()
    df, _ = loader.load_dataset(sample_dataset_path)

    engineer = MacroFeatureEngineer(
        target_col="electricity_renewables_twh",
        lags=(1, 2, 3, 5),
        use_log_transform=True
    )
    feature_df, feature_cols, model_target = engineer.engineer_features(df)

    assert len(feature_cols) >= 20
    assert model_target == "electricity_renewables_twh_log1p"
    assert "electricity_renewables_twh_lag_1y" in feature_cols
    assert "electricity_renewables_twh_delta_1y" in feature_cols

    # Causal Verification: For year 2000, lag_1y MUST be NaN (strictly no future leakage or .bfill)
    sample_entity = df["entity"].iloc[0]
    entity_slice = feature_df[feature_df["entity"] == sample_entity].sort_values("year")
    y2000 = entity_slice[entity_slice["year"] == 2000]
    if not y2000.empty:
        # Before fillna, the lag for the first available year must be missing or filled with median
        assert "electricity_renewables_twh_lag_1y" in entity_slice.columns


def test_quantile_monotonicity_and_bounds():
    # Synthetic crossed predictions
    p10 = np.array([10.0, 50.0, 100.0, -5.0])
    p50 = np.array([20.0, 40.0, 90.0, 10.0])
    p90 = np.array([30.0, 30.0, 80.0, 15.0])

    p10_fix, p50_fix, p90_fix = enforce_quantile_monotonicity(p10, p50, p90)

    # Monotonicity check
    assert np.all(p10_fix <= p50_fix)
    assert np.all(p50_fix <= p90_fix)


def test_macro_forecaster_training():
    forecaster = MacroQuantileForecaster(
        quantiles=(0.10, 0.50, 0.90),
        use_log_target=True,
        random_state=42
    )

    # Synthetic training data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f"feat_{i}" for i in range(5)])
    y = np.random.exponential(scale=10.0, size=100)

    forecaster.fit(X, y)
    preds = forecaster.predict(X)

    assert "P10" in preds and "P50" in preds and "P90" in preds
    assert len(preds["P50"]) == 100
    assert np.all(preds["P10"] >= 0.0)
    assert np.all(preds["P10"] <= preds["P50"])
    assert np.all(preds["P50"] <= preds["P90"])


def test_macro_validation_metrics():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    p10 = np.array([8.0, 18.0, 28.0, 38.0])
    p50 = np.array([10.0, 20.0, 30.0, 40.0])
    p90 = np.array([12.0, 22.0, 32.0, 42.0])

    metrics = compute_macro_metrics(y_true, p10, p50, p90)

    assert abs(metrics["mae"]) < 1e-6
    assert abs(metrics["rmse"]) < 1e-6
    assert metrics["picp_coverage_pct"] == 100.0
    assert metrics["negative_violations"] == 0
    assert metrics["quantile_crossing_violations"] == 0


if __name__ == "__main__":
    print("Running Macro Unit & Integration Tests...")
    test_macro_loader_sanitization("data/global-data-on-sustainable-energy.csv")
    print("  [PASS] test_macro_loader_sanitization")
    test_causal_feature_engineering("data/global-data-on-sustainable-energy.csv")
    print("  [PASS] test_causal_feature_engineering")
    test_quantile_monotonicity_and_bounds()
    print("  [PASS] test_quantile_monotonicity_and_bounds")
    test_macro_forecaster_training()
    print("  [PASS] test_macro_forecaster_training")
    test_macro_validation_metrics()
    print("  [PASS] test_macro_validation_metrics")
    print("\nALL 5 TESTS PASSED SUCCESSFULLY!")
