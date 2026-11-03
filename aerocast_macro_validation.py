"""
================================================================================
AEROCAST-AI: MACRO PANEL WALK-FORWARD VALIDATION & METRIC SUITE
AI SANSAD 2026 | Team Prompt Diplomats (ID: UYU672B3)
================================================================================

Implements rigorous, leakage-free evaluation for macro-energy panel data:
1. Chronological expanding-window walk-forward splits:
   - Split A (Historic Validation): Train <= 2014, Evaluate 2015-2017 (3 years)
   - Split B (Holdout Competition): Train <= 2017, Evaluate 2018-2020 (3 years)
2. Comprehensive multi-metric scoring:
   - Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
   - Normalized MAE (NMAE %) against target dynamic range
   - Weighted Absolute Percentage Error (WAPE %): sum(|y - y_hat|) / sum(y)
   - Pinball Losses (tau = 0.10, 0.50, 0.90) and Mean Pinball Loss
   - Empirical Prediction Interval Coverage (PICP %) [Target: 80% nominal]
   - Prediction Interval Normalized Average Width (PINAW %)
   - Physical boundary & quantile monotonicity compliance checks
================================================================================
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Calculates asymmetric pinball loss for quantile alpha."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def compute_macro_metrics(
    y_true: np.ndarray,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    target_name: str = "Target"
) -> Dict[str, Any]:
    """
    Computes complete scientific evaluation suite for probabilistic macro forecasting.
    """
    y_arr = np.asarray(y_true, dtype=float)
    p10_arr = np.asarray(p10, dtype=float)
    p50_arr = np.asarray(p50, dtype=float)
    p90_arr = np.asarray(p90, dtype=float)

    n_samples = len(y_arr)
    if n_samples == 0:
        return {}

    mae = float(mean_absolute_error(y_arr, p50_arr))
    rmse = float(np.sqrt(mean_squared_error(y_arr, p50_arr)))

    # Normalization basis: Dynamic range
    dyn_range = float(np.max(y_arr) - np.min(y_arr))
    denom = dyn_range if dyn_range > 1e-4 else 1.0
    nmae = (mae / denom) * 100.0
    nrmse = (rmse / denom) * 100.0

    # WAPE (Weighted Absolute Percentage Error) - avoids division by zero on 0 TWh
    total_actual = float(np.sum(np.abs(y_arr)))
    total_abs_err = float(np.sum(np.abs(y_arr - p50_arr)))
    wape = (total_abs_err / total_actual * 100.0) if total_actual > 0 else 0.0

    # Quantile Pinball Loss
    pb10 = compute_pinball_loss(y_arr, p10_arr, 0.10)
    pb50 = compute_pinball_loss(y_arr, p50_arr, 0.50)
    pb90 = compute_pinball_loss(y_arr, p90_arr, 0.90)
    mean_pinball = (pb10 + pb50 + pb90) / 3.0

    # Empirical Coverage (PICP): Percentage of actuals inside [P10, P90]
    covered = (y_arr >= p10_arr) & (y_arr <= p90_arr)
    picp = float(np.mean(covered) * 100.0)

    # Average Interval Width (PINAW)
    avg_width = float(np.mean(p90_arr - p10_arr))
    pinaw = (avg_width / denom) * 100.0

    # Boundary Violations
    negative_violations = int(np.sum(p50_arr < 0.0) + np.sum(p10_arr < 0.0))
    crossing_violations = int(np.sum(p10_arr > p50_arr) + np.sum(p50_arr > p90_arr))

    return {
        "n_samples": n_samples,
        "target_name": target_name,
        "mae": mae,
        "rmse": rmse,
        "nmae_pct": nmae,
        "nrmse_pct": nrmse,
        "wape_pct": wape,
        "pinball_p10": pb10,
        "pinball_p50": pb50,
        "pinball_p90": pb90,
        "mean_pinball": mean_pinball,
        "picp_coverage_pct": picp,
        "pinaw_width_pct": pinaw,
        "negative_violations": negative_violations,
        "quantile_crossing_violations": crossing_violations
    }


class MacroPanelValidator:
    """
    Executes chronological walk-forward validation across country panels.
    Dynamically adapts cutoffs based on the temporal span of the dataset.
    """

    def __init__(
        self,
        train_end_year: Optional[int] = None,
        val_end_year: Optional[int] = None,
        test_end_year: Optional[int] = None,
        test_years_count: int = 2,
        val_years_count: int = 2
    ):
        self.train_end_year = train_end_year
        self.val_end_year = val_end_year
        self.test_end_year = test_end_year
        self.test_years_count = test_years_count
        self.val_years_count = val_years_count

    def resolve_cutoffs(self, df: pd.DataFrame) -> Tuple[int, int, int]:
        """Dynamically computes train/val/test year boundaries."""
        max_yr = int(df["year"].max())
        min_yr = int(df["year"].min())

        # If user explicitly supplied custom valid cutoffs
        if (
            self.train_end_year is not None
            and self.val_end_year is not None
            and self.test_end_year is not None
            and self.train_end_year < self.val_end_year < self.test_end_year <= max_yr
        ):
            return self.train_end_year, self.val_end_year, self.test_end_year

        # Dynamic cutoffs
        test_end = max_yr
        val_end = max_yr - self.test_years_count
        train_end = val_end - self.val_years_count

        if train_end <= min_yr + 2:
            train_end = min_yr + max(2, int((max_yr - min_yr) * 0.7))
            val_end = min_yr + max(3, int((max_yr - min_yr) * 0.85))
            test_end = max_yr

        return train_end, val_end, test_end

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits data chronologically by year without random shuffling leakage.
        """
        train_end, val_end, test_end = self.resolve_cutoffs(df)
        train_df = df[df["year"] <= train_end].copy()
        val_df = df[(df["year"] > train_end) & (df["year"] <= val_end)].copy()
        test_df = df[(df["year"] > val_end) & (df["year"] <= test_end)].copy()

        return train_df, val_df, test_df

