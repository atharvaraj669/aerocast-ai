"""
================================================================================
AEROCAST-AI: MACRO PROBABILISTIC QUANTILE FORECASTER & CLUSTERING SUITE
AI SANSAD 2026 | Team Prompt Diplomats (ID: UYU672B3)
================================================================================

Implements:
1. Multi-Quantile Panel Forecasters (P10, P50, P90):
   - Primary: LightGBM Quantile Pinball Regressor (Kaggle & Linux native)
   - Fallback: Scikit-Learn HistGradientBoostingRegressor (Zero-dependency portable)
2. Log1p Target Inversion & Physical Boundary Enforcement:
   - Quantile monotonicity guarantee: P10 <= P50 <= P90
   - Unconditional non-negativity: y >= 0
   - Percentage saturation clipping: y in [0, 100]% (when predicting shares)
3. CountryArchetypeClusterer:
   - PCA + K-Means clustering to discover national decarbonization typologies
================================================================================
"""

import time
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Check LightGBM availability
try:
    import lightgbm as lgb
    # Test if libomp is functional
    _test_data = lgb.Dataset(np.array([[1.0, 2.0], [3.0, 4.0]]), label=np.array([1.0, 2.0]))
    _test_param = {"verbosity": -1, "objective": "regression"}
    _test_bst = lgb.train(_test_param, _test_data, num_boost_round=1)
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False

from sklearn.ensemble import HistGradientBoostingRegressor


def calculate_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Pinball (quantile) loss for quantile alpha in (0, 1)."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def enforce_quantile_monotonicity(
    p10: np.ndarray, p50: np.ndarray, p90: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Strictly guarantees P10 <= P50 <= P90 without quantile crossing."""
    p10_fixed = np.minimum(p10, p50)
    p90_fixed = np.maximum(p90, p50)
    p50_fixed = np.clip(p50, p10_fixed, p90_fixed)
    return p10_fixed, p50_fixed, p90_fixed


class MacroQuantileForecaster:
    """
    Calibrated quantile regression suite with physical boundaries and log-target handling.
    """

    def __init__(
        self,
        quantiles: Tuple[float, ...] = (0.10, 0.50, 0.90),
        use_log_target: bool = True,
        is_percentage: bool = False,
        random_state: int = 42
    ):
        self.quantiles = quantiles
        self.use_log_target = use_log_target
        self.is_percentage = is_percentage
        self.random_state = random_state
        self.models: Dict[float, Any] = {}
        self.feature_names: List[str] = []
        self.backend = "LIGHTGBM" if HAS_LIGHTGBM else "SKLEARN_HIST_GBM"

    def fit(self, X: pd.DataFrame, y: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Fits asymmetric quantile pinball models for each quantile tau in [0.10, 0.50, 0.90].
        """
        self.feature_names = feature_names or list(X.columns)
        X_arr = np.nan_to_num(np.asarray(X), nan=0.0)
        y_arr = np.nan_to_num(np.asarray(y), nan=0.0)

        for q in self.quantiles:
            if HAS_LIGHTGBM:
                model = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=q,
                    n_estimators=150,
                    learning_rate=0.05,
                    num_leaves=31,
                    max_depth=6,
                    min_child_samples=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    verbosity=-1,
                    n_jobs=-1
                )
            else:
                # Portable Scikit-Learn native HistGradientBoosting with quantile loss
                model = HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=q,
                    max_iter=150,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    max_depth=6,
                    min_samples_leaf=5,
                    random_state=self.random_state
                )

            model.fit(X_arr, y_arr)
            self.models[q] = model

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generates calibrated P10, P50, P90 predictions with monotonicity and bounds.
        """
        X_arr = np.nan_to_num(np.asarray(X), nan=0.0)
        raw_preds = {}

        for q in self.quantiles:
            pred = self.models[q].predict(X_arr)
            raw_preds[q] = pred

        # Map to canonical quantile keys
        p10 = raw_preds.get(0.10, raw_preds.get(min(self.quantiles)))
        p50 = raw_preds.get(0.50, raw_preds.get(0.5, p10))
        p90 = raw_preds.get(0.90, raw_preds.get(max(self.quantiles)))

        # Invert log-transform if active: exp(y) - 1
        if self.use_log_target:
            p10 = np.expm1(np.maximum(0.0, p10))
            p50 = np.expm1(np.maximum(0.0, p50))
            p90 = np.expm1(np.maximum(0.0, p90))

        # Enforce physical boundaries
        p10 = np.maximum(0.0, p10)
        p50 = np.maximum(0.0, p50)
        p90 = np.maximum(0.0, p90)

        # Percentage target capping: [0, 100]
        if self.is_percentage:
            p10 = np.clip(p10, 0.0, 100.0)
            p50 = np.clip(p50, 0.0, 100.0)
            p90 = np.clip(p90, 0.0, 100.0)

        # Enforce strict quantile monotonicity: P10 <= P50 <= P90
        p10, p50, p90 = enforce_quantile_monotonicity(p10, p50, p90)

        return {"P10": p10, "P50": p50, "P90": p90}


class CountryArchetypeClusterer:
    """
    Unsupervised PCA + K-Means pipeline to segment countries into decarbonization archetypes.
    """

    ARCHETYPE_LABELS = {
        0: "Archetype A: Green Pioneers (High Renewables & Decoupled Growth)",
        1: "Archetype B: Rapid Industrializers (High Dual Fossil & Renewable Expansion)",
        2: "Archetype C: Energy Transition & Developing Access (Biomass Modernization)",
        3: "Archetype D: Fossil Heavy & Energy Rentiers (High Carbon Intensity)"
    }

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3, random_state=random_state)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.features_used: List[str] = []

    def fit_predict(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Fits clustering on representative country records and returns archetype assignments.
        """
        self.features_used = feature_cols
        X = np.nan_to_num(df[feature_cols].values, nan=0.0)
        n_comp = min(3, X.shape[1], X.shape[0])
        self.pca = PCA(n_components=n_comp, random_state=self.random_state)
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        n_clusters = min(self.n_clusters, len(df))
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        cluster_ids = self.kmeans.fit_predict(X_pca)

        res = df[["entity", "year"]].copy() if "entity" in df.columns else pd.DataFrame()
        res["cluster_id"] = cluster_ids
        for c_idx in range(n_comp):
            res[f"pca_{c_idx+1}"] = X_pca[:, c_idx]
        res["archetype_name"] = [self.ARCHETYPE_LABELS.get(cid, f"Cluster {cid}") for cid in cluster_ids]

        return res
