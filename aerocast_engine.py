"""
================================================================================
AEROCAST-AI: UNIFIED ADAPTIVE HYBRID FORECASTING ENGINE (REPAIRED)
AI SANSAD 2026 | Department of AI & ML, Bharati Vidyapeeth's College of Engineering
Team: Prompt Diplomats | Team ID: UYU672B3 | Leader: Atharva Sandeep Patil
================================================================================

Assigned Problem Statement:
"Forecast hourly wind power output using historical wind-farm generation and available wind forecasts."

Core Research Contribution:
"Physics-Informed AI with Probabilistic Bounds"

Audited & Repaired Highlights:
1. Data-Dependent Physics Engine: Only physically valid calculations activated.
   No default specs fabricated. Unknown specs remain None.
2. Two Forecasting Paths:
   - Path 1: Physics-Informed Residual Forecasting (when verified physics baseline exists)
   - Path 2: Direct Probabilistic Quantile Forecasting (when specs are unverified)
3. Causal Feature Engineering:
   - Strictly NO .bfill() on target features.
   - Lags/rolling stats computed independently per farm/turbine group.
   - True inference-history interface for separate test datasets.
   - Explicit flag_missing_target_history indicator.
4. Model Implementations:
   - LightGBM, CatBoost with categorical support, and TemporalQuantileGBM challenger.
   - Exact feature alignment across training and test inference.
   - Full-dataset champion refitting prior to test scoring.
================================================================================
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from aerocast_physics import DataDependentPhysicsEngine, TurbineSpecs, PhysicsActivationReport
from aerocast_loader import AdaptiveDataLoader, DatasetConfig
from aerocast_models import (
    PersistenceBaseline,
    LightGBMQuantileForecaster,
    CatBoostQuantileForecaster,
    TemporalQuantileGBM,
    PhysicsResidualForecaster,
    DirectQuantileForecaster,
    SelectiveValidationEnsembler,
    enforce_quantile_monotonicity,
    HAS_LIGHTGBM,
    HAS_CATBOOST
)
from aerocast_validation import (
    RollingOriginValidator,
    compute_full_metric_suite,
    resolve_normalization_denominator
)


class AeroCastAIEngine:
    """
    Main orchestrator for the AeroCast-AI forecasting framework.
    """

    def __init__(
        self,
        config: Optional[DatasetConfig] = None,
        specs: Optional[TurbineSpecs] = None,
        random_state: int = 42
    ):
        self.config = config or DatasetConfig()
        # Keep unknown specs as None - do NOT invent default capacities or heights!
        self.specs = specs or TurbineSpecs(
            rated_power_mw=self.config.rated_power_mw,
            hub_height_m=self.config.hub_height_m,
            mast_height_m=self.config.mast_height_m,
            v_cut_in_ms=self.config.v_cut_in_ms,
            v_rated_ms=self.config.v_rated_ms,
            v_cut_out_ms=self.config.v_cut_out_ms,
            shear_alpha=self.config.shear_alpha
        )
        self.random_state = random_state

        self.loader = AdaptiveDataLoader(config=self.config)
        self.physics = DataDependentPhysicsEngine(specs=self.specs)
        self.validator = RollingOriginValidator(horizon_hours=self.config.forecast_horizon_hours)

        self.active_path: str = ""  # "PATH_1_RESIDUAL" or "PATH_2_DIRECT"
        self.column_map: Dict[str, Optional[str]] = {}
        self.feature_columns: List[str] = []
        self.champion_model: Any = None
        self.champion_name: str = ""
        self.is_ensemble: bool = False
        self.ensemble_weights: Dict[str, float] = {}
        self.ensembler: Optional[SelectiveValidationEnsembler] = None
        self.trained_models: Dict[str, Any] = {}
        self.last_training_data: Optional[pd.DataFrame] = None

    def build_causal_features(
        self,
        df: pd.DataFrame,
        col_map: Dict[str, Optional[str]],
        horizon_hours: int = 24,
        historical_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Builds strictly causal temporal and lag/rolling features:
        - NEVER applies .bfill() to target-derived features.
        - Sorts observations chronologically.
        - Generates target lags and rolling stats independently for each farm group.
        - Supports an inference-history interface for separate test datasets.
        - Explicitly marks missing history with 'flag_missing_target_history'.
        """
        ts_col = col_map.get("timestamp")
        target_col = col_map.get("target_power")
        farm_col = col_map.get("farm_id")
        ws_col = col_map.get("wind_speed")
        status_col = col_map.get("curtailment")

        eval_df = df.copy()
        eval_df["_orig_row_idx"] = np.arange(len(eval_df))
        eval_df["_is_eval_row"] = True

        # Combine with historical generation if provided for test inference
        if historical_df is not None and target_col and target_col in historical_df:
            hist_subset = historical_df.copy()
            hist_subset["_orig_row_idx"] = -1
            hist_subset["_is_eval_row"] = False
            combined = pd.concat([hist_subset, eval_df], ignore_index=True)
        else:
            combined = eval_df

        # Sort chronologically by [farm_col, ts_col] or ts_col
        sort_keys = []
        if farm_col and farm_col in combined.columns:
            sort_keys.append(farm_col)
        if ts_col and ts_col in combined.columns:
            combined[ts_col] = pd.to_datetime(combined[ts_col], errors="coerce")
            sort_keys.append(ts_col)

        if sort_keys:
            combined = combined.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)

        generated_cols: List[str] = []

        # 1. Cyclical Time Features (diurnal & seasonal)
        if ts_col and ts_col in combined.columns:
            dt = combined[ts_col]
            valid_dt = dt.dropna()
            if len(valid_dt) > 0:
                hour = dt.dt.hour
                try:
                    doy = dt.dt.day_of_year
                except AttributeError:
                    doy = dt.dt.dayofyear
                combined["sin_hour"] = np.sin(2 * np.pi * hour / 24.0).fillna(0.0)
                combined["cos_hour"] = np.cos(2 * np.pi * hour / 24.0).fillna(0.0)
                combined["sin_dayofyear"] = np.sin(2 * np.pi * doy / 365.25).fillna(0.0)
                combined["cos_dayofyear"] = np.cos(2 * np.pi * doy / 365.25).fillna(0.0)
                generated_cols.extend(["sin_hour", "cos_hour", "sin_dayofyear", "cos_dayofyear"])

        # 2. Causal Autoregressive Target Lags (Independently per Farm Group, NO .bfill())
        has_targets = target_col and target_col in combined.columns and combined[target_col].notna().any()
        lag_col_h = f"target_lag_{horizon_hours}h"
        lag_col_h24 = f"target_lag_{horizon_hours+24}h"
        roll_mean_col = f"target_roll_mean_{horizon_hours}h"
        roll_std_col = f"target_roll_std_{horizon_hours}h"
        miss_flag_col = "flag_missing_target_history"

        if has_targets:
            if farm_col and farm_col in combined.columns:
                # Group-isolated lag computation
                combined[lag_col_h] = combined.groupby(farm_col, group_keys=False)[target_col].shift(horizon_hours)
                combined[lag_col_h24] = combined.groupby(farm_col, group_keys=False)[target_col].shift(horizon_hours + 24)
                combined[roll_mean_col] = (
                    combined.groupby(farm_col, group_keys=False)[target_col]
                    .shift(horizon_hours)
                    .rolling(24, min_periods=1)
                    .mean()
                )
                combined[roll_std_col] = (
                    combined.groupby(farm_col, group_keys=False)[target_col]
                    .shift(horizon_hours)
                    .rolling(24, min_periods=1)
                    .std()
                )
            else:
                combined[lag_col_h] = combined[target_col].shift(horizon_hours)
                combined[lag_col_h24] = combined[target_col].shift(horizon_hours + 24)
                combined[roll_mean_col] = combined[target_col].shift(horizon_hours).rolling(24, min_periods=1).mean()
                combined[roll_std_col] = combined[target_col].shift(horizon_hours).rolling(24, min_periods=1).std()

            # Strictly NO .bfill()! Record missing history condition explicitly
            combined[miss_flag_col] = combined[lag_col_h].isna().astype(int)
            # Impute NaN with 0.0 only accompanied by the explicit missing flag
            combined[lag_col_h] = combined[lag_col_h].fillna(0.0)
            combined[lag_col_h24] = combined[lag_col_h24].fillna(0.0)
            combined[roll_mean_col] = combined[roll_mean_col].fillna(0.0)
            combined[roll_std_col] = combined[roll_std_col].fillna(0.0)
        else:
            # Entire target history missing for this dataset
            combined[lag_col_h] = 0.0
            combined[lag_col_h24] = 0.0
            combined[roll_mean_col] = 0.0
            combined[roll_std_col] = 0.0
            combined[miss_flag_col] = 1

        generated_cols.extend([lag_col_h, lag_col_h24, roll_mean_col, roll_std_col, miss_flag_col])

        # 3. Weather Forecast Dynamics (per farm group)
        if ws_col and ws_col in combined.columns:
            if farm_col and farm_col in combined.columns:
                ws_grp = combined.groupby(farm_col, group_keys=False)[ws_col]
                combined["ws_delta_3h"] = ws_grp.diff(3).fillna(0.0)
                combined["ws_roll_max_6h"] = ws_grp.rolling(6, min_periods=1).max().fillna(combined[ws_col])
                combined["ws_roll_min_6h"] = ws_grp.rolling(6, min_periods=1).min().fillna(combined[ws_col])
            else:
                combined["ws_delta_3h"] = combined[ws_col].diff(3).fillna(0.0)
                combined["ws_roll_max_6h"] = combined[ws_col].rolling(6, min_periods=1).max().fillna(combined[ws_col])
                combined["ws_roll_min_6h"] = combined[ws_col].rolling(6, min_periods=1).min().fillna(combined[ws_col])
            combined["ws_acceleration"] = combined["ws_delta_3h"].diff(1).fillna(0.0)
            generated_cols.extend(["ws_delta_3h", "ws_roll_max_6h", "ws_roll_min_6h", "ws_acceleration"])

        # 4. Curtailment / Operating Flag Feature
        if status_col and status_col in combined.columns:
            combined["status_numeric"] = pd.to_numeric(combined[status_col], errors="coerce").fillna(1.0)
            generated_cols.append("status_numeric")

        if target_col and target_col in combined.columns and ws_col and ws_col in combined.columns:
            p_val = pd.to_numeric(combined[target_col], errors="coerce").fillna(0.0)
            ws_val = pd.to_numeric(combined[ws_col], errors="coerce").fillna(0.0)
            combined["flag_suspicious_outage"] = ((p_val <= 1e-4) & (ws_val >= 5.0)).astype(int)
            generated_cols.append("flag_suspicious_outage")

        # 5. Farm / Turbine Categorical Encoding
        if farm_col and farm_col in combined.columns:
            if farm_col in self.loader.category_mappings:
                combined, enc_cols = self.loader.transform_categories(combined, [farm_col])
            else:
                combined, enc_cols = self.loader.fit_and_encode_categories(combined, [farm_col])
            generated_cols.extend(enc_cols)

        # Slice back to evaluation rows and restore original ordering
        out_eval = combined[combined["_is_eval_row"]].copy()
        out_eval = out_eval.sort_values("_orig_row_idx").reset_index(drop=True)
        out_eval = out_eval.drop(columns=["_orig_row_idx", "_is_eval_row"], errors="ignore")

        return out_eval, list(set(generated_cols))

    def preprocess_dataset(
        self, df: pd.DataFrame, require_confident_target: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
        """
        Detects schema, runs physics activations, builds causal features, and selects forecasting path.
        """
        role_matches = self.loader.detect_column_roles(df, require_confident_target=require_confident_target)
        col_map = {role: match.column_name for role, match in role_matches.items()}
        for k in ["timestamp", "target_power", "wind_speed", "wind_direction", "temperature", "pressure", "farm_id", "turbine_id", "curtailment"]:
            if k not in col_map:
                col_map[k] = None

        self.column_map = col_map

        # Apply Data-Dependent Physics Engine
        df_phys, phys_features = self.physics.transform(df, col_map)
        print(self.physics.report.summary())

        # Determine forecasting path based strictly on verified physics feasibility
        if self.physics.report.can_support_residual_path:
            self.active_path = "PATH_1_RESIDUAL"
            print("\n[+] PATH SELECTION: PATH 1 (Physics-Informed Residual Forecasting)")
            print("    Equation: P_actual = P_physics + P_residual")
        else:
            self.active_path = "PATH_2_DIRECT"
            print("\n[+] PATH SELECTION: PATH 2 (Direct Probabilistic Quantile Forecasting)")
            print("    Reason: Grounded physics baseline cannot be calculated without unverified specs.")

        # Build causal temporal & weather dynamics features
        df_full, causal_features = self.build_causal_features(df_phys, col_map, self.config.forecast_horizon_hours)

        # Establish feature column set (exclude raw targets, timestamps, raw IDs)
        exclude = {col_map.get("target_power"), col_map.get("timestamp"), col_map.get("farm_id"), col_map.get("turbine_id")}
        feature_cols = [c for c in df_full.columns if c not in exclude and pd.api.types.is_numeric_dtype(df_full[c])]

        self.feature_columns = feature_cols
        self.last_training_data = df_full
        return df_full, col_map

    def train_fast_mode(
        self, df_train: pd.DataFrame, df_val: pd.DataFrame, rated_capacity: Optional[float]
    ) -> Dict[str, Any]:
        """
        FAST COMPETITION MODE:
        Rapid LightGBM quantile regression with physical guards and zero speculation.
        """
        print("\n" + "=" * 75)
        print("RUNNING IN FAST MODE (Optimized for Rapid Reliable Submissions)")
        print("=" * 75)
        target_col = self.column_map["target_power"]

        X_train = df_train[self.feature_columns]
        y_train = df_train[target_col]
        X_val = df_val[self.feature_columns]
        y_val = df_val[target_col]

        t0 = time.time()
        if self.active_path == "PATH_1_RESIDUAL":
            p_phys_train = df_train["theoretical_power_physics"]
            p_phys_val = df_val["theoretical_power_physics"]
            model = PhysicsResidualForecaster(base_engine="lightgbm", rated_power_mw=rated_capacity, random_state=self.random_state)
            model.fit(X_train, y_train, p_phys_train)
            val_preds = model.predict(X_val, p_phys_val)
        else:
            model = DirectQuantileForecaster(base_engine="lightgbm", rated_power_mw=rated_capacity, random_state=self.random_state)
            model.fit(X_train, y_train)
            val_preds = model.predict(X_val)

        eff_ws_val = df_val.get("v_hub_physics", df_val.get(self.column_map.get("wind_speed", "")))
        for q in ["P10", "P50", "P90"]:
            val_preds[q] = self.physics.enforce_guards(val_preds[q], eff_ws_val.values if eff_ws_val is not None else None)

        val_preds["P10"], val_preds["P50"], val_preds["P90"] = enforce_quantile_monotonicity(
            val_preds["P10"], val_preds["P50"], val_preds["P90"]
        )

        metrics = compute_full_metric_suite(y_val.values, val_preds["P10"], val_preds["P50"], val_preds["P90"], rated_capacity)
        elapsed = time.time() - t0
        self.champion_model = model
        self.champion_name = "Fast_LightGBM_Champion"
        self.is_ensemble = False

        print(f"[OK] Fast Model Trained in {elapsed:.2f}s!")
        print(f"    - Normalization Basis: {metrics.get('Normalization_Basis', 'N/A')}")
        print(f"    - NMAE:                {metrics['NMAE_%']:.2f}%")
        print(f"    - NRMSE:               {metrics['NRMSE_%']:.2f}%")
        print(f"    - Mean Pinball:        {metrics['Mean_Pinball']:.3f}")
        print(f"    - PICP Coverage:       {metrics['Coverage_PICP_%']:.1f}%")
        return {"model": model, "metrics": metrics, "val_preds": val_preds, "elapsed": elapsed}

    def train_full_experiment_mode(
        self, df_train: pd.DataFrame, df_val: pd.DataFrame, rated_capacity: Optional[float]
    ) -> Dict[str, Any]:
        """
        FULL EXPERIMENT MODE:
        Multi-model comparison: Persistence, LightGBM, CatBoost, TemporalQuantileGBM,
        Physics Residual (if Path 1 active), and Selective Validation Ensembling.
        """
        print("\n" + "=" * 75)
        print("RUNNING IN FULL EXPERIMENT MODE (Multi-Model Scientific Benchmark)")
        print("=" * 75)
        target_col = self.column_map["target_power"]

        X_train = df_train[self.feature_columns]
        y_train = df_train[target_col]
        X_val = df_val[self.feature_columns]
        y_val = df_val[target_col]

        eff_ws_val = df_val.get("v_hub_physics", df_val.get(self.column_map.get("wind_speed", "")))
        ws_val_arr = eff_ws_val.values if eff_ws_val is not None else None

        candidate_preds: Dict[str, Dict[str, np.ndarray]] = {}

        # 1. Persistence Baseline
        print("[1/5] Fitting Persistence Baseline (Diurnal repeating cycle)...")
        pb = PersistenceBaseline(diurnal_lag=self.config.forecast_horizon_hours)
        pb.fit(y_train.values)
        candidate_preds["Persistence_Baseline"] = pb.predict(y_train.values, len(y_val))

        # 2. LightGBM Direct
        print("[2/5] Training LightGBM Quantile Forecaster (P10, P50, P90)...")
        lgb_model = DirectQuantileForecaster(base_engine="lightgbm", rated_power_mw=rated_capacity, random_state=self.random_state)
        lgb_model.fit(X_train, y_train)
        candidate_preds["LightGBM_Quantile"] = lgb_model.predict(X_val)

        # 3. CatBoost Direct
        cat_cols = [c for c in ["farm_code", "farm_id_code"] if c in X_train.columns]
        if HAS_CATBOOST:
            print("[3/5] Training CatBoost Quantile Forecaster (P10, P50, P90)...")
            cb_model = DirectQuantileForecaster(base_engine="catboost", rated_power_mw=rated_capacity, random_state=self.random_state)
            cb_model.fit(X_train, y_train)
            candidate_preds["CatBoost_Quantile"] = cb_model.predict(X_val)
        else:
            print("[3/5] CatBoost not available; skipping.")

        # 4. Temporal Quantile GBM (Challenger)
        print("[4/5] Training Temporal Quantile GBM Challenger...")
        temporal_gbm = TemporalQuantileGBM(random_state=self.random_state)
        temporal_gbm.fit(X_train, y_train)
        candidate_preds["Temporal_Quantile_GBM"] = temporal_gbm.predict(X_val)

        # 5. Physics-Informed Residual (Path 1)
        if self.active_path == "PATH_1_RESIDUAL":
            print("[5/5] Training Path 1 Physics-Informed Residual Forecaster...")
            p_phys_train = df_train["theoretical_power_physics"]
            p_phys_val = df_val["theoretical_power_physics"]
            res_model = PhysicsResidualForecaster(base_engine="lightgbm", rated_power_mw=rated_capacity, random_state=self.random_state)
            res_model.fit(X_train, y_train, p_phys_train)
            candidate_preds["AeroCast_Physics_Residual"] = res_model.predict(X_val, p_phys_val)
        else:
            print("[5/5] Path 1 disabled: unverified turbine specifications. Path 2 active.")

        # Apply physical boundary clamps to all candidates
        for name in candidate_preds:
            for q in ["P10", "P50", "P90"]:
                candidate_preds[name][q] = self.physics.enforce_guards(candidate_preds[name][q], ws_val_arr)
            candidate_preds[name]["P10"], candidate_preds[name]["P50"], candidate_preds[name]["P90"] = enforce_quantile_monotonicity(
                candidate_preds[name]["P10"], candidate_preds[name]["P50"], candidate_preds[name]["P90"]
            )

        # 6. Selective Validation Ensembler
        print("\n[+] Evaluating Candidates on Validation Split...")
        ensembler = SelectiveValidationEnsembler(rated_power_mw=rated_capacity, min_improvement_pct=0.5)
        champion_name, is_ensemble, weights = ensembler.evaluate_candidates(candidate_preds, y_val.values)
        self.ensembler = ensembler
        self.champion_name = champion_name
        self.is_ensemble = is_ensemble
        self.ensemble_weights = weights

        print("\n" + "=" * 75)
        print("AEROCAST-AI CANDIDATE MODEL LEADERBOARD (Tuning Validation Split)")
        print(f"Normalization Basis: {ensembler.norm_basis}")
        print("=" * 75)
        print(f"{'Model Architecture':<32} | {'NMAE (%)':<10} | {'RMSE (MW)':<10} | {'Pinball':<10} | {'PICP (%)'}")
        print("-" * 75)
        for row in ensembler.comparison_table:
            print(f"{row['model']:<32} | {row['NMAE_%']:<10.2f} | {row['RMSE']:<10.2f} | {row['Mean_Pinball']:<10.3f} | {row['Coverage_PICP_%']:<10.1f}%")
        print("-" * 75)
        print(f"[*] SELECTION DECISION: Champion is '{champion_name}'")
        if is_ensemble:
            print(f"    Validation improvement verified. Weights: {weights}")
        else:
            print(f"    Standalone champion retained. Forced ensembling rejected (insufficient gain).")

        # 7. Curtailment Sensitivity Experiment
        if "flag_suspicious_outage" in X_train.columns:
            outage_cnt = int(X_train["flag_suspicious_outage"].sum())
            if outage_cnt > 0:
                print("\n[+] Curtailment Sensitivity Experiment:")
                print(f"    - Identified {outage_cnt} ({outage_cnt / len(X_train) * 100:.2f}%) suspicious outage hours in training.")
                sample_weights = np.where(X_train["flag_suspicious_outage"] == 1, 0.25, 1.0)
                if HAS_LIGHTGBM:
                    import lightgbm as lgb
                    sens_model = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=100, verbose=-1, random_state=42)
                    sens_model.fit(X_train.select_dtypes(include=[np.number]), y_train, sample_weight=sample_weights)
                    sens_preds = np.maximum(0.0, sens_model.predict(X_val.select_dtypes(include=[np.number])))
                else:
                    from sklearn.ensemble import HistGradientBoostingRegressor
                    sens_model = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=100, random_state=42)
                    sens_model.fit(X_train.select_dtypes(include=[np.number]), y_train, sample_weight=sample_weights)
                    sens_preds = np.maximum(0.0, sens_model.predict(X_val.select_dtypes(include=[np.number])))
                sens_mae = float(np.mean(np.abs(y_val.values - sens_preds)))
                print(f"    - Sensitivity MAE (down-weighted outages): {sens_mae:.2f} MW")

        # Store model instances
        trained_models = {
            "Persistence_Baseline": pb,
            "LightGBM_Quantile": lgb_model,
            "Temporal_Quantile_GBM": temporal_gbm,
        }
        if HAS_CATBOOST:
            trained_models["CatBoost_Quantile"] = cb_model
        if self.active_path == "PATH_1_RESIDUAL":
            trained_models["AeroCast_Physics_Residual"] = res_model

        self.trained_models = trained_models
        self.champion_model = ensembler
        return {
            "leaderboard": ensembler.comparison_table,
            "champion_name": champion_name,
            "is_ensemble": is_ensemble,
            "weights": weights,
            "candidate_preds": candidate_preds
        }

    def refit_champion_on_full_training(self, df_full_train: pd.DataFrame, rated_capacity: Optional[float]):
        """
        Refits the selected champion model on 100% of the training data prior to final test inference.
        Maximizes sample efficiency while preserving the validated architecture and weights.
        """
        target_col = self.column_map["target_power"]
        X_full = df_full_train[self.feature_columns]
        y_full = df_full_train[target_col]

        print(f"\n[+] Refitting Champion '{self.champion_name}' on all {len(X_full):,} training samples...")
        models_to_fit = self.ensemble_weights.keys() if self.is_ensemble else [self.champion_name]

        for name in models_to_fit:
            if name == "LightGBM_Quantile":
                m = DirectQuantileForecaster(base_engine="lightgbm", rated_power_mw=rated_capacity, random_state=self.random_state)
                m.fit(X_full, y_full)
                self.trained_models[name] = m
            elif name == "CatBoost_Quantile" and HAS_CATBOOST:
                m = DirectQuantileForecaster(base_engine="catboost", rated_power_mw=rated_capacity, random_state=self.random_state)
                m.fit(X_full, y_full)
                self.trained_models[name] = m
            elif name == "Temporal_Quantile_GBM":
                m = TemporalQuantileGBM(random_state=self.random_state)
                m.fit(X_full, y_full)
                self.trained_models[name] = m
            elif name == "AeroCast_Physics_Residual" and "theoretical_power_physics" in df_full_train:
                m = PhysicsResidualForecaster(base_engine="lightgbm", rated_power_mw=rated_capacity, random_state=self.random_state)
                m.fit(X_full, y_full, df_full_train["theoretical_power_physics"])
                self.trained_models[name] = m
            elif name == "Persistence_Baseline":
                pb = PersistenceBaseline(diurnal_lag=self.config.forecast_horizon_hours)
                pb.fit(y_full.values)
                self.trained_models[name] = pb
        print("[OK] Champion refitted successfully on complete dataset.")

    def predict_test(
        self,
        df_test: pd.DataFrame,
        rated_capacity: Optional[float] = None,
        historical_df: Optional[pd.DataFrame] = None,
        col_p10: str = "P10",
        col_p50: str = "P50_optimal",
        col_p90: str = "P90"
    ) -> pd.DataFrame:
        """
        Processes separate test dataset without target labels and generates calibrated submission.
        Preserves original row order and identifiers.
        """
        print("\n[+] Generating Predictions for Test Dataset...")
        df_test_phys, _ = self.physics.transform(df_test, self.column_map)

        # Use historical training data to provide causal target history for test rows
        hist_data = historical_df if historical_df is not None else self.last_training_data
        df_test_full, _ = self.build_causal_features(
            df_test_phys, self.column_map, self.config.forecast_horizon_hours, historical_df=hist_data
        )

        # Enforce exact feature alignment
        for c in self.feature_columns:
            if c not in df_test_full.columns:
                df_test_full[c] = 0.0

        X_test = df_test_full[self.feature_columns]
        eff_ws_test = df_test_full.get("v_hub_physics", df_test_full.get(self.column_map.get("wind_speed", "")))
        ws_test_arr = eff_ws_test.values if eff_ws_test is not None else None

        # Predict using champion configuration
        if self.is_ensemble and self.ensembler:
            candidate_test_preds = {}
            for name in self.ensemble_weights.keys():
                m = self.trained_models[name]
                if name == "AeroCast_Physics_Residual":
                    candidate_test_preds[name] = m.predict(X_test, df_test_full["theoretical_power_physics"])
                elif name == "Persistence_Baseline":
                    candidate_test_preds[name] = m.predict(np.zeros(0), len(X_test))
                else:
                    candidate_test_preds[name] = m.predict(X_test)
            preds = self.ensembler.predict_champion(candidate_test_preds)
        elif self.champion_name in self.trained_models:
            m = self.trained_models[self.champion_name]
            if self.champion_name == "AeroCast_Physics_Residual":
                preds = m.predict(X_test, df_test_full["theoretical_power_physics"])
            elif self.champion_name == "Persistence_Baseline":
                preds = m.predict(np.zeros(0), len(X_test))
            else:
                preds = m.predict(X_test)
        elif hasattr(self.champion_model, "predict"):
            if self.active_path == "PATH_1_RESIDUAL" and "theoretical_power_physics" in df_test_full:
                preds = self.champion_model.predict(X_test, df_test_full["theoretical_power_physics"])
            else:
                preds = self.champion_model.predict(X_test)
        else:
            raise RuntimeError("Champion model has not been trained.")

        # Apply physical boundary clamps
        p10 = self.physics.enforce_guards(preds["P10"], ws_test_arr)
        p50 = self.physics.enforce_guards(preds["P50"], ws_test_arr)
        p90 = self.physics.enforce_guards(preds["P90"], ws_test_arr)
        p10, p50, p90 = enforce_quantile_monotonicity(p10, p50, p90)

        # Assemble submission preserving original test identifiers and order
        sub = pd.DataFrame(index=df_test.index)
        ts_col = self.column_map.get("timestamp")
        if ts_col and ts_col in df_test:
            sub[ts_col] = df_test[ts_col]
        farm_col = self.column_map.get("farm_id")
        if farm_col and farm_col in df_test:
            sub[farm_col] = df_test[farm_col]
        turb_col = self.column_map.get("turbine_id")
        if turb_col and turb_col in df_test:
            sub[turb_col] = df_test[turb_col]

        sub[col_p10] = np.round(p10, 3)
        sub[col_p50] = np.round(p50, 3)
        sub[col_p90] = np.round(p90, 3)
        if "predicted_power_mw" not in sub.columns:
            sub["predicted_power_mw"] = sub[col_p50]

        return sub
