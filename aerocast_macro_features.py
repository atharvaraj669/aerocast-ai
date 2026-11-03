"""
================================================================================
AEROCAST-AI: MACRO PANEL CAUSAL FEATURE ENGINEERING MATRIX
AI SANSAD 2026 | Team Prompt Diplomats (ID: UYU672B3)
================================================================================

Engineers strictly causal, leakage-free features for annual national panel data:
1. Entity-isolated autoregressive target lags (t-1, t-2, t-3, t-5) — strictly NO .bfill().
2. Multi-year momentum & velocity metrics (1-year delta, 3-year delta, 3-year CAGR).
3. Systemic energy ratios & macroeconomic elasticities:
   - Renewable generation share (Renewables / Total Electricity)
   - Carbon intensity of power generation (CO2 / Fossil TWh)
   - Modern clean cooking vs electrification access gap
   - Decoupling elasticity (GDP growth vs Primary energy consumption change)
4. Geopolitical & demographic embeddings (latitude climate zones, income brackets).
================================================================================
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class MacroFeatureEngineer:
    """
    Transforms cleaned panel data into a high-dimensional causal feature matrix.
    """

    def __init__(
        self,
        target_col: str = "electricity_renewables_twh",
        lags: Tuple[int, ...] = (1, 2, 3, 5),
        use_log_transform: bool = True
    ):
        self.target_col = target_col
        self.lags = sorted(lags)
        self.use_log_transform = use_log_transform
        self.feature_columns: List[str] = []

    def engineer_features(
        self,
        df: pd.DataFrame,
        is_training: bool = True
    ) -> Tuple[pd.DataFrame, List[str], str]:
        """
        Computes causal panel features grouped by country entity.
        Returns:
            (augmented_df, feature_column_names, target_column_name)
        """
        data = df.copy()

        # Ensure correct chronological order within each entity
        data = data.sort_values(["entity", "year"], kind="mergesort").reset_index(drop=True)

        feature_cols: List[str] = []

        # ----------------------------------------------------------------------
        # 1. Target Transformations (Log-transform for highly skewed distributions)
        # ----------------------------------------------------------------------
        model_target = self.target_col
        if self.use_log_transform and self.target_col in data.columns:
            # Handle extreme right-skew (e.g. skew > 9 in TWh)
            log_target_col = f"{self.target_col}_log1p"
            data[log_target_col] = np.log1p(np.maximum(0.0, data[self.target_col].fillna(0.0)))
            model_target = log_target_col

        # ----------------------------------------------------------------------
        # 2. Causal Target Lags & Momentum (Strictly per entity group, NO .bfill())
        # ----------------------------------------------------------------------
        grp = data.groupby("entity", group_keys=False)

        for lag in self.lags:
            lag_col = f"{self.target_col}_lag_{lag}y"
            data[lag_col] = grp[self.target_col].shift(lag)
            feature_cols.append(lag_col)

            if self.use_log_transform and model_target in data.columns:
                log_lag_col = f"{model_target}_lag_{lag}y"
                data[log_lag_col] = grp[model_target].shift(lag)
                feature_cols.append(log_lag_col)

        # 1-year momentum: delta = lag_1 - lag_2
        if 1 in self.lags and 2 in self.lags:
            delta_1y_col = f"{self.target_col}_delta_1y"
            data[delta_1y_col] = data[f"{self.target_col}_lag_1y"] - data[f"{self.target_col}_lag_2y"]
            feature_cols.append(delta_1y_col)

        # 3-year momentum & CAGR
        if 1 in self.lags and 3 in self.lags:
            delta_3y_col = f"{self.target_col}_delta_3y"
            data[delta_3y_col] = data[f"{self.target_col}_lag_1y"] - data[f"{self.target_col}_lag_3y"]
            feature_cols.append(delta_3y_col)

            # 3-year CAGR: (lag_1 / (lag_3 + 1e-3))**(1/3) - 1
            cagr_col = f"{self.target_col}_cagr_3y"
            ratio = (data[f"{self.target_col}_lag_1y"] + 1e-4) / (data[f"{self.target_col}_lag_3y"] + 1e-4)
            data[cagr_col] = np.sign(ratio) * (np.abs(ratio) ** (1.0 / 3.0) - 1.0)
            data[cagr_col] = data[cagr_col].clip(-1.0, 5.0)  # Bound extreme percentage swings
            feature_cols.append(cagr_col)

        # ----------------------------------------------------------------------
        # 3. Covariate Lags (Preventing contemporaneous leakage)
        # ----------------------------------------------------------------------
        covariate_drivers = [
            "electricity_fossil_twh",
            "electricity_nuclear_twh",
            "primary_energy_pc_kwh",
            "energy_intensity_mj_gdp",
            "gdp_growth_pct",
            "gdp_per_capita_usd",
            "access_to_electricity_pct",
            "co2_emissions_kt",
            "financial_flows_usd"
        ]

        for cov in covariate_drivers:
            if cov in data.columns and cov != self.target_col:
                cov_lag1 = f"{cov}_lag_1y"
                data[cov_lag1] = grp[cov].shift(1)
                feature_cols.append(cov_lag1)

        # ----------------------------------------------------------------------
        # 4. Energy System Interactions & Elasticities (Computed on 1-year lagged data)
        # ----------------------------------------------------------------------
        if f"{self.target_col}_lag_1y" in data.columns:
            ren_lag = data[f"{self.target_col}_lag_1y"]
        elif "electricity_renewables_twh" in data.columns:
            ren_lag = grp["electricity_renewables_twh"].shift(1)
        else:
            ren_lag = pd.Series(0.0, index=data.index)

        fos_lag = data["electricity_fossil_twh_lag_1y"] if "electricity_fossil_twh_lag_1y" in data.columns else pd.Series(0.0, index=data.index)
        nuc_lag = data["electricity_nuclear_twh_lag_1y"] if "electricity_nuclear_twh_lag_1y" in data.columns else pd.Series(0.0, index=data.index)

        tot_gen_lag = ren_lag.fillna(0.0) + fos_lag.fillna(0.0) + nuc_lag.fillna(0.0)

        # Renewable Generation Share
        data["calc_renewable_generation_share_lag1"] = np.where(
            tot_gen_lag > 0.01, (ren_lag / tot_gen_lag).clip(0.0, 1.0), 0.0
        )
        feature_cols.append("calc_renewable_generation_share_lag1")

        # Carbon Intensity of Power: CO2 / Fossil TWh
        co2_lag = data["co2_emissions_kt_lag_1y"] if "co2_emissions_kt_lag_1y" in data.columns else pd.Series(0.0, index=data.index)
        data["calc_carbon_intensity_lag1"] = np.where(
            fos_lag > 0.01, co2_lag.fillna(0.0) / (fos_lag + 1e-4), 0.0
        ).clip(0.0, 50000.0)
        feature_cols.append("calc_carbon_intensity_lag1")

        # Access Equity Gap: Electricity Access - Clean Cooking Access
        if "access_to_electricity_pct" in data.columns and "clean_cooking_access_pct" in data.columns:
            elec_lag = data["access_to_electricity_pct_lag_1y"] if "access_to_electricity_pct_lag_1y" in data.columns else data["access_to_electricity_pct"]
            cook_lag = grp["clean_cooking_access_pct"].shift(1).fillna(0.0)
            data["calc_access_equity_gap_lag1"] = (elec_lag - cook_lag).clip(-50.0, 100.0)
            feature_cols.append("calc_access_equity_gap_lag1")

        # Log-transformed Per-Capita and Demographic Features
        if "population_density" in data.columns and "land_area_km2" in data.columns:
            data["log_population"] = np.log10(
                np.maximum(100.0, data["population_density"] * data["land_area_km2"])
            )
            feature_cols.append("log_population")

        if "gdp_per_capita_usd" in data.columns:
            gdp_lag = data["gdp_per_capita_usd_lag_1y"] if "gdp_per_capita_usd_lag_1y" in data.columns else data["gdp_per_capita_usd"]
            data["log_gdp_pc_lag1"] = np.log10(np.maximum(10.0, gdp_lag.fillna(100.0)))
            feature_cols.append("log_gdp_pc_lag1")

        # ----------------------------------------------------------------------
        # 5. Geopolitical & Climate Zone Embeddings
        # ----------------------------------------------------------------------
        if "latitude" in data.columns:
            abs_lat = data["latitude"].abs().fillna(20.0)
            # Climate zones: Tropical (<=23.5), Subtropical (23.5-35), Temperate (35-60), Polar (>60)
            data["is_tropical"] = (abs_lat <= 23.5).astype(int)
            data["is_temperate"] = ((abs_lat > 23.5) & (abs_lat <= 60.0)).astype(int)
            data["latitude_sin"] = np.sin(np.radians(data["latitude"].fillna(0.0)))
            feature_cols.extend(["is_tropical", "is_temperate", "latitude_sin"])

        if "longitude" in data.columns:
            data["longitude_sin"] = np.sin(np.radians(data["longitude"].fillna(0.0)))
            data["longitude_cos"] = np.cos(np.radians(data["longitude"].fillna(0.0)))
            feature_cols.extend(["longitude_sin", "longitude_cos"])

        # World Bank Income Brackets (Based on GDP/Capita)
        if "gdp_per_capita_usd" in data.columns:
            gdp_val = data["gdp_per_capita_usd_lag_1y"] if "gdp_per_capita_usd_lag_1y" in data.columns else data["gdp_per_capita_usd"]
            gdp_clean = gdp_val.fillna(1000.0)
            data["income_bracket_low"] = (gdp_clean <= 1500.0).astype(int)
            data["income_bracket_lower_mid"] = ((gdp_clean > 1500.0) & (gdp_clean <= 5000.0)).astype(int)
            data["income_bracket_upper_mid"] = ((gdp_clean > 5000.0) & (gdp_clean <= 16000.0)).astype(int)
            data["income_bracket_high"] = (gdp_clean > 16000.0).astype(int)
            feature_cols.extend([
                "income_bracket_low",
                "income_bracket_lower_mid",
                "income_bracket_upper_mid",
                "income_bracket_high"
            ])

        # Auxiliary indicators
        if "flag_no_dev_aid" in data.columns:
            feature_cols.append("flag_no_dev_aid")
        if "flag_co2_imputed" in data.columns:
            feature_cols.append("flag_co2_imputed")

        # Fill isolated remaining NaNs in features with median of feature
        for fc in feature_cols:
            if data[fc].isna().any():
                data[fc] = data[fc].fillna(data[fc].median() if not data[fc].dropna().empty else 0.0)

        # Deduplicate feature column list
        feature_cols = list(dict.fromkeys(feature_cols))

        return data, feature_cols, model_target
