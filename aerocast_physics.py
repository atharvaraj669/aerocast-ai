"""
================================================================================
AEROCAST-AI: DATA-DEPENDENT AERODYNAMIC PHYSICS ENGINE (AUDITED & REPAIRED)
AI SANSAD 2026 | Team Prompt Diplomats (ID: UYU672B3)
================================================================================

Core Identity: Physics-Informed AI with Probabilistic Bounds
Principle: Activate only physically valid calculations supported by verified
input fields and turbine metadata. Never fabricate turbine specifications or
substitute arbitrary constants as ground truth. Unknown specs remain None.
================================================================================
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


@dataclass
class TurbineSpecs:
    """
    Turbine physical specifications.
    Every unknown attribute strictly defaults to None without arbitrary fallback values.
    """
    rated_power_mw: Optional[float] = None
    hub_height_m: Optional[float] = None
    mast_height_m: Optional[float] = None
    v_cut_in_ms: Optional[float] = None
    v_rated_ms: Optional[float] = None
    v_cut_out_ms: Optional[float] = None
    shear_alpha: Optional[float] = None
    power_curve: Optional[Dict[float, float]] = None  # speed -> power mapping


@dataclass
class PhysicsActivationReport:
    """Audit log of which physical laws were activated based on data availability."""
    air_density_activated: bool = False
    air_density_rationale: str = ""
    vector_decomposition_activated: bool = False
    vector_decomposition_rationale: str = ""
    wind_shear_extrapolation_activated: bool = False
    wind_shear_rationale: str = ""
    theoretical_power_activated: bool = False
    theoretical_power_rationale: str = ""
    kinetic_flux_activated: bool = False
    kinetic_flux_rationale: str = ""
    physical_guards_activated: bool = False
    physical_guards_rationale: str = ""
    can_support_residual_path: bool = False

    def summary(self) -> str:
        lines = [
            "----------------------------------------------------------------------",
            "AEROCAST-AI PHYSICS ACTIVATION AUDIT REPORT",
            "----------------------------------------------------------------------",
            f"[*] Air Density (rho = P/RT): {'[ACTIVE]' if self.air_density_activated else '[DISABLED]'} -> {self.air_density_rationale}",
            f"[*] Vector Trig (u, v):       {'[ACTIVE]' if self.vector_decomposition_activated else '[DISABLED]'} -> {self.vector_decomposition_rationale}",
            f"[*] Wind Shear Extrapolation: {'[ACTIVE]' if self.wind_shear_extrapolation_activated else '[DISABLED]'} -> {self.wind_shear_rationale}",
            f"[*] Theoretical Power Curve:  {'[ACTIVE]' if self.theoretical_power_activated else '[DISABLED]'} -> {self.theoretical_power_rationale}",
            f"[*] Kinetic Energy Flux:      {'[ACTIVE]' if self.kinetic_flux_activated else '[DISABLED]'} -> {self.kinetic_flux_rationale}",
            f"[*] Physical Boundary Guards: {'[ACTIVE]' if self.physical_guards_activated else '[DISABLED]'} -> {self.physical_guards_rationale}",
            f"[*] Path 1 (Residual) Viable: {'[YES]' if self.can_support_residual_path else '[NO - Falling back to Path 2 Direct]'}",
            "----------------------------------------------------------------------"
        ]
        return "\n".join(lines)


class DataDependentPhysicsEngine:
    """
    Inspects available inputs and selectively activates physical transformations
    without ever inventing missing turbine specifications.
    """
    R_SPECIFIC = 287.058  # J/(kg·K) for dry air

    def __init__(self, specs: Optional[TurbineSpecs] = None):
        self.specs = specs or TurbineSpecs()
        self.report = PhysicsActivationReport()

    def audit_and_prepare(self, available_columns: Dict[str, Optional[str]]) -> PhysicsActivationReport:
        """
        Inspects available columns and specs to determine which physics equations
        can be computed legitimately. Never assumes default values for missing specs.
        """
        rep = PhysicsActivationReport()

        has_ws = bool(available_columns.get("wind_speed"))
        has_wd = bool(available_columns.get("wind_direction"))
        has_temp = bool(available_columns.get("temperature"))
        has_press = bool(available_columns.get("pressure"))

        # 1. Dynamic Air Density (Requires both temperature and barometric pressure)
        if has_temp and has_press:
            rep.air_density_activated = True
            rep.air_density_rationale = "Both temperature and barometric pressure supplied."
        elif has_temp:
            rep.air_density_activated = False
            rep.air_density_rationale = "Temperature present but surface pressure missing; refusing to fabricate atmospheric pressure."
        else:
            rep.air_density_activated = False
            rep.air_density_rationale = "Missing atmospheric temperature and pressure."

        # 2. Vector Decomposition (Requires both wind speed and wind direction)
        if has_ws and has_wd:
            rep.vector_decomposition_activated = True
            rep.vector_decomposition_rationale = "Both wind speed and direction available for continuous u/v mapping."
        else:
            rep.vector_decomposition_activated = False
            rep.vector_decomposition_rationale = "Directional wind vectors require both speed and direction."

        # 3. Wind Shear Extrapolation (Hellmann Power Law)
        # Strictly requires hub height, mast height, and shear alpha to be verified metadata.
        if (
            has_ws
            and self.specs.hub_height_m is not None
            and self.specs.mast_height_m is not None
            and self.specs.hub_height_m > self.specs.mast_height_m
            and self.specs.shear_alpha is not None
        ):
            rep.wind_shear_extrapolation_activated = True
            rep.wind_shear_rationale = (
                f"Mast ({self.specs.mast_height_m}m), Hub ({self.specs.hub_height_m}m), "
                f"and shear alpha={self.specs.shear_alpha:.3f} verified."
            )
        else:
            rep.wind_shear_extrapolation_activated = False
            missing_parts = []
            if self.specs.hub_height_m is None:
                missing_parts.append("hub_height")
            if self.specs.mast_height_m is None:
                missing_parts.append("mast_height")
            if self.specs.shear_alpha is None:
                missing_parts.append("shear_alpha")
            rep.wind_shear_rationale = (
                f"Wind shear extrapolation disabled: unverified metadata ({', '.join(missing_parts) if missing_parts else 'no wind speed'}). "
                f"Using measured wind directly."
            )

        # 4. Theoretical Power Curve
        # Requires either custom power curve dict OR verified rated power, cut-in, rated, and cut-out speeds.
        has_custom_curve = bool(self.specs.power_curve)
        has_standard_specs = (
            self.specs.rated_power_mw is not None
            and self.specs.v_cut_in_ms is not None
            and self.specs.v_rated_ms is not None
            and self.specs.v_cut_out_ms is not None
        )
        if has_ws and has_custom_curve:
            rep.theoretical_power_activated = True
            rep.theoretical_power_rationale = "Verified empirical/manufacturer turbine power curve available."
        elif has_ws and has_standard_specs:
            rep.theoretical_power_activated = True
            rep.theoretical_power_rationale = (
                f"Aerodynamic cubic profile verified (Cut-in: {self.specs.v_cut_in_ms}m/s, "
                f"Rated: {self.specs.v_rated_ms}m/s, Cut-out: {self.specs.v_cut_out_ms}m/s, "
                f"Capacity: {self.specs.rated_power_mw}MW)."
            )
        else:
            rep.theoretical_power_activated = False
            missing_specs = []
            if self.specs.rated_power_mw is None:
                missing_specs.append("rated_power")
            if self.specs.v_cut_in_ms is None:
                missing_specs.append("cut_in_speed")
            if self.specs.v_rated_ms is None:
                missing_specs.append("rated_speed")
            if self.specs.v_cut_out_ms is None:
                missing_specs.append("cut_out_speed")
            rep.theoretical_power_rationale = (
                f"Theoretical power disabled: missing specifications ({', '.join(missing_specs)}). "
                f"Refusing to fabricate turbine power curve."
            )

        # 5. Kinetic Energy Flux Index
        if has_ws:
            rep.kinetic_flux_activated = True
            rep.kinetic_flux_rationale = "Calculated as velocity kinetic flux (v^3), density-weighted if available."
        else:
            rep.kinetic_flux_activated = False
            rep.kinetic_flux_rationale = "Requires wind speed."

        # 6. Physical Boundary Guards
        guards_active = []
        guards_active.append("Non-negative power (P >= 0) enforced unconditionally")
        if self.specs.rated_power_mw is not None:
            guards_active.append(f"Nameplate capacity ceiling (P <= {self.specs.rated_power_mw} MW) enforced")
        else:
            guards_active.append("Upper bound ceiling not enforced (rated capacity unknown)")
        if self.specs.v_cut_out_ms is not None:
            guards_active.append(f"Storm cut-out shutdown (P = 0 if v > {self.specs.v_cut_out_ms} m/s) enforced")
        else:
            guards_active.append("Storm cut-out shutdown not enforced (cut-out speed unknown)")

        rep.physical_guards_activated = True
        rep.physical_guards_rationale = "; ".join(guards_active) + "."

        # Can support Path 1 (Physics Residual)?
        # Path 1 is valid ONLY if theoretical power can genuinely be calculated without guessing.
        rep.can_support_residual_path = rep.theoretical_power_activated

        self.report = rep
        return rep

    def compute_air_density(self, temp_series: pd.Series, press_series: pd.Series) -> pd.Series:
        """Computes dynamic air density (kg/m³) using the Ideal Gas Law: rho = P / (R * T_kelvin)."""
        p = press_series.copy()
        # Detect if pressure is in hPa / mbar vs Pa
        mean_p = p.dropna().mean()
        if 800.0 <= mean_p <= 1200.0:
            p = p * 100.0  # convert hPa to Pa
        t_k = temp_series + 273.15
        rho = p / (self.R_SPECIFIC * t_k)
        return rho.clip(lower=0.8, upper=1.6)

    def extrapolate_hub_wind(self, ws_series: pd.Series) -> pd.Series:
        """Applies Hellmann Wind Shear Law only if verified heights and shear alpha are set."""
        if not self.report.wind_shear_extrapolation_activated:
            return ws_series
        ratio = (self.specs.hub_height_m / self.specs.mast_height_m) ** self.specs.shear_alpha
        return ws_series * ratio

    def decompose_vectors(self, ws_series: pd.Series, wd_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Orthogonal U (zonal) and V (meridional) vector decomposition."""
        rad = np.radians(wd_series)
        u = ws_series * np.cos(rad)
        v = ws_series * np.sin(rad)
        return u, v

    def calculate_theoretical_power(self, effective_ws: pd.Series) -> pd.Series:
        """
        Computes theoretical aerodynamic generation from known curve or cubic aerodynamic curve.
        Returns Series of NaNs if theoretical power is not activated.
        """
        if not self.report.theoretical_power_activated:
            return pd.Series(np.nan, index=effective_ws.index)

        # Case A: Explicit empirical or manufacturer curve table
        if self.specs.power_curve:
            sorted_speeds = sorted(self.specs.power_curve.keys())
            sorted_powers = [self.specs.power_curve[s] for s in sorted_speeds]
            theo = np.interp(effective_ws.values, sorted_speeds, sorted_powers, left=0.0, right=0.0)
            return pd.Series(theo, index=effective_ws.index)

        # Case B: Standard aerodynamic cubic profile with verified cut-in/rated/cut-out
        v = effective_ws.values
        v_in = self.specs.v_cut_in_ms
        v_rat = self.specs.v_rated_ms
        v_out = self.specs.v_cut_out_ms
        p_rat = self.specs.rated_power_mw

        theo = np.zeros(len(v), dtype=float)
        ramp_mask = (v >= v_in) & (v <= v_rat)
        plateau_mask = (v > v_rat) & (v <= v_out)

        ratio = np.clip((v[ramp_mask] - v_in) / (v_rat - v_in), 0.0, 1.0)
        theo[ramp_mask] = (ratio ** 3) * p_rat
        theo[plateau_mask] = p_rat
        # Above cut-out or below cut-in remains 0.0
        return pd.Series(theo, index=effective_ws.index)

    def transform(self, df: pd.DataFrame, col_map: Dict[str, Optional[str]]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Transforms dataframe by adding only physically valid columns.
        Returns the modified dataframe and the list of newly created physics feature names.
        """
        self.audit_and_prepare(col_map)
        out = df.copy()
        new_features: List[str] = []

        ws_col = col_map.get("wind_speed")
        wd_col = col_map.get("wind_direction")
        temp_col = col_map.get("temperature")
        press_col = col_map.get("pressure")

        # Effective Wind Speed (Extrapolated if justified, otherwise raw)
        if ws_col and ws_col in out:
            raw_ws = out[ws_col]
            if self.report.wind_shear_extrapolation_activated:
                out["v_hub_physics"] = self.extrapolate_hub_wind(raw_ws)
                new_features.append("v_hub_physics")
                effective_ws = out["v_hub_physics"]
            else:
                effective_ws = raw_ws

            # Kinetic flux index
            if self.report.kinetic_flux_activated:
                out["kinetic_flux_v3"] = effective_ws ** 3
                new_features.append("kinetic_flux_v3")
        else:
            effective_ws = None

        # Air Density
        if self.report.air_density_activated and temp_col and press_col:
            out["air_density_rho"] = self.compute_air_density(out[temp_col], out[press_col])
            new_features.append("air_density_rho")
            if effective_ws is not None:
                out["kinetic_flux_density_adjusted"] = 0.5 * out["air_density_rho"] * (effective_ws ** 3)
                new_features.append("kinetic_flux_density_adjusted")

        # Vector Decomposition
        if self.report.vector_decomposition_activated and effective_ws is not None and wd_col:
            u, v = self.decompose_vectors(effective_ws, out[wd_col])
            out["wind_vector_u"] = u
            out["wind_vector_v"] = v
            new_features.extend(["wind_vector_u", "wind_vector_v"])

        # Theoretical Power
        if self.report.theoretical_power_activated and effective_ws is not None:
            out["theoretical_power_physics"] = self.calculate_theoretical_power(effective_ws)
            new_features.append("theoretical_power_physics")

        return out, new_features

    def enforce_guards(self, predictions: np.ndarray, effective_ws: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Enforces verified physical boundaries:
        - Output >= 0.0 (non-negative power enforced unconditionally).
        - Output <= rated_capacity (ONLY if rated_capacity is verified).
        - Cut-out zeroing (ONLY if cut-out speed and effective wind speed are verified).
        """
        preds = np.asarray(predictions).copy()
        # Strictly non-negative
        preds = np.maximum(preds, 0.0)

        # Upper bound ONLY if rated capacity is known and verified
        if self.specs.rated_power_mw is not None:
            preds = np.minimum(preds, self.specs.rated_power_mw)

        # Storm shutdown clamp ONLY if cut-out speed is verified
        if effective_ws is not None and self.specs.v_cut_out_ms is not None:
            cut_out_mask = effective_ws > self.specs.v_cut_out_ms
            preds[cut_out_mask] = 0.0

        return preds
