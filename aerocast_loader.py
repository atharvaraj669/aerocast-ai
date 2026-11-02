"""
================================================================================
AEROCAST-AI: ADAPTIVE COMPETITION DATA LOADER & SCHEMA INSPECTOR (REPAIRED)
AI SANSAD 2026 | Team Prompt Diplomats (ID: UYU672B3)
================================================================================

Handles completely unpredictable competition dataset schemas:
1. Multi-format ingestion (CSV, TSV, Parquet, Excel).
2. Explicit TSV support and robust file existence validation.
3. Column role classification with config override mapping and confidence guards.
4. Timestamp verification (sorting, duplicate detection, interval consistency).
5. Consistent categorical encoding between training and testing.
6. Non-destructive curtailment and zero-generation diagnostic auditing.
================================================================================
"""

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd


@dataclass
class ColumnRoleMatch:
    column_name: str
    role: str
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    score: float
    reason: str


@dataclass
class CurtailmentReport:
    total_records: int = 0
    zero_power_count: int = 0
    zero_power_pct: float = 0.0
    high_wind_zero_power_count: int = 0
    high_wind_zero_power_pct: float = 0.0
    flat_top_clipping_count: int = 0
    max_consecutive_zero_hours: int = 0
    curtailment_flag_column_found: bool = False
    curtailment_flag_column_name: Optional[str] = None
    audit_notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "----------------------------------------------------------------------",
            "AEROCAST-AI CURTAILMENT & OPERATIONAL EVENT AUDIT",
            "----------------------------------------------------------------------",
            f"[*] Total Records Inspected:          {self.total_records:,}",
            f"[*] Zero-Generation Records:           {self.zero_power_count:,} ({self.zero_power_pct:.2f}%)",
            f"[*] High-Wind Zero Generation (v>5m/s):{self.high_wind_zero_power_count:,} ({self.high_wind_zero_power_pct:.2f}%) [Suspicious Outage/Curtailment]",
            f"[*] Potential Flat-Top Derating:       {self.flat_top_clipping_count:,} occurrences",
            f"[*] Max Consecutive Zero Generation:   {self.max_consecutive_zero_hours} intervals",
            f"[*] Dedicated Curtailment/Status Flag: {'[FOUND: ' + str(self.curtailment_flag_column_name) + ']' if self.curtailment_flag_column_found else '[NOT PROVIDED - Relying on statistical diagnostic flags]'}",
            "----------------------------------------------------------------------"
        ]
        for note in self.audit_notes:
            lines.append(f"  > NOTE: {note}")
        return "\n".join(lines)


@dataclass
class DatasetConfig:
    """Explicit column overrides to eliminate guessing."""
    timestamp_col: Optional[str] = None
    target_col: Optional[str] = None
    wind_speed_col: Optional[str] = None
    wind_direction_col: Optional[str] = None
    temperature_col: Optional[str] = None
    pressure_col: Optional[str] = None
    farm_id_col: Optional[str] = None
    turbine_id_col: Optional[str] = None
    curtailment_col: Optional[str] = None
    rated_power_mw: Optional[float] = None
    hub_height_m: Optional[float] = None
    mast_height_m: Optional[float] = None
    v_cut_in_ms: Optional[float] = None
    v_rated_ms: Optional[float] = None
    v_cut_out_ms: Optional[float] = None
    shear_alpha: Optional[float] = None
    forecast_horizon_hours: int = 24


# Mapping between internal role keys and DatasetConfig field names
ROLE_TO_CONFIG_ATTR = {
    "timestamp": "timestamp_col",
    "target_power": "target_col",
    "wind_speed": "wind_speed_col",
    "wind_direction": "wind_direction_col",
    "temperature": "temperature_col",
    "pressure": "pressure_col",
    "farm_id": "farm_id_col",
    "turbine_id": "turbine_id_col",
    "curtailment": "curtailment_col",
}

# Priority regex patterns for column role resolution
ROLE_PATTERNS = {
    "timestamp": [
        (r"^(timestamp|datetime|time|date|valid_time|datetime_utc|ts)$", 1.0),
        (r"(timestamp|datetime|valid_time)", 0.85),
        (r"(time|date|period)", 0.65),
    ],
    "target_power": [
        (r"^(active_power|power_mw|power_kw|generation_mw|generation|target|actual_power|power|yield_mw)$", 1.0),
        (r"(active.*power|power.*mw|power.*kw|power.*output|target.*power|gen.*mw|actual.*power)", 0.90),
        (r"(target|power|generation|output|yield)", 0.60),
    ],
    "wind_speed": [
        (r"^(wind_speed|ws|wind_speed_10m|wind_speed_100m|v_10|v_hub|wind_spd|speed)$", 1.0),
        (r"(wind.*speed|wind.*spd|velocity|ws_\d+m?)", 0.90),
        (r"(speed|ws|spd)", 0.60),
    ],
    "wind_direction": [
        (r"^(wind_direction|wd|wind_dir|direction|bearing|wind_deg|deg)$", 1.0),
        (r"(wind.*dir|wind.*direction|wind.*bearing|wd_\d+m?)", 0.90),
        (r"(direction|bearing|deg|angle)", 0.60),
    ],
    "temperature": [
        (r"^(temperature|temp|ambient_temp|t_2m|temp_c|temperature_c)$", 1.0),
        (r"(ambient.*temp|air.*temp|temperature|temp)", 0.85),
        (r"(temp|t2m)", 0.60),
    ],
    "pressure": [
        (r"^(pressure|surface_pressure|baro_pressure|p_surface|press|pa|hpa)$", 1.0),
        (r"(surface.*press|baro.*press|pressure)", 0.85),
        (r"(press|baro|hpa|pa)", 0.60),
    ],
    "farm_id": [
        (r"^(farm_id|farm|wind_farm|site_id|location_id|plant_id)$", 1.0),
        (r"(farm.*id|site.*id|plant.*id|location)", 0.80),
    ],
    "turbine_id": [
        (r"^(turbine_id|turbine|wtg_id|unit_id|wtg)$", 1.0),
        (r"(turbine.*id|wtg.*id|unit.*id)", 0.80),
    ],
    "curtailment": [
        (r"^(curtailment|curtailed|derated|status|operating_status|is_available|availability|fault)$", 1.0),
        (r"(curtail|status|avail|derat|outage|fault)", 0.80),
    ],
}


class AdaptiveDataLoader:
    """
    Robust reader and inspector for competition datasets.
    """

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self.category_mappings: Dict[str, Dict[Any, int]] = {}

    def read_file(self, file_path: str) -> pd.DataFrame:
        """
        Reads multiple formats based on file extension.
        Explicitly supports TSV, CSV, Parquet, and Excel.
        Raises FileNotFoundError if file is missing.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at: '{file_path}'. Verify the file path.")

        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".tsv", ".tab"]:
            df = pd.read_csv(file_path, sep="\t", low_memory=False)
        elif ext in [".csv", ".txt"]:
            try:
                df = pd.read_csv(file_path, low_memory=False)
                # If only 1 column was parsed but lines have tabs or semicolons, re-parse
                if len(df.columns) == 1:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        first_line = f.readline()
                    if "\t" in first_line:
                        df = pd.read_csv(file_path, sep="\t", low_memory=False)
                    elif ";" in first_line:
                        df = pd.read_csv(file_path, sep=";", low_memory=False)
            except Exception:
                try:
                    df = pd.read_csv(file_path, sep="\t", low_memory=False)
                except Exception:
                    df = pd.read_csv(file_path, sep=";", low_memory=False)
        elif ext in [".parquet", ".pq"]:
            df = pd.read_parquet(file_path)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        elif ext == ".feather":
            df = pd.read_feather(file_path)
        else:
            df = pd.read_csv(file_path, low_memory=False)

        return df

    def detect_column_roles(self, df: pd.DataFrame, require_confident_target: bool = False) -> Dict[str, ColumnRoleMatch]:
        """
        Scores each column against candidate roles and resolves best assignments.
        Strictly respects user configuration overrides and validates them against dataset columns.
        """
        cols = list(df.columns)
        assigned: Dict[str, ColumnRoleMatch] = {}
        used_cols = set()

        # Step 1: Process explicit user overrides first
        for role, config_field in ROLE_TO_CONFIG_ATTR.items():
            override_val = getattr(self.config, config_field, None)
            if override_val is not None:
                if override_val in cols:
                    assigned[role] = ColumnRoleMatch(
                        column_name=override_val,
                        role=role,
                        confidence="HIGH (CONFIG OVERRIDE)",
                        score=1.0,
                        reason=f"Explicitly specified via config '{config_field}'"
                    )
                    used_cols.add(override_val)
                else:
                    raise ValueError(
                        f"Explicit column override '{override_val}' for role '{role}' "
                        f"was not found in dataset. Available columns: {cols}"
                    )

        # Step 2: Auto-detect unassigned roles
        target_candidates: List[Tuple[str, float]] = []

        for role, pattern_list in ROLE_PATTERNS.items():
            if role in assigned:
                continue

            best_match: Optional[ColumnRoleMatch] = None
            best_score = 0.0

            for col in cols:
                if col in used_cols:
                    continue
                clean_name = col.lower().strip().replace(" ", "_")
                for pat, weight in pattern_list:
                    if re.search(pat, clean_name):
                        type_bonus = 0.0
                        if role == "timestamp":
                            if pd.api.types.is_datetime64_any_dtype(df[col]) or "time" in clean_name or "date" in clean_name:
                                type_bonus = 0.1
                        elif role in ["target_power", "wind_speed", "wind_direction", "temperature", "pressure"]:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                type_bonus = 0.1

                        total_score = min(weight + type_bonus, 1.0)
                        if role == "target_power" and total_score >= 0.60:
                            target_candidates.append((col, total_score))

                        if total_score > best_score:
                            best_score = total_score
                            conf = "HIGH" if total_score >= 0.85 else ("MEDIUM" if total_score >= 0.65 else "LOW")
                            best_match = ColumnRoleMatch(
                                column_name=col,
                                role=role,
                                confidence=conf,
                                score=total_score,
                                reason=f"Pattern matched '{pat}' with score {total_score:.2f}"
                            )

            if best_match and best_match.score >= 0.60:
                assigned[role] = best_match
                used_cols.add(best_match.column_name)

        # Step 3: Guard against ambiguous target detection
        if "target_power" in assigned and require_confident_target:
            match = assigned["target_power"]
            if match.confidence == "LOW":
                raise ValueError(
                    f"Target column detection has low confidence for column '{match.column_name}' (score: {match.score:.2f}). "
                    f"Please provide an explicit target column override via '--target-col <column_name>'."
                )

        return assigned

    def verify_and_sort_timestamps(
        self, df: pd.DataFrame, ts_col: Optional[str], farm_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validates timestamps:
        - Parses to datetime.
        - Checks for NaT/invalid timestamps.
        - Detects duplicate timestamps within the same farm/group.
        - Sorts chronologically.
        """
        audit_info: Dict[str, Any] = {
            "has_timestamps": False,
            "null_timestamps": 0,
            "duplicate_timestamps": 0,
            "median_interval_minutes": None,
            "is_monotonic": True
        }

        if not ts_col or ts_col not in df:
            return df, audit_info

        out = df.copy()
        out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")
        null_count = int(out[ts_col].isna().sum())
        audit_info["has_timestamps"] = True
        audit_info["null_timestamps"] = null_count

        # Check duplicates
        dup_subset = [ts_col]
        if farm_col and farm_col in out:
            dup_subset.append(farm_col)
        dup_count = int(out.duplicated(subset=dup_subset).sum())
        audit_info["duplicate_timestamps"] = dup_count

        # Sort chronologically by [farm_col, ts_col] or ts_col
        sort_cols = [farm_col, ts_col] if (farm_col and farm_col in out) else [ts_col]
        out = out.sort_values(sort_cols).reset_index(drop=True)

        # Compute median interval
        valid_ts = out[ts_col].dropna().sort_values()
        if len(valid_ts) > 2:
            diffs = valid_ts.diff().dropna()
            diff_seconds = diffs.dt.total_seconds()
            pos_diffs = diff_seconds[diff_seconds > 0]
            if len(pos_diffs) > 0:
                audit_info["median_interval_minutes"] = float(pos_diffs.median() / 60.0)

        return out, audit_info

    def fit_and_encode_categories(
        self, df: pd.DataFrame, categorical_cols: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Fits categorical encoders on training data and preserves mappings for test consistency.
        """
        out = df.copy()
        encoded_cols = []
        for col in categorical_cols:
            if col in out:
                unique_vals = sorted(out[col].dropna().unique().tolist())
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                self.category_mappings[col] = mapping

                encoded_col_name = f"{col}_code"
                out[encoded_col_name] = out[col].map(mapping).fillna(-1).astype(int)
                encoded_cols.append(encoded_col_name)

        return out, encoded_cols

    def transform_categories(
        self, df: pd.DataFrame, categorical_cols: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Applies fitted categorical encodings to test data, mapping unseen categories to -1.
        """
        out = df.copy()
        encoded_cols = []
        for col in categorical_cols:
            if col in out and col in self.category_mappings:
                mapping = self.category_mappings[col]
                encoded_col_name = f"{col}_code"
                out[encoded_col_name] = out[col].map(mapping).fillna(-1).astype(int)
                encoded_cols.append(encoded_col_name)

        return out, encoded_cols

    def audit_curtailment_and_zero_generation(
        self, df: pd.DataFrame, target_col: Optional[str], ws_col: Optional[str], status_col: Optional[str]
    ) -> CurtailmentReport:
        """
        Audits zero-power and potential curtailment events without dropping rows.
        """
        rep = CurtailmentReport(total_records=len(df))
        if target_col is None or target_col not in df:
            rep.audit_notes.append("Target column missing; curtailment audit skipped.")
            return rep

        p = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0)
        rep.zero_power_count = int((p <= 1e-4).sum())
        rep.zero_power_pct = (rep.zero_power_count / len(df)) * 100.0 if len(df) > 0 else 0.0

        if ws_col and ws_col in df:
            ws = pd.to_numeric(df[ws_col], errors="coerce").fillna(0.0)
            high_wind_mask = (p <= 1e-4) & (ws >= 5.0)
            rep.high_wind_zero_power_count = int(high_wind_mask.sum())
            rep.high_wind_zero_power_pct = (rep.high_wind_zero_power_count / len(df)) * 100.0 if len(df) > 0 else 0.0

            diff_p = p.diff().abs()
            flat_mask = (diff_p < 0.01) & (p > 5.0) & (p < p.max() * 0.85)
            rep.flat_top_clipping_count = int(flat_mask.sum())

        is_zero = (p <= 1e-4).astype(int)
        consec = is_zero * (is_zero.groupby((is_zero != is_zero.shift()).cumsum()).cumcount() + 1)
        rep.max_consecutive_zero_hours = int(consec.max()) if len(consec) > 0 else 0

        if status_col and status_col in df:
            rep.curtailment_flag_column_found = True
            rep.curtailment_flag_column_name = status_col
            rep.audit_notes.append(f"Operating flag '{status_col}' will be leveraged in feature engineering.")
        else:
            rep.audit_notes.append(
                "Zero-generation and high-wind zero output preserved in dataset. "
                "Diagnostic flags will be added for sensitivity experiments."
            )

        return rep

    def inspect_dataset(self, data_or_path: Any, sample_rows: int = 5) -> Dict[str, Any]:
        """
        Produces comprehensive inspection report for the user from a file path or dataframe.
        """
        if isinstance(data_or_path, pd.DataFrame):
            df = data_or_path
            file_path = "In-Memory / Provided DataFrame"
        else:
            file_path = str(data_or_path)
            df = self.read_file(file_path)

        role_matches = self.detect_column_roles(df)
        target_col = role_matches.get("target_power").column_name if "target_power" in role_matches else None
        ws_col = role_matches.get("wind_speed").column_name if "wind_speed" in role_matches else None
        status_col = role_matches.get("curtailment").column_name if "curtailment" in role_matches else None
        ts_col = role_matches.get("timestamp").column_name if "timestamp" in role_matches else None
        farm_col = role_matches.get("farm_id").column_name if "farm_id" in role_matches else None

        df_sorted, ts_audit = self.verify_and_sort_timestamps(df, ts_col, farm_col)
        curtail_rep = self.audit_curtailment_and_zero_generation(df_sorted, target_col, ws_col, status_col)

        freq_str = f"~{ts_audit['median_interval_minutes']:.0f} minutes" if ts_audit.get("median_interval_minutes") else "Unknown"

        report = {
            "file_path": file_path,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "detected_roles": role_matches,
            "timestamp_audit": ts_audit,
            "detected_frequency": freq_str,
            "curtailment_audit": curtail_rep,
            "columns_info": [
                {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "missing_count": int(df[col].isna().sum()),
                    "missing_pct": float((df[col].isna().sum() / len(df)) * 100.0),
                    "sample_values": df[col].dropna().head(3).tolist()
                }
                for col in df.columns
            ],
            "sample_head": df.head(sample_rows)
        }
        return report

    def print_inspection_report(self, report: Dict[str, Any]):
        """Prints beautifully formatted audit report."""
        print("=" * 75)
        print("AEROCAST-AI ADAPTIVE DATASET INSPECTION & AVAILABILITY REPORT")
        print("=" * 75)
        print(f"File Path:       {report['file_path']}")
        print(f"Dimensions:      {report['total_rows']:,} rows x {report['total_columns']} columns")
        print(f"Sampling Rate:   {report['detected_frequency']}")

        ts_info = report.get("timestamp_audit", {})
        if ts_info.get("has_timestamps"):
            print(f"Timestamp Audit: Nulls: {ts_info.get('null_timestamps', 0)} | Duplicates: {ts_info.get('duplicate_timestamps', 0)}")

        print("\n[+] DETECTED COLUMN ROLES:")
        print(f"    {'Role':<18} | {'Assigned Column':<25} | {'Confidence':<20} | {'Score'}")
        print("    " + "-" * 73)
        for role, match in report["detected_roles"].items():
            print(f"    {role:<18} | {match.column_name:<25} | {match.confidence:<20} | {match.score:.2f}")

        print("\n[+] DATA SCHEMA & NULL AUDIT (First 15 Columns):")
        print(f"    {'Column Name':<28} | {'Type':<10} | {'Null Count':<12} | {'Null %'}")
        print("    " + "-" * 65)
        for item in report["columns_info"][:15]:
            print(f"    {item['name']:<28} | {item['dtype']:<10} | {item['missing_count']:<12} | {item['missing_pct']:.2f}%")
        if len(report["columns_info"]) > 15:
            print(f"    ... and {len(report['columns_info']) - 15} more columns")

        print("\n" + report["curtailment_audit"].summary())
        print("=" * 75)
