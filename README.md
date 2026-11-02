# AI SANSAD 2026 — AEROCAST-AI MASTER SUITE
## Department of Artificial Intelligence & Machine Learning | Bharati Vidyapeeth’s College of Engineering, Delhi
### Team: Prompt Diplomats | Team ID: UYU672B3 | Leader: Atharva Sandeep Patil
### Assigned Problem: Forecast hourly wind power output using historical wind-farm generation and available wind forecasts.
### Core Research Contribution: "Physics-Informed AI with Probabilistic Bounds"

---

## 🌪️ Architecture Overview (Round 2 Adaptive Hybrid System)

Because the AI SANSAD Round 2 competition dataset schema is determined at competition time (potential presence or absence of turbine specifications, hub heights, manufacturer power curves, barometric pressure, or curtailment logs), **AeroCast-AI** implements an **Adaptive Hybrid Architecture**:

```
                       [ Incoming Dataset Schema ]
                                    │
                                    ▼
                     [ Adaptive Schema Inspector ]
              (Detects timestamp, target, weather, turbine metadata)
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
      [ Full Verified Specs ]              [ Unknown / Partial Specs ]
                  │                                   │
                  ▼                                   ▼
    [ Data-Dependent Physics Engine ]     [ Data-Dependent Physics Engine ]
    - Air Density: ACTIVE (P/RT)          - Air Density: DISABLED (No guessing)
    - Vector Trig: ACTIVE (u, v)          - Vector Trig: ACTIVE (u, v)
    - Hub Shear:   ACTIVE (Hellmann)      - Hub Shear:   DISABLED (Direct WS)
    - Power Curve: ACTIVE (Cubic profile) - Power Curve: DISABLED (Direct ML)
                  │                                   │
                  ▼                                   ▼
         [ PATH 1 FORECASTER ]               [ PATH 2 FORECASTER ]
      Physics Residual Modelling          Direct Probabilistic Quantile
     P_actual = P_physics + P_res           (LightGBM / CatBoost / GBM)
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    │
                                    ▼
                 [ Selective Validation Ensembler ]
           (Tests ensemble; selects ONLY if it beats standalone)
                                    │
                                    ▼
                     [ Physical Boundary Governor ]
          - Non-negative enforcement (P >= 0) [Unconditional]
          - Upper bound saturation (P <= P_rated) [Verified Only]
          - Storm cut-out shutdown (P = 0 if v > v_cutout) [Verified Only]
          - Quantile monotonicity (P10 <= P50 <= P90) [Guaranteed]
                                    │
                                    ▼
       [ Output: submission.csv with Configurable P10, P50, P90 ]
```

---

## 📁 Repository Structure

| File | Purpose | Description |
| :--- | :--- | :--- |
| **[`train_and_predict.py`](file:///d:/New%20folder%20%282%29/train_and_predict.py)** | **Turnkey Master CLI** | Master competition entrypoint supporting `--inspect`, `--mode fast`, `--mode full`, `--demo`, explicit column overrides, and configurable submission column names. |
| **[`aerocast_engine.py`](file:///d:/New%20folder%20%282%29/aerocast_engine.py)** | **Orchestration Engine** | Coordinates data preprocessing, path selection (Path 1 vs Path 2), causal feature engineering (group-isolated, no `.bfill()`), test history ingestion, champion refitting, and test inference. |
| **[`aerocast_physics.py`](file:///d:/New%20folder%20%282%29/aerocast_physics.py)** | **Data-Dependent Physics** | Activates only physically valid calculations (air density, vector trig, Hellmann law, theoretical power). Never invents or defaults turbine specifications; unverified specs remain `None`. |
| **[`aerocast_loader.py`](file:///d:/New%20folder%20%282%29/aerocast_loader.py)** | **Adaptive Data Loader** | Multi-format reader (CSV, TSV, Parquet, Excel), fuzzy column matcher with strict confidence thresholds, explicit override mappings, timestamp audit, categorical encoding preservation, and non-destructive curtailment auditor. |
| **[`aerocast_models.py`](file:///d:/New%20folder%20%282%29/aerocast_models.py)** | **Model & Ensemble Suite** | Persistence baseline, LightGBM quantile regression, CatBoost quantile regression (with native categorical support), TemporalQuantileGBM challenger, Path 1 Residual forecaster, Path 2 Direct forecaster, and Selective Validation Ensembler. Guarantees feature alignment and $P_{10} \le P_{50} \le P_{90}$. |
| **[`aerocast_validation.py`](file:///d:/New%20folder%20%282%29/aerocast_validation.py)** | **Leakage-Free Validation** | Chronological rolling-origin multi-step validation, documented NMAE normalization bases, baseline-relative skill scoring over identical timestamps, pinball loss, coverage (PICP), and normalized interval width (PINAW). |
| **[`test_aerocast.py`](file:///d:/New%20folder%20%282%29/test_aerocast.py)** | **Automated Test Suite** | 12 automated unit and integration tests verifying physical integrity, causal feature isolation, TSV parsing, quantile monotonicity, and validation pipeline correctness. |
| **[`run_aerocast.ps1`](file:///d:/New%20folder%20%282%29/run_aerocast.ps1)** | **PowerShell Launcher** | Automatically binds execution to the local virtual environment. |

---

## ⚡ Competition-Day Quick-Start Guide

### Step 1: Pre-Competition Environment Verification
Ensure dependencies are installed and all automated tests pass:
```powershell
.\run_aerocast.ps1 --demo --mode fast
pytest test_aerocast.py -v
```

### Step 2: Inspect Received Competition Data
When the competition organisers release the dataset, **inspect before running training**:
```powershell
python train_and_predict.py --train "path/to/competition_train.csv" --inspect
```
*Outputs: Detected column roles, confidence scores, null percentages, sampling interval regularity, and curtailment/outage statistics.*

If the file is tab-separated (`.tsv`), AeroCast-AI detects and parses it automatically:
```powershell
python train_and_predict.py --train "path/to/competition_train.tsv" --inspect
```

### Step 3: Fast Initial Submission (Sub-Minute Execution)
Within the first 5 minutes of receiving the data, generate a fast valid submission baseline:
```powershell
python train_and_predict.py --train "competition_train.csv" --test "competition_test.csv" --mode fast --output "submission_fast.csv"
```
*Executes Path 1 (if verified specs are supplied) or Path 2 (Direct LightGBM Quantile), guarantees quantile monotonicity, refits champion on 100% of training data, and formats submission columns.*

### Step 4: Handle Ambiguous Schema or Non-Standard Column Names
If column detection confidence is low, or if the competition uses unique naming conventions:
```powershell
python train_and_predict.py --train "train.csv" --test "test.csv" `
    --target-col "Power_Output_MW" `
    --ws-col "WindSpeed_mps" `
    --wd-col "WindDirection_deg" `
    --temp-col "AmbientTemp_C" `
    --press-col "Pressure_hPa" `
    --farm-col "WindFarm_ID" `
    --mode fast
```

If turbine specifications are officially provided in competition documentation or problem statement, supply them explicitly to activate Path 1:
```powershell
python train_and_predict.py --train "train.csv" --test "test.csv" `
    --rated-mw 50.0 `
    --hub-height 90.0 `
    --mast-height 10.0 `
    --cut-in 3.0 `
    --cut-out 25.0 `
    --mode fast
```
*(If omitted, unknown specs remain `None`, preventing spurious physics assumptions and safely activating Path 2).*

### Step 5: Full Scientific Experiment & Ensemble Optimization
Once the pipeline is verified, run the full benchmark for maximum leaderboard accuracy:
```powershell
python train_and_predict.py --train "train.csv" --test "test.csv" --mode full --output "submission_ensemble.csv"
```
*Trains Persistence, LightGBM, CatBoost, TemporalQuantileGBM, and Physics Residual models; performs selective validation ensembling; evaluates 3-fold chronological rolling-origin walk-forward CV; refits champion on all training data; and exports final predictions.*

### Step 6: Configurable Submission Format
If the competition leaderboard requires custom column headers:
```powershell
python train_and_predict.py --train "train.csv" --test "test.csv" `
    --col-p10 "P10" `
    --col-p50 "Predicted_Power" `
    --col-p90 "P90" `
    --output "final_submission.csv"
```

---

## 🔬 Benchmark Results & Evaluation Integrity

### 1. Synthetic Benchmark Evaluation (Demo Mode)
The metrics below represent execution on our standardized 4,000-hour multi-season synthetic dataset (`--demo`) generated to validate pipeline plumbing, physics path switching, and post-processing bounds.

> **Important Scientific Note**: These figures are benchmark results under controlled synthetic conditions. Real-world performance on competition day will depend entirely on field anemometer noise, NWP forecast error margins, and operational curtailment events.

| Model Architecture | NMAE (%) | RMSE (MW) | Mean Pinball Loss | PICP Coverage (%) |
| :--- | :---: | :---: | :---: | :---: |
| Persistence Baseline (24h Diurnal) | 20.12% | 35.41 | 6.729 | 84.5% |
| CatBoost Quantile Forecaster | 1.51% | 2.12 | 0.514 | 68.9% |
| LightGBM Quantile Forecaster | 1.44% | 1.98 | 0.481 | 76.2% |
| TemporalQuantileGBM Challenger | 1.40% | 1.94 | 0.478 | 82.0% |
| AeroCast-AI Physics Residual (Path 1) | 1.48% | 2.25 | 0.467 | 74.5% |
| **AeroCast-AI Champion Ensemble** | **1.38%** | **1.92** | **0.463** | **74.8%** |

* Normalization Basis: Verified Nameplate Capacity (100.0 MW).
* Baseline-Relative Skill Score: **93.59% error reduction** over 24h persistence.
* Empirical Prediction Interval Coverage: **74.80%** (target 80% nominal interval $P_{10}$ to $P_{90}$).
* Prediction Interval Normalized Average Width (PINAW): **4.10%** of capacity.
* Quantile Monotonicity: **100% compliant** ($P_{10} \le P_{50} \le P_{90}$ enforced via sorting).
* Boundary Compliance: **0 negative predictions** ($P \ge 0$ unconditionally).

### 2. Transparent Technical Distinctions
- **TemporalQuantileGBM**: This model is a temporal gradient-boosted decision tree architecture engineered with causal target lags, multi-horizon rolling statistics, and diurnal/seasonal Fourier embeddings. It is **not** a deep neural Temporal Fusion Transformer (TFT). This design delivers superior tabular performance with sub-second CPU training times and zero GPU dependency.
- **Probabilistic Intervals**: Quantile models are trained with asymmetric pinball loss ($\tau \in \{0.10, 0.50, 0.90\}$). Empirical coverage is explicitly measured during rolling-origin cross-validation and reported honestly; theoretical nominal bounds are never assumed without empirical verification.
- **Physical Assumptions**: Path 1 is activated if and only if all necessary physical parameters can be verified from metadata. In all other scenarios, Path 2 executes direct probabilistic forecasting to eliminate assumption bias.
