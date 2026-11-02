# AeroCast-AI v2.0 — macOS Event Day Setup Guide
**Team: Prompt Diplomats | Team ID: `UYU672B3` | Leader: Atharva Sandeep Patil**

> This guide assumes you have transferred the project folder to your MacBook Air M5
> via USB drive or GitHub clone. Run every command from the **project root directory**.

---

## Step 0 — Locate the Project & Open Terminal

1. Open **Terminal** (⌘ + Space → type "Terminal" → Enter)
2. Navigate to your project folder (adjust the path to match where you put it):

```bash
# If transferred via USB:
cd /Volumes/YourUSBName/aerocast-ai

# If cloned from GitHub:
cd ~/Desktop/aerocast-ai
```

Confirm you are in the right place:
```bash
ls
# Expected output should include:
# ml_pipeline/  data/  train_and_predict.py  requirements.txt  test_ml_pipeline.py
```

---

## Step 1 — Check Python Version

The pipeline requires Python 3.10 or newer. macOS Sonoma ships Python 3.12+.

```bash
python3 --version
# Expected: Python 3.12.x  (or 3.10+ is acceptable)
```

If Python is missing or older than 3.10:
```bash
# Install via Homebrew (run once):
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.12
```

---

## Step 2 — Create Virtual Environment

Always use an isolated virtual environment to avoid conflicts with system Python:

```bash
python3 -m venv .venv
```

> **Why not conda?** The `venv` approach is faster to activate and zero-configuration.
> LightGBM and CatBoost both ship pre-compiled ARM64 wheels for M-series chips via PyPI.

---

## Step 3 — Activate the Virtual Environment

```bash
source .venv/bin/activate
```

Your terminal prompt will change to show `(.venv)` at the start.  
**Every subsequent command in this guide must be run with the venv active.**

To verify the correct Python is active:
```bash
which python
# Expected: /path/to/aerocast-ai/.venv/bin/python
```

---

## Step 4 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Apple Silicon Note:** `lightgbm>=4.3.0` and `catboost>=1.2.0` both publish
> native `arm64` wheels. pip will select them automatically — no Rosetta or
> source compilation required on M1/M2/M3/M4/M5.

Expected output (last few lines):
```
Successfully installed lightgbm-4.x.x ... catboost-1.x.x ... pandas-2.x.x ...
```

---

## Step 5 — Verify Environment: Generate Dummy Data

Run this to confirm the entire path stack resolves correctly on macOS:

```bash
python3 ml_pipeline/generate_dummy_gefcom.py
```

Expected output:
```
===========================================================================
AEROCAST-AI v1.5: GENERATING SYNTHETIC GEFCOM 2012 DATASET
Target Directory: /Users/.../aerocast-ai/data/gefcom2012
Duration:         30 days (720 hourly timesteps)
===========================================================================
[OK] Generated 720 weather forecast records -> .../data/gefcom2012/windforecasts_wf1.csv
[OK] Generated 720 target power records     -> .../data/gefcom2012/train.csv
```

> If you see this output with **macOS-style forward-slash paths**, the path resolution
> is confirmed cross-platform correct.

---

## Step 6 — Run the Training Pipeline

```bash
python3 ml_pipeline/train.py
```

Expected final lines:
```
COMPARISON: PERSISTENCE BASELINE vs LIGHTGBM v2.0
  Persistence Baseline (24h Diurnal)    0.2778    0.3800
  LightGBM v2.0 (P50, +target_lag_24h) 0.0499    0.0704
  Skill Score (MAE reduction)           82.05%

[OK] Predictions exported to: '.../ml_pipeline/output/predictions.csv'
```

---

## Step 7 — Run the Evaluation Report

```bash
python3 ml_pipeline/evaluate.py
```

Expected output includes:
```
LIGHTGBM v2.0 POINT FORECAST METRICS (P50 Quantile):
    - Mean Absolute Error (MAE):           0.0499 (normalized power)
    - Skill Score vs Persistence:          82.04% MAE reduction
    - Prediction Interval Coverage (PICP): 66.90%  [Target: 80.00%]
    - Negative Generation Violations:      0  [PASS]
    - Quantile Monotonicity Violations:    0  [PASS]
```

---

## Step 8 — Run All Automated Tests

```bash
python3 -m pytest -v
```

Expected:
```
21 passed in X.XXs
```

If any test fails on macOS that passed on Windows, it is most likely a timestamp
parsing edge case. Run with `-s` for detailed output:
```bash
python3 -m pytest -v -s
```

---

## Step 9 — Run the Competition Pipeline (AeroCast-AI Core Engine)

Once you receive the real AI SANSAD dataset:

```bash
# 1. Inspect the competition dataset schema first (zero-guess, no training):
python3 train_and_predict.py --train /path/to/competition_train.csv --inspect

# 2. Fast submission (sub-minute):
python3 train_and_predict.py \
    --train /path/to/competition_train.csv \
    --test  /path/to/competition_test.csv  \
    --mode fast \
    --output submission.csv

# 3. Full scientific ensemble (if time allows):
python3 train_and_predict.py \
    --train /path/to/competition_train.csv \
    --test  /path/to/competition_test.csv  \
    --mode full \
    --output submission_ensemble.csv

# 4. If competition columns differ from defaults, specify them explicitly:
python3 train_and_predict.py \
    --train competition_train.csv \
    --test  competition_test.csv  \
    --target-col "Power_MW" \
    --ws-col "WindSpeed" \
    --wd-col "WindDir" \
    --mode fast
```

---

## Quick Reference: Column Name Mapping

When you open the competition dataset and find non-standard column names,
edit **[`ml_pipeline/config.py`](ml_pipeline/config.py)** — **nothing else**.

The fuzzy schema inspector in `aerocast_loader.py` will attempt auto-detection
first. If it fails, use `--target-col`, `--ws-col`, `--wd-col`, `--temp-col`,
`--press-col` CLI flags (see Step 9 above).

---

## Step 10 — Deactivate When Done

```bash
deactivate
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `python3: command not found` | Install via `brew install python@3.12` |
| `pip: externally-managed-environment` error | Use venv (Step 2-3) — never install into system Python |
| `lightgbm` install takes >10 minutes | Force binary: `pip install lightgbm --only-binary=:all:` |
| `catboost` install fails | Try: `pip install catboost --no-build-isolation` |
| Tests fail with `FileNotFoundError` | Run Step 5 (generate dummy data) before tests |
| `submission.csv` has wrong column names | Use `--col-p10 P10 --col-p50 Predicted --col-p90 P90` flags |
| Port or permission error running scripts | Ensure `.venv` is active (`which python` shows venv path) |

---

*AeroCast-AI v2.0 — Physics-Informed AI with Probabilistic Bounds*  
*AI SANSAD 2026 | Department of AI & ML, BVCOE Delhi*
