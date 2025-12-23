# frankfurt-air-quality-predictor

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Frankfurt PM2.5 forecasting with WHO exceedance detection using OpenAQ API

```
# Frankfurt Air Quality Predictor

End-to-end machine learning pipeline forecasting PM2.5 concentrations and WHO guideline exceedance detection for Frankfurt, Germany.

## What It Does

Built dual XGBoost models on 17,280 hourly PM2.5 records (2024-2025):

| Model | Metric | Performance |
|-------|--------|-------------|
| PM2.5 Regression | MAE | 1.32 µg/m³ |
| WHO Violation Classification | F1 Score | 0.892 |

**Key Capabilities:**
- Next-hour PM2.5 concentration forecasting
- WHO guideline exceedance detection (>15 µg/m³)
- Confidence-scored alerts (92-100% accuracy)
- Production CLI interface

**Data:** Open-Meteo Air Quality API (Frankfurt: 50.11°N, 8.68°E)
**Features:** 22+ time-series (lags, rolling stats, EMA, temporal encodings)

## Pipeline Architecture

dataset.py → 17K raw records
            ↓
features.py → 22+ engineered features
            ↓
modeling/train.py → Dual XGBoost models
            ↓
modeling/predict.py → Live predictions
            ↓
plots.py → 3 publication plots

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/frankfurt-air-quality-predictor
cd frankfurt-air-quality-predictor
pip install -r requirements.txt

# 2. Run full pipeline
python -m frankfurt_air_quality_predictor.dataset    # Fetch 17K records
python -m frankfurt_air_quality_predictor.features   # 22+ features
python -m frankfurt_air_quality_predictor.modeling.train  # Train models
python -m frankfurt_air_quality_predictor.plots      # Generate plots

# High pollution scenario
python -m frankfurt_air_quality_predictor.modeling.predict quick
# PM2.5: 26.0 µg/m³ | Violation: WHO VIOLATION (100.0% confidence)

# Safe morning scenario
python -m frankfurt_air_quality_predictor.modeling.predict quick --hour 9 --day-of-week 1 --rolling-24h 8.2 --lag-1 7.9 --lag-24 9.1 --rolling-mean-24 8.5
# PM2.5: 6.6 µg/m³ | Violation: Safe (0.0% confidence)
data/processed/frankfurt_pm25_processed.csv     # 17,280 records
data/processed/features.csv                     # 22+ ML features
models/xgboost_pm25_regressor.pkl               # MAE: 1.32 µg/m³
models/xgboost_violation_classifier.pkl         # F1: 0.892
reports/figures/*.png                           # 3 publication plots
data/processed/predictions.csv                  # Batch predictions

Temporal: hour, day_of_week, day_of_month, month
Lags: lag_1h, lag_24h, lag_168h
Rolling: mean/std (6h, 24h, 168h)
EMA: 24h, 7-day
Targets: target_pm25, target_violation

#Model Architecture:
Regressor: XGBoost (n_estimators=100, max_depth=6)
Classifier: XGBoost (n_estimators=100, max_depth=4, class-balanced)
Train/Test: 13,804 / 3,451 (time-aware 80/20 split)

#Production Commands
python -m frankfurt_air_quality_predictor.modeling.predict test     # Verify models
python -m frankfurt_air_quality_predictor.modeling.predict main     # 100 predictions → CSV
python -m frankfurt_air_quality_predictor.modeling.predict quick    --hour 18 --rolling-24h 20.5

## Project Organization

frankfurt-air-quality-predictor/                    # PROJECT ROOT
├── README.md
├── requirements.txt
├── pyproject.toml
├── LICENSE
├── .gitignore                                      # CREATE THIS
│
├── data/
│   ├── raw/
│   └── processed/                                 # Keep your CSVs temporarily
│
├── models/                                        # Keep your .pkl files temporarily
│
├── reports/
│   └── figures/                                   # Keep your PNG plots
│
└── frankfurt_air_quality_predictor/               # ALL SOURCE CODE
    ├── __init__.py
    ├── config.py
    ├── dataset.py                                 # Data pipeline
    ├── features.py                                # Feature engineering
    ├── modeling/
    │   ├── __init__.py
    │   ├── train.py                               # Dual XGBoost
    │   └── predict.py                             # Live predictions
    └── plots.py                                   # Visualizations

#Author: Desalegn Yehuala
#Built: December 2025
#Tech Stack: Python, XGBoost, Pandas, Typer CLI

```

--------

