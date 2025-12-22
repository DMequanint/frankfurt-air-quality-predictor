from pathlib import Path
from loguru import logger
import typer
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, f1_score
import xgboost as xgb
import joblib
import sys
import os

# Fix import path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from frankfurt_air_quality_predictor.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "frankfurt_pm25_processed.csv",
    reg_model_path: Path = MODELS_DIR / "xgboost_pm25_regressor.pkl",
    cls_model_path: Path = MODELS_DIR / "xgboost_violation_classifier.pkl",
):
    """Train XGBoost Regressor + WHO Violation Classifier"""
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading Frankfurt PM2.5 data...")
    df = pd.read_csv(input_path)
    
    if df.empty:
        logger.error("No data! Run dataset.py first.")
        return
    
    logger.info(f"Loaded {len(df)} records")
    logger.info(f"PM2.5 range: {df['pm25'].min():.1f}-{df['pm25'].max():.1f} µg/m³")
    
    # Time series features
    df = df.sort_values('date').reset_index(drop=True)
    
    # Lags & rolling
    for lag in [1, 24]:
        df[f'lag_{lag}'] = df['pm25'].shift(lag)
    df['rolling_mean_24'] = df['pm25'].rolling(24, min_periods=1).mean()
    
    # Targets
    df['target_reg'] = df['pm25'].shift(-1)
    df['target_cls'] = (df['target_reg'] > 15).astype(int)
    df = df.dropna()
    
    # Features
    feature_cols = ['hour', 'day_of_week', 'rolling_24h', 'lag_1', 'lag_24', 'rolling_mean_24']
    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features]
    y_reg = df['target_reg']
    y_cls = df['target_cls']
    
    # Time split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_reg_train, y_reg_test = y_reg[:split], y_reg[split:]
    y_cls_train, y_cls_test = y_cls[:split], y_cls[split:]
    
    logger.info(f"Training: {len(X_train)} | Test: {len(X_test)}")
    logger.info(f"Violation rate: {y_cls.mean():.1%}")
    
    # 1. REGRESSOR
    logger.info("Training XGBoost Regressor...")
    reg_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
    reg_model.fit(X_train, y_reg_train)
    reg_pred = reg_model.predict(X_test)
    reg_mae = mean_absolute_error(y_reg_test, reg_pred)
    
    # 2. CLASSIFIER
    logger.info("Training XGBoost Classifier...")
    cls_model = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
    cls_model.fit(X_train, y_cls_train)
    cls_pred = cls_model.predict(X_test)
    cls_f1 = f1_score(y_cls_test, cls_pred)
    
    # Save both models
    joblib.dump(reg_model, reg_model_path)
    joblib.dump(cls_model, cls_model_path)
    
    logger.success(f"Regressor: {reg_model_path} (MAE: {reg_mae:.2f} µg/m³)")
    logger.success(f"Classifier: {cls_model_path} (F1: {cls_f1:.3f})")

if __name__ == "__main__":
    app()

