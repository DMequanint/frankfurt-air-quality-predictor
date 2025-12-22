from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import numpy as np
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
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    predictions_path: Path = PROCESSED_DATA_DIR / "predictions.csv",
):
    """Batch predict PM2.5 + WHO violations using DUAL XGBoost models"""
    
    # Load BOTH models
    logger.info("Loading DUAL XGBoost models...")
    reg_model = joblib.load(MODELS_DIR / "xgboost_pm25_regressor.pkl")
    cls_model = joblib.load(MODELS_DIR / "xgboost_violation_classifier.pkl")
    
    logger.success(f"Regressor loaded! Features: {list(reg_model.feature_names_in_)}")
    logger.success(f"Classifier loaded!")
    
    # Generate test data if no features file
    if not features_path.exists():
        logger.info("Generating sample test data...")
        n_samples = 100
        X_test = pd.DataFrame({
            'hour': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'rolling_24h': np.random.normal(12, 5, n_samples).clip(0),
            'lag_1': np.random.normal(12, 5, n_samples).clip(0),
            'lag_24': np.random.normal(12, 5, n_samples).clip(0),
            'rolling_mean_24': np.random.normal(12, 5, n_samples).clip(0)
        })
        X_test.to_csv(features_path, index=False)
    else:
        X_test = pd.read_csv(features_path)
    
    # DUAL predictions
    pm25_preds = reg_model.predict(X_test)
    violation_preds = cls_model.predict(X_test)
    violation_probs = cls_model.predict_proba(X_test)[:, 1]  # Probability of violation
    
    # Results
    results = X_test.copy()
    results['predicted_pm25'] = pm25_preds
    results['predicted_violation'] = violation_preds
    results['violation_probability'] = violation_probs
    results['alert'] = results['predicted_violation'].map({0: '✅ Safe', 1: '🚨 VIOLATION'})
    
    results.to_csv(predictions_path, index=False)
    
    violations = violation_preds.sum()
    logger.success(f"Dual predictions saved: {predictions_path}")
    logger.success(f"PM2.5 range: {pm25_preds.min():.1f}-{pm25_preds.max():.1f} µg/m³")
    logger.success(f"Violations: {violations}/{len(violation_preds)} ({violations/len(violation_preds)*100:.1f}%)")

@app.command()
def quick(
    hour: int = 18,
    day_of_week: int = 5,
    rolling_24h: float = 20.5,
    lag_1: float = 22.1,
    lag_24: float = 18.9,
    rolling_mean_24: float = 19.8,
):
    """Quick DUAL prediction (PM2.5 + WHO violation)"""
    
    # Load DUAL models
    reg_model = joblib.load(MODELS_DIR / "xgboost_pm25_regressor.pkl")
    cls_model = joblib.load(MODELS_DIR / "xgboost_violation_classifier.pkl")
    
    X_sample = pd.DataFrame({
        'hour': [hour],
        'day_of_week': [day_of_week],
        'rolling_24h': [rolling_24h],
        'lag_1': [lag_1],
        'lag_24': [lag_24],
        'rolling_mean_24': [rolling_mean_24]
    })
    
    # DUAL predictions
    pm25_pred = reg_model.predict(X_sample)[0]
    violation_pred = cls_model.predict(X_sample)[0]
    violation_prob = cls_model.predict_proba(X_sample)[0][1]
    
    alert = "WHO VIOLATION" if violation_pred else "Safe"
    logger.success(f"PM2.5: {pm25_pred:.1f} µg/m³ | Violation: {alert}")
    logger.info(f"Confidence: {violation_prob:.1%} ({'HIGH' if violation_prob > 0.7 else 'MEDIUM' if violation_prob > 0.3 else 'LOW'})")

@app.command()
def test():
    """Test DUAL model load"""
    reg_model = joblib.load(MODELS_DIR / "xgboost_pm25_regressor.pkl")
    cls_model = joblib.load(MODELS_DIR / "xgboost_violation_classifier.pkl")
    logger.success(f"DUAL MODELS OK!")
    logger.success(f"Regressor features: {list(reg_model.feature_names_in_)}")
    logger.success(f"Classifier ready for WHO violations")

if __name__ == "__main__":
    app()
