from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import numpy as np

from frankfurt_air_quality_predictor.config import PROCESSED_DATA_DIR

app = typer.Typer()


def create_time_series_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create comprehensive time-series features for PM2.5 forecasting.
    
    Args:
        df: Raw dataset with 'date' and 'pm25' columns
        
    Returns:
        DataFrame with lag, rolling, and temporal features
    """
    df = df.copy()
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Temporal features
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    
    # Lag features (previous values)
    for lag in [1, 24, 168]:  # 1h, 24h, 7 days
        df[f'lag_{lag}h'] = df['pm25'].shift(lag)
    
    # Rolling statistics
    for window in [6, 24, 168]:
        df[f'rolling_mean_{window}h'] = df['pm25'].rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}h'] = df['pm25'].rolling(window=window, min_periods=1).std()
    
    # Exponential moving averages
    df['ema_24h'] = df['pm25'].ewm(span=24).mean()
    df['ema_7d'] = df['pm25'].ewm(span=168).mean()
    
    # WHO violation targets
    df['target_pm25'] = df['pm25'].shift(-1)  # Next hour prediction
    df['target_violation'] = (df['target_pm25'] > 15).astype(int)
    
    # Drop rows with missing targets
    df = df.dropna(subset=['target_pm25'])
    
    logger.info(
        f"Generated {len(df)} samples with {len(df.columns)-2} features"
    )
    
    return df


def validate_features(df: pd.DataFrame):
    """Validate feature quality and log statistics."""
    feature_cols = [col for col in df.columns if col.startswith(('lag_', 'rolling_', 'ema_'))]
    
    stats = {
        'samples': len(df),
        'features': len(feature_cols),
        'target_violation_rate': df['target_violation'].mean(),
        'pm25_range': (df['pm25'].min(), df['pm25'].max()),
        'missing_values': df[feature_cols].isnull().sum().sum()
    }
    
    logger.info("Feature statistics:")
    logger.info(f"  Samples: {stats['samples']:,}")
    logger.info(f"  Features: {stats['features']}")
    logger.info(f"  Target violation rate: {stats['target_violation_rate']:.1%}")
    logger.info(f"  PM2.5 range: {stats['pm25_range'][0]:.1f} - {stats['pm25_range'][1]:.1f} µg/m³")
    logger.info(f"  Missing values: {stats['missing_values']:,}")


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "frankfurt_pm25_processed.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
):
    """
    Generate time-series features for ML training from raw PM2.5 dataset.
    
    Args:
        input_path: Raw dataset with 'date' and 'pm25' columns
        output_path: ML-ready features CSV
    """
    # Ensure directories exist
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading dataset: {input_path}")
    df = pd.read_csv(input_path)
    
    if df.empty:
        logger.error("Input dataset is empty")
        return
    
    logger.info(f"Loaded {len(df)} records from {input_path}")
    
    # Generate features
    logger.info("Creating time-series features...")
    df_features = create_time_series_features(df)
    
    # Validate
    validate_features(df_features)
    
    # Save
    df_features.to_csv(output_path, index=False)
    
    logger.success(f"Features saved: {output_path}")
    logger.success(f"Ready for XGBoost training: {len(df_features)} samples")


if __name__ == "__main__":
    app()

