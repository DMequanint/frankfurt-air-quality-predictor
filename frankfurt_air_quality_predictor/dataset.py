from pathlib import Path
from loguru import logger
import typer
import requests
import pandas as pd

from frankfurt_air_quality_predictor.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def fetch_open_meteo_data(
    city: str = "Frankfurt",
    lat: float = 50.11,
    lon: float = 8.68,
) -> dict:
    """
    Fetch PM2.5 hourly data from Open-Meteo Air Quality API.
    
    Args:
        city: City name for logging
        lat: Latitude coordinate
        lon: Longitude coordinate
        
    Returns:
        Dictionary containing API response data
        
    Raises:
        requests.RequestException: If API request fails
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5",
        "start_date": "2024-01-01",
        "end_date": "2025-12-20",
        "timezone": "auto"
    }
    
    logger.info(f"Fetching data for {city} ({lat}, {lon}) from Open-Meteo API...")
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def process_air_quality_data(
    raw_data: dict,
    city: str = "Frankfurt"
) -> pd.DataFrame:
    """
    Process raw API data into ML-ready DataFrame with engineered features.
    
    Args:
        raw_data: API response dictionary
        city: City name for labeling
        
    Returns:
        Cleaned DataFrame with features for time-series forecasting
        
    Raises:
        KeyError: If expected data structure is missing
    """
    if "hourly" not in raw_data:
        raise KeyError("Unexpected API response format - missing 'hourly' data")
    
    hourly_data = raw_data["hourly"]
    df = pd.DataFrame({
        "date": pd.to_datetime(hourly_data["time"]),
        "pm25": hourly_data["pm2_5"]
    })
    
    # Data cleaning
    df = df.dropna(subset=["pm25"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # Feature engineering
    df["city"] = city
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    
    # Rolling statistics for time-series patterns
    df["rolling_6h"] = df["pm25"].rolling(window=6, min_periods=1).mean()
    df["rolling_24h"] = df["pm25"].rolling(window=24, min_periods=1).mean()
    
    # WHO guideline violation indicator (PM2.5 > 15 µg/m³)
    df["is_high_pollution"] = (df["pm25"] > 15).astype(int)
    
    logger.info(f"Processed {len(df)} records with features")
    return df


def save_processed_data(
    df: pd.DataFrame,
    output_path: Path,
    raw_backup_path: Path
):
    """
    Save processed data to CSV and pipe-delimited backup format.
    
    Args:
        df: Processed DataFrame
        output_path: Path for main CSV output
        raw_backup_path: Path for pipe-delimited backup
    """
    if df.empty:
        logger.error("Cannot save empty DataFrame")
        return
    
    # Main processed data (ML ready)
    df.to_csv(output_path, index=False)
    
    # Raw backup (pipe-delimited format)
    df.to_csv(raw_backup_path, sep="|", index=False)
    
    stats = {
        "records": len(df),
        "mean_pm25": df["pm25"].mean(),
        "violation_rate": df["is_high_pollution"].mean()
    }
    
    logger.success(
        f"Saved {stats['records']} records to {output_path}\n"
        f"Mean PM2.5: {stats['mean_pm25']:.2f} µg/m³\n"
        f"WHO violations: {stats['violation_rate']:.1%}"
    )


def main(
    city: str = "Frankfurt",
    lat: float = 50.11,
    lon: float = 8.68,
    output_path: Path = PROCESSED_DATA_DIR / "frankfurt_pm25_processed.csv",
):
    """
    Main function to fetch, process, and save Frankfurt PM2.5 data for ML pipeline.
    
    This creates ML-ready training data with time-series features and WHO violation labels.
    
    Args:
        city: Target city name (default: "Frankfurt")
        lat: Latitude coordinate (default: Frankfurt)
        lon: Longitude coordinate (default: Frankfurt)
        output_path: Output CSV path for processed data
    """
    # Ensure output directories exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    raw_backup_path = RAW_DATA_DIR / f"StationData-{city}.txt"
    
    try:
        # Step 1: Fetch data
        raw_data = fetch_open_meteo_data(city, lat, lon)
        
        # Step 2: Process data
        df = process_air_quality_data(raw_data, city)
        
        # Step 3: Save results
        save_processed_data(df, output_path, raw_backup_path)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    typer.run(main)

