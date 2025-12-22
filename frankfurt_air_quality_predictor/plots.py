from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from frankfurt_air_quality_predictor.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def load_and_prepare_data(input_path: Path) -> pd.DataFrame:
    """
    Load PM2.5 data and ensure date column is datetime.
    
    Args:
        input_path: Path to processed dataset
        
    Returns:
        DataFrame with datetime 'date' column
    """
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)


def plot_timeseries_trends(df: pd.DataFrame, output_path: Path):
    """Create time-series plot of PM2.5 concentrations."""
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Plot PM2.5 and rolling average
    ax.plot(df['date'], df['pm25'], alpha=0.7, linewidth=0.8, label='PM2.5 (hourly)')
    ax.plot(df['date'], df['rolling_24h'], linewidth=2, label='24h rolling mean')
    
    # WHO guideline line
    ax.axhline(y=15, color='red', linestyle='--', alpha=0.8, label='WHO guideline (15 µg/m³)')
    
    ax.set_title('Frankfurt PM2.5 Time Series (2024-2025)', fontsize=14, fontweight='bold')
    ax.set_ylabel('PM2.5 Concentration (µg/m³)')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Timeseries saved: {output_path}")


def plot_hourly_patterns(df: pd.DataFrame, output_path: Path):
    """Create heatmap of hourly PM2.5 patterns by day of week."""
    hourly_avg = df.groupby(['hour', 'day_of_week'])['pm25'].mean().unstack()
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(hourly_avg, annot=False, cmap='YlOrRd', cbar_kws={'label': 'PM2.5 (µg/m³)'})
    plt.title('Hourly PM2.5 Patterns by Day of Week', fontsize=14, fontweight='bold')
    plt.xlabel('Day of Week (0=Mon, 6=Sun)')
    plt.ylabel('Hour of Day')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Hourly heatmap saved: {output_path}")


def plot_violation_distribution(df: pd.DataFrame, output_path: Path):
    """Create distribution plot with WHO violation highlights."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # PM2.5 histogram
    ax1.hist(df['pm25'], bins=50, alpha=0.7, edgecolor='black', density=True)
    ax1.axvline(15, color='red', linestyle='--', linewidth=2, label='WHO guideline')
    ax1.set_title('PM2.5 Distribution')
    ax1.set_xlabel('PM2.5 (µg/m³)')
    ax1.legend()
    
    # Violation pie chart
    violation_counts = df['is_high_pollution'].value_counts()
    ax2.pie(violation_counts.values, labels=['Safe (≤15)', 'Violation (>15)'], 
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('WHO Guideline Violations')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Distribution saved: {output_path}")


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "frankfurt_pm25_processed.csv",
):
    """
    Generate comprehensive air quality visualizations.
    
    Creates 3 publication-ready plots:
    1. Time-series trends with rolling average and WHO guideline
    2. Hourly heatmap by day of week
    3. PM2.5 distribution + violation pie chart
    
    Args:
        input_path: Processed dataset with PM2.5 data
    """
    # Ensure output directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading data: {input_path}")
    df = load_and_prepare_data(input_path)
    
    if df.empty:
        logger.error("No data found")
        return
    
    logger.info(f"Loaded {len(df)} records. Mean PM2.5: {df['pm25'].mean():.1f} µg/m³")
    
    # Generate plots
    timestamps = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    plot_timeseries_trends(
        df, 
        FIGURES_DIR / f"timeseries_pm25_{timestamps}.png"
    )
    
    plot_hourly_patterns(
        df, 
        FIGURES_DIR / f"hourly_heatmap_{timestamps}.png"
    )
    
    plot_violation_distribution(
        df, 
        FIGURES_DIR / f"distribution_violations_{timestamps}.png"
    )
    
    logger.success("All visualizations generated successfully!")
    logger.info(f"Plots saved in: {FIGURES_DIR}")
    logger.info("Files:")
    for plot_file in FIGURES_DIR.glob("*.png"):
        logger.info(f"  - {plot_file.name}")


if __name__ == "__main__":
    app()

