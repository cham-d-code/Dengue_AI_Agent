"""
Historical Analysis Module for DengueGuard AI
Provides real historical dengue case data and weather correlation analysis.
Uses 2021-2024 data for Week X comparisons across years.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# Path configuration
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
DATA_FILE = os.path.join(PROJECT_ROOT, "data_processed", "weekly_district_dataset_2021_2024.csv")
WEATHER_FILE = os.path.join(PROJECT_ROOT, "data_processed", "weather_weekly_all_districts_2010_2024.csv")

# Load data at module level for caching
_df_combined = None
_df_weather = None

def _load_data():
    """Load and cache the combined dengue + weather dataset."""
    global _df_combined, _df_weather
    
    if _df_combined is None:
        if os.path.exists(DATA_FILE):
            _df_combined = pd.read_csv(DATA_FILE)
        else:
            _df_combined = pd.DataFrame()
    
    if _df_weather is None:
        if os.path.exists(WEATHER_FILE):
            _df_weather = pd.read_csv(WEATHER_FILE)
        else:
            _df_weather = pd.DataFrame()
    
    return _df_combined, _df_weather


def get_historical_cases(district: str, week: int, years: list = None) -> dict:
    """
    Get dengue cases for a specific week across multiple years.
    
    Args:
        district: District name (e.g., "Colombo")
        week: Week number (1-52)
        years: List of years to include (default: [2021, 2022, 2023, 2024])
    
    Returns:
        Dict with year as key, cases as value. E.g., {2024: 45, 2023: 38, ...}
    """
    if years is None:
        years = [2021, 2022, 2023, 2024]
    
    df, _ = _load_data()
    if df.empty:
        return {year: 0 for year in years}
    
    result = {}
    for year in years:
        mask = (df['district'] == district) & (df['year'] == year) & (df['week'] == week)
        matches = df[mask]
        if not matches.empty:
            result[year] = int(matches.iloc[0]['cases'])
        else:
            result[year] = None  # No data for this week/year
    
    return result


def get_weather_lagged(district: str, year: int, week: int, lag: int = 2) -> dict:
    """
    Get weather data from (week - lag) weeks ago for correlation analysis.
    
    Args:
        district: District name
        year: Year to query
        week: Current week number
        lag: Number of weeks to look back (default: 2)
    
    Returns:
        Dict with rainfall_mm, temp_avg_c, humidity_pct
    """
    df, _ = _load_data()
    if df.empty:
        return {'rainfall_mm': 0, 'temp_avg_c': 28.0, 'humidity_pct': 75.0}
    
    lagged_week = week - lag
    lagged_year = year
    
    # Handle week wraparound (e.g., week -1 becomes week 51 of previous year)
    if lagged_week <= 0:
        lagged_week = 52 + lagged_week
        lagged_year = year - 1
    
    mask = (df['district'] == district) & (df['year'] == lagged_year) & (df['week'] == lagged_week)
    matches = df[mask]
    
    if not matches.empty:
        row = matches.iloc[0]
        return {
            'rainfall_mm': float(row.get('rainfall_mm', 0)),
            'temp_avg_c': float(row.get('temp_avg_c', 28.0)),
            'humidity_pct': float(row.get('humidity_pct', 75.0))
        }
    
    return {'rainfall_mm': 0, 'temp_avg_c': 28.0, 'humidity_pct': 75.0}


def get_historical_with_weather(district: str, week: int, years: list = None) -> list:
    """
    Get historical cases with corresponding t-2 lagged weather for each year.
    
    Returns:
        List of dicts with year, cases, and weather data
    """
    if years is None:
        years = [2024, 2023, 2022, 2021]
    
    result = []
    for year in years:
        cases_data = get_historical_cases(district, week, [year])
        weather = get_weather_lagged(district, year, week, lag=2)
        
        result.append({
            'year': year,
            'cases': cases_data.get(year),
            'rainfall_mm': weather['rainfall_mm'],
            'temp_avg_c': weather['temp_avg_c'],
            'humidity_pct': weather['humidity_pct']
        })
    
    return result


def find_similar_weather_periods(district: str, rainfall: float, temp: float, humidity: float, top_n: int = 3) -> list:
    """
    Find historical weeks with similar weather conditions and their dengue outcomes.
    Uses Euclidean distance on normalized weather features.
    
    Returns:
        List of dicts with year, week, cases, and similarity score
    """
    df, _ = _load_data()
    if df.empty:
        return []
    
    # Filter by district
    dist_df = df[df['district'] == district].copy()
    if dist_df.empty:
        return []
    
    # Normalize features for distance calculation
    dist_df['rain_norm'] = (dist_df['rainfall_mm'] - dist_df['rainfall_mm'].mean()) / (dist_df['rainfall_mm'].std() + 1e-6)
    dist_df['temp_norm'] = (dist_df['temp_avg_c'] - dist_df['temp_avg_c'].mean()) / (dist_df['temp_avg_c'].std() + 1e-6)
    dist_df['humid_norm'] = (dist_df['humidity_pct'] - dist_df['humidity_pct'].mean()) / (dist_df['humidity_pct'].std() + 1e-6)
    
    # Normalize query values
    rain_norm = (rainfall - dist_df['rainfall_mm'].mean()) / (dist_df['rainfall_mm'].std() + 1e-6)
    temp_norm = (temp - dist_df['temp_avg_c'].mean()) / (dist_df['temp_avg_c'].std() + 1e-6)
    humid_norm = (humidity - dist_df['humidity_pct'].mean()) / (dist_df['humidity_pct'].std() + 1e-6)
    
    # Calculate Euclidean distance
    dist_df['distance'] = np.sqrt(
        (dist_df['rain_norm'] - rain_norm)**2 +
        (dist_df['temp_norm'] - temp_norm)**2 +
        (dist_df['humid_norm'] - humid_norm)**2
    )
    
    # Get top N most similar periods
    top = dist_df.nsmallest(top_n, 'distance')
    
    result = []
    for _, row in top.iterrows():
        result.append({
            'year': int(row['year']),
            'week': int(row['week']),
            'cases': int(row['cases']),
            'rainfall_mm': float(row['rainfall_mm']),
            'temp_avg_c': float(row['temp_avg_c']),
            'similarity': round(1 / (1 + row['distance']), 2)  # Convert distance to similarity score
        })
    
    return result


def calculate_weather_case_correlation(district: str) -> dict:
    """
    Calculate Pearson correlation between t-2 lagged weather and dengue cases.
    
    Returns:
        Dict with correlation coefficients for rainfall, temp, humidity
    """
    df, _ = _load_data()
    if df.empty or len(df) < 10:
        return {'rainfall': 0, 'temperature': 0, 'humidity': 0}
    
    dist_df = df[df['district'] == district].copy()
    if dist_df.empty or len(dist_df) < 10:
        return {'rainfall': 0, 'temperature': 0, 'humidity': 0}
    
    # Sort by year and week for proper lag calculation
    dist_df = dist_df.sort_values(['year', 'week']).reset_index(drop=True)
    
    # Create lagged weather (shift by 2)
    dist_df['rain_lag2'] = dist_df['rainfall_mm'].shift(2)
    dist_df['temp_lag2'] = dist_df['temp_avg_c'].shift(2)
    dist_df['humid_lag2'] = dist_df['humidity_pct'].shift(2)
    
    # Drop NaN rows
    dist_df = dist_df.dropna(subset=['rain_lag2', 'temp_lag2', 'humid_lag2', 'cases'])
    
    if len(dist_df) < 5:
        return {'rainfall': 0, 'temperature': 0, 'humidity': 0}
    
    return {
        'rainfall': round(dist_df['cases'].corr(dist_df['rain_lag2']), 3),
        'temperature': round(dist_df['cases'].corr(dist_df['temp_lag2']), 3),
        'humidity': round(dist_df['cases'].corr(dist_df['humid_lag2']), 3)
    }


def get_trend_analysis(district: str, current_week: int, lookback_weeks: int = 4) -> dict:
    """
    Analyze recent trend for a district based on available historical data.
    
    Returns:
        Dict with trend direction and percentage change
    """
    df, _ = _load_data()
    if df.empty:
        return {'trend': 'stable', 'change_pct': 0}
    
    # Use 2024 data for trend (most recent year)
    year = 2024
    dist_df = df[(df['district'] == district) & (df['year'] == year)].copy()
    
    if dist_df.empty:
        return {'trend': 'stable', 'change_pct': 0}
    
    # Get cases for recent weeks
    recent_weeks = range(max(1, current_week - lookback_weeks), current_week + 1)
    recent_data = dist_df[dist_df['week'].isin(recent_weeks)][['week', 'cases']]
    
    if len(recent_data) < 2:
        return {'trend': 'stable', 'change_pct': 0}
    
    recent_data = recent_data.sort_values('week')
    first_val = recent_data.iloc[0]['cases']
    last_val = recent_data.iloc[-1]['cases']
    
    if first_val == 0:
        change_pct = 100 if last_val > 0 else 0
    else:
        change_pct = ((last_val - first_val) / first_val) * 100
    
    if change_pct > 10:
        trend = 'increasing'
    elif change_pct < -10:
        trend = 'decreasing'
    else:
        trend = 'stable'
    
    return {'trend': trend, 'change_pct': round(change_pct, 1)}
