#!/bin/bash
# Weekly Data Update Script for Dengue Prediction System
# Runs every Monday at 6 AM Sri Lanka time

set -e  # Exit on error

cd /app

echo "============================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting weekly data update..."
echo "============================================"

# 1. Fetch past 14 days weather data (for t-2 lag analysis)
echo ""
echo "[Step 1/4] Fetching past 14 days weather from OpenMeteo Archive..."
python dengue_pipeline/scripts/fetch_openmeteo_past_14_days.py
echo "✓ Past weather data updated"

# 2. Fetch 7-day weather forecast
echo ""
echo "[Step 2/4] Fetching 7-day weather forecast from OpenMeteo..."
python dengue_pipeline/scripts/fetch_openmeteo_forecast.py
echo "✓ Weather forecast updated"

# 3. Generate weather-based risk alerts
echo ""
echo "[Step 3/4] Generating weather risk alerts..."
python dengue_pipeline/scripts/weather_trigger_alerts.py
echo "✓ Weather alerts generated"

# 4. Optional: Re-run forecast model (if new case data available)
echo ""
echo "[Step 4/4] Updating forecasts..."
# Uncomment below if you have automated case data ingestion
# python dengue_pipeline/scripts/forecast_next_two_weeks.py
echo "✓ Forecast update complete (manual case data update required)"

echo ""
echo "============================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Weekly update completed successfully!"
echo "============================================"
echo ""
echo "Next scheduled run: Monday 6:00 AM"
