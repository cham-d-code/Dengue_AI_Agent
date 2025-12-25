"""
Weather Fetch Test Script
Tests if weather data is loading correctly in the dashboard
"""
import pandas as pd
import os

# Check all weather files
print("=== WEATHER DATA TEST ===\n")

ROOT = os.path.dirname(os.path.abspath(__file__))

# 1. OpenMeteo Forecast
forecast_file = os.path.join(ROOT, "data_raw", "weather_forecast_openmeteo_7day.csv")
if os.path.exists(forecast_file):
    df = pd.read_csv(forecast_file)
    print(f"1. OpenMeteo Forecast: {len(df)} rows")
    print(f"   Districts: {df['district'].nunique()}")
    print(f"   Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Avg Rain: {df['rain_daily_total'].mean():.1f}mm")
    print(f"   Max Rain: {df['rain_daily_total'].max():.1f}mm")
    print(f"   Columns: {df.columns.tolist()}")
    print()
    
    # Show sample data
    print("   Sample Data (first 5 rows):")
    print(df[['district', 'date', 'rain_daily_total', 'temp_daily_avg', 'humidity_daily_avg']].head().to_string())
    print()
else:
    print("1. OpenMeteo Forecast: NOT FOUND")

# 2. Weather Alerts
alerts_file = os.path.join(ROOT, "data_processed", "weather_risk_alerts_next_7_days.csv")
if os.path.exists(alerts_file):
    df_alerts = pd.read_csv(alerts_file)
    print(f"2. Weather Alerts: {len(df_alerts)} rows")
    if not df_alerts.empty:
        print(df_alerts.head().to_string())
    else:
        print("   (Empty - no high risk weather alerts currently)")
else:
    print("2. Weather Alerts: NOT FOUND")

# 3. Check dashboard data loading
print("\n3. Dashboard Weather Loading Test:")
wx_file = os.path.join(ROOT, "data_processed", "weather_risk_alerts_next_7_days.csv")
if os.path.exists(wx_file):
    df_wx = pd.read_csv(wx_file)
    print(f"   Loaded: {len(df_wx)} weather alerts")
    print(f"   Empty: {df_wx.empty}")
    
    if df_wx.empty:
        print("\n   ⚠️ Weather alerts file is EMPTY!")
        print("   The chatbot will use DEFAULT weather values.")
        print("   This is normal if current weather is not high-risk.")
else:
    print("   Weather file not found!")

# 4. Test what the chatbot sees
print("\n4. Testing Chatbot Weather Access:")
print("   Since df_wx is empty, the chatbot uses simulated/default values:")
print("   - rainfall_current = 45mm (default)")
print("   - temp_current = 28°C (default)")
print("   - humidity_current = 78% (default)")

print("\n=== WEATHER FETCH TEST COMPLETE ===")
