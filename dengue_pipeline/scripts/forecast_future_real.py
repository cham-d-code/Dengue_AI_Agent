import os
import sys
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ---------------- Paths ---------------- #

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

HISTORICAL_FILE = os.path.join(PROJECT_ROOT, "data_processed", "weekly_district_dataset_2021_2024.csv")
FUTURE_WEATHER_FILE = os.path.join(PROJECT_ROOT, "data_raw", "weather_next_14_days.csv")
OUT_FILE = os.path.join(PROJECT_ROOT, "data_processed", "future_forecast_next_2_weeks_weather_based.csv")

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# ---------------- Capacity & risk config ---------------- #

HOSPITALISATION_RATE = 0.15          # 15% of dengue cases need admission
AVERAGE_LENGTH_OF_STAY_DAYS = 4      # days
BLOOD_NEED_RATE = 0.05               # 5% of admitted need blood/platelets
BLOOD_UNITS_PER_PATIENT = 4          # units per patient

HIGH_RISK_MULTIPLIER = 1.5
CRITICAL_RISK_MULTIPLIER = 2.0


def estimate_beds_needed(predicted_cases: float) -> float:
    hospitalised = predicted_cases * HOSPITALISATION_RATE
    bed_days = hospitalised * AVERAGE_LENGTH_OF_STAY_DAYS
    return bed_days / 7.0


def estimate_blood_units_needed(predicted_cases: float) -> float:
    hospitalised = predicted_cases * HOSPITALISATION_RATE
    patients_needing_blood = hospitalised * BLOOD_NEED_RATE
    return patients_needing_blood * BLOOD_UNITS_PER_PATIENT


def classify_risk(pred_cases: float, baseline: float) -> str:
    if baseline is None or np.isnan(baseline) or baseline <= 0:
        if pred_cases < 10:
            return "low"
        elif pred_cases < 50:
            return "moderate"
        elif pred_cases < 100:
            return "high"
        else:
            return "critical"

    ratio = pred_cases / baseline
    if ratio < 1.0:
        return "low"
    elif ratio < HIGH_RISK_MULTIPLIER:
        return "moderate"
    elif ratio < CRITICAL_RISK_MULTIPLIER:
        return "high"
    else:
        return "critical"


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52.0)
    return df


def add_week(year: int, week: int, delta_weeks: int = 1):
    """
    Move (year, week) forward by delta_weeks using ISO weeks.
    Simple loop but good enough.
    """
    curr = date.fromisocalendar(year, week, 1)  # Monday of that week
    future = curr + timedelta(weeks=delta_weeks)
    iso = future.isocalendar()
    return iso.year, iso.week


def main():
    # 1) Load historical weekly dengue + weather
    if not os.path.exists(HISTORICAL_FILE):
        raise FileNotFoundError(f"Missing historical file: {HISTORICAL_FILE}")
    hist = pd.read_csv(HISTORICAL_FILE)

    # Expect columns: district, year, week, cases, rainfall_mm, temp_avg_c, humidity_pct
    hist["district"] = hist["district"].astype(str).str.strip()
    hist["year"] = hist["year"].astype(int)
    hist["week"] = hist["week"].astype(int)
    hist["cases"] = hist["cases"].astype(float)
    hist["rainfall_mm"] = hist["rainfall_mm"].astype(float)
    hist["temp_avg_c"] = hist["temp_avg_c"].astype(float)

    # Add seasonal features
    hist = add_seasonal_features(hist)

    # Historical baseline by district+week (2021–2024)
    baseline = (
        hist.groupby(["district", "week"])["cases"]
        .mean()
        .reset_index()
        .rename(columns={"cases": "historical_avg_cases"})
    )

    # 2) Train simple weather+seasonality model (no case lags, so we can forecast any year)
    feature_cols = ["week", "rainfall_mm", "temp_avg_c", "week_sin", "week_cos"]
    X = hist[feature_cols]
    y = hist["cases"]

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=18,
        random_state=42,
        n_jobs=-1
    )
    print("Training weather+seasonality model on 2021–2024...")
    model.fit(X, y)

    # 3) Load future daily weather for next 14 days
    if not os.path.exists(FUTURE_WEATHER_FILE):
        raise FileNotFoundError(
            f"Missing future weather file: {FUTURE_WEATHER_FILE}\n"
            "Run fetch_openmeteo_next_14_days.py first."
        )

    fw = pd.read_csv(FUTURE_WEATHER_FILE)
    # columns: district, date, rain_sum, temp_max, temp_min
    fw["district"] = fw["district"].astype(str).str.strip()
    fw["date"] = pd.to_datetime(fw["date"])
    fw["year"] = fw["date"].dt.isocalendar().year.astype(int)
    fw["week"] = fw["date"].dt.isocalendar().week.astype(int)
    fw["temp_avg_c"] = (fw["temp_max"] + fw["temp_min"]) / 2.0

    # Aggregate to weekly weather: sum rain, avg temp
    fw_weekly = (
        fw.groupby(["district", "year", "week"], as_index=False)
        .agg({
            "rain_sum": "sum",
            "temp_avg_c": "mean"
        })
        .rename(columns={"rain_sum": "rainfall_mm"})
    )

    # 4) Determine next 2 ISO weeks from *today*
    today = datetime.now().date()
    iso = today.isocalendar()
    curr_year, curr_week = iso.year, iso.week

    target_year_1, target_week_1 = add_week(curr_year, curr_week, 1)
    target_year_2, target_week_2 = add_week(curr_year, curr_week, 2)

    targets = [
        (target_year_1, target_week_1, 1),
        (target_year_2, target_week_2, 2),
    ]

    print(f"Today: {today} (ISO year={curr_year}, week={curr_week})")
    print("Forecasting for:")
    for y_, w_, h_ in targets:
        print(f"  +{h_} week(s): year={y_}, week={w_}")

    # 5) For each district & target week, build features and predict
    districts = sorted(hist["district"].unique())
    results = []

    for d in districts:
        for (ty, tw, horizon) in targets:
            # Get weekly weather forecast for that district & target week
            wrow = fw_weekly[
                (fw_weekly["district"] == d)
                & (fw_weekly["year"] == ty)
                & (fw_weekly["week"] == tw)
            ]

            if wrow.empty:
                # Not enough forecast days to form this week
                print(f"⚠️ No forecast weather for {d}, year={ty}, week={tw}, skipping.")
                continue

            wrow = wrow.iloc[0]
            rainfall_mm = float(wrow["rainfall_mm"])
            temp_avg_c = float(wrow["temp_avg_c"])

            # Seasonal features for the target week (week number only matters, not year)
            week_sin = np.sin(2 * np.pi * tw / 52.0)
            week_cos = np.cos(2 * np.pi * tw / 52.0)

            feat = pd.DataFrame([{
                "week": tw,
                "rainfall_mm": rainfall_mm,
                "temp_avg_c": temp_avg_c,
                "week_sin": week_sin,
                "week_cos": week_cos,
            }])[feature_cols]

            pred_cases = float(model.predict(feat)[0])

            # Historical baseline from 2021–2024
            b = baseline[
                (baseline["district"] == d)
                & (baseline["week"] == tw)
            ]
            if b.empty:
                hist_avg = np.nan
            else:
                hist_avg = float(b["historical_avg_cases"].iloc[0])

            risk = classify_risk(pred_cases, hist_avg)
            beds = estimate_beds_needed(pred_cases)
            blood = estimate_blood_units_needed(pred_cases)

            results.append({
                "district": d,
                "target_year": ty,
                "target_week": tw,
                "horizon_weeks_ahead": horizon,
                "predicted_cases": pred_cases,
                "historical_avg_cases": hist_avg,
                "rainfall_mm_forecast": rainfall_mm,
                "temp_avg_c_forecast": temp_avg_c,
                "beds_needed": beds,
                "blood_units_needed": blood,
                "risk": risk,
            })

    if not results:
        print("❌ No forecasts produced (check your weather_next_14_days.csv coverage).")
        return

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_FILE, index=False)

    print(f"\n🎉 Saved real-future 2-week forecast to:\n  {OUT_FILE}\n")

    # Print a summary for a few districts
    print("=== Sample of future forecasts ===")
    for _, row in (
        out_df.sort_values(["horizon_weeks_ahead", "district"])
        .groupby("horizon_weeks_ahead")
        .head(5)
        .iterrows()
    ):
        print(
            f"{row['district']}: year={int(row['target_year'])}, "
            f"week={int(row['target_week'])}, +{int(row['horizon_weeks_ahead'])}w, "
            f"cases≈{row['predicted_cases']:.1f}, risk={row['risk']}, "
            f"beds≈{row['beds_needed']:.1f}, blood≈{row['blood_units_needed']:.1f}"
        )


if __name__ == "__main__":
    main()
