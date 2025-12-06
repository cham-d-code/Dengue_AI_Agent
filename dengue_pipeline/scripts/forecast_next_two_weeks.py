import os
import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------- Paths ----------------- #

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_FILE = os.path.join(PROJECT_ROOT, "data_processed", "weekly_district_dataset_2021_2024.csv")
OUT_MULTI = os.path.join(PROJECT_ROOT, "data_processed", "multiweek_forecast_next_2_weeks.csv")

# ----------------- Config / assumptions ----------------- #

HOSPITALISATION_RATE = 0.15
AVERAGE_LENGTH_OF_STAY_DAYS = 4
BLOOD_NEED_RATE = 0.05
BLOOD_UNITS_PER_PATIENT = 4

HIGH_RISK_MULTIPLIER = 1.5
CRITICAL_RISK_MULTIPLIER = 2.0


# ----------------- Helper functions ----------------- #

def estimate_beds_needed(predicted_cases: float) -> float:
    hospitalised = predicted_cases * HOSPITALISATION_RATE
    bed_days = hospitalised * AVERAGE_LENGTH_OF_STAY_DAYS
    return bed_days / 7.0


def estimate_blood_units_needed(predicted_cases: float) -> float:
    hospitalised = predicted_cases * HOSPITALISATION_RATE
    patients_needing_blood = hospitalised * BLOOD_NEED_RATE
    return patients_needing_blood * BLOOD_UNITS_PER_PATIENT


def classify_risk(predicted_cases: float, historical_avg: float) -> str:
    if historical_avg is None or np.isnan(historical_avg) or historical_avg <= 0:
        if predicted_cases < 10:
            return "low"
        elif predicted_cases < 50:
            return "moderate"
        elif predicted_cases < 100:
            return "high"
        else:
            return "critical"

    ratio = predicted_cases / historical_avg
    if ratio < 1.0:
        return "low"
    elif ratio < HIGH_RISK_MULTIPLIER:
        return "moderate"
    elif ratio < CRITICAL_RISK_MULTIPLIER:
        return "high"
    else:
        return "critical"


def suggest_actions(row: pd.Series) -> str:
    risk = row["risk"]
    district = row["district"]
    beds_needed = row["beds_needed"]
    blood_needed = row["blood_units_needed"]

    actions = []

    if risk in ["high", "critical"]:
        actions.append(
            "Intensify vector control (fogging, source reduction) "
            f"in high-risk GN divisions in {district}."
        )
        actions.append(
            "Strengthen community awareness (schools, workplaces, media) "
            "with targeted dengue prevention messages."
        )
        actions.append(
            "Alert PHI and MOH teams to increase field inspections and case follow-up."
        )
        if risk == "critical":
            actions.append(
                "Activate or reinforce the district dengue emergency response plan "
                "and hold frequent situation review meetings."
            )

    actions.append(
        f"Prepare inpatient capacity for approximately {beds_needed:.0f} beds "
        "for dengue patients."
    )
    actions.append(
        f"Coordinate with the blood bank to ensure around {blood_needed:.0f} "
        "blood/platelet units are available for dengue-related needs."
    )
    actions.append(
        "Enhance triage and early warning at OPD/emergency units with clear pathways "
        "for severe case transfer."
    )

    return " ".join(actions)


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["district", "year", "week"]).copy()

    for lag in [1, 2, 3, 4]:
        df[f"cases_lag_{lag}"] = df.groupby("district")["cases"].shift(lag)
        df[f"rainfall_lag_{lag}"] = df.groupby("district")["rainfall_mm"].shift(lag)
        df[f"temp_lag_{lag}"] = df.groupby("district")["temp_avg_c"].shift(lag)
        df[f"humidity_lag_{lag}"] = df.groupby("district")["humidity_pct"].shift(lag)

    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52.0)

    df = df.dropna().reset_index(drop=True)
    return df


def compute_baseline_cases(df: pd.DataFrame, train_years: list) -> pd.DataFrame:
    base = (
        df[df["year"].isin(train_years)]
        .groupby(["district", "week"])["cases"]
        .mean()
        .reset_index()
        .rename(columns={"cases": "historical_avg_cases"})
    )
    return base


def compute_baseline_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average weather by district + week across all historical years.
    Used as 'typical' weather for future weeks.
    """
    base = (
        df.groupby(["district", "week"])[["rainfall_mm", "temp_avg_c", "humidity_pct"]]
        .mean()
        .reset_index()
        .rename(columns={
            "rainfall_mm": "clim_rainfall_mm",
            "temp_avg_c": "clim_temp_avg_c",
            "humidity_pct": "clim_humidity_pct",
        })
    )
    return base


def add_week_forward(year: int, week: int, delta_weeks: int = 1):
    """
    Move (year, week) forward by delta_weeks assuming 52 weeks per year.
    (Simple approximation, good enough for planning.)
    """
    new_week = week + delta_weeks
    new_year = year
    while new_week > 52:
        new_week -= 52
        new_year += 1
    return new_year, new_week


# ----------------- Main ----------------- #

def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Combined dataset not found: {DATA_FILE}")

    print(f"Loading data from {DATA_FILE} ...")
    df = pd.read_csv(DATA_FILE)

    df["district"] = df["district"].astype(str).str.strip()
    df["year"] = df["year"].astype(int)
    df["week"] = df["week"].astype(int)
    df["cases"] = df["cases"].astype(float)

    df = df.dropna(subset=["rainfall_mm", "temp_avg_c", "humidity_pct"])

    # Feature engineering
    df_feat = add_lag_features(df)

    feature_cols = [
        "rainfall_mm", "temp_avg_c", "humidity_pct",
        "cases_lag_1", "cases_lag_2", "cases_lag_3", "cases_lag_4",
        "rainfall_lag_1", "rainfall_lag_2", "rainfall_lag_3", "rainfall_lag_4",
        "temp_lag_1", "temp_lag_2", "temp_lag_3", "temp_lag_4",
        "humidity_lag_1", "humidity_lag_2", "humidity_lag_3", "humidity_lag_4",
        "week_sin", "week_cos",
    ]
    target_col = "cases"

    df_feat = df_feat.dropna(subset=feature_cols + [target_col])

    all_years = sorted(df_feat["year"].unique())
    if len(all_years) < 2:
        raise RuntimeError("Need at least 2 years of data for train/test split.")

    last_year = max(all_years)
    train_years = [y for y in all_years if y < last_year]

    train_mask = df_feat["year"].isin(train_years)
    test_mask = df_feat["year"] == last_year  # last year used just for evaluation (optional)

    train = df_feat[train_mask].copy()
    test = df_feat[test_mask].copy()

    if train.empty:
        raise RuntimeError("Train set is empty after split.")
    if test.empty:
        print("Warning: no explicit test set for last year; will still train on previous years.")

    X_train = train[feature_cols]
    y_train = train[target_col]

    print(f"Training on years: {train_years}")
    print(f"Training samples: {len(X_train)}")

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=16,
        random_state=42,
        n_jobs=-1
    )
    print("Training model...")
    model.fit(X_train, y_train)

    # Optional quick eval on last_year
    if not test.empty:
        X_test = test[feature_cols]
        y_test = test[target_col]
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        print(f"Quick eval on year {last_year}: MAE={mae:.2f}, RMSE={rmse:.2f}")

    # Baselines
    baseline_cases = compute_baseline_cases(df, train_years=train_years)
    baseline_weather = compute_baseline_weather(df)

    # For each district, find latest (year, week)
    latest_week = (
        df.groupby("district")[["year", "week"]]
        .max()
        .reset_index()
    )

    forecasts = []

    for _, row in latest_week.iterrows():
        district = row["district"]
        last_y = int(row["year"])
        last_w = int(row["week"])

        print(f"\nForecasting next 2 weeks for {district} (last data: year={last_y}, week={last_w})")

        # Get last 4 weeks of history for this district
        hist = (
            df[(df["district"] == district)]
            .sort_values(["year", "week"])
        )

        # We need last 4 rows to build lags
        hist_tail = hist.tail(4).copy()
        if len(hist_tail) < 4:
            print(f"  Not enough history for {district}, skipping.")
            continue

        # Prepare sequences for lags as lists (ordered oldest→newest)
        cases_seq = hist_tail["cases"].tolist()
        rain_seq = hist_tail["rainfall_mm"].tolist()
        temp_seq = hist_tail["temp_avg_c"].tolist()
        hum_seq = hist_tail["humidity_pct"].tolist()

        cur_year, cur_week = last_y, last_w

        for horizon in [1, 2]:
            # Move forward one week each loop
            cur_year, cur_week = add_week_forward(cur_year, cur_week, delta_weeks=1)

            # Get "typical" future weather from climatology baseline
            clim = baseline_weather[
                (baseline_weather["district"] == district) &
                (baseline_weather["week"] == cur_week)
            ]

            if clim.empty:
                # fallback: use last observed week's weather
                clim_rain = rain_seq[-1]
                clim_temp = temp_seq[-1]
                clim_hum = hum_seq[-1]
            else:
                clim_rain = float(clim["clim_rainfall_mm"].iloc[0])
                clim_temp = float(clim["clim_temp_avg_c"].iloc[0])
                clim_hum = float(clim["clim_humidity_pct"].iloc[0])

            # Build feature row
            feat = {}

            feat["rainfall_mm"] = clim_rain
            feat["temp_avg_c"] = clim_temp
            feat["humidity_pct"] = clim_hum

            # Lags from sequences (last 4 weeks)
            feat["cases_lag_1"] = cases_seq[-1]
            feat["cases_lag_2"] = cases_seq[-2]
            feat["cases_lag_3"] = cases_seq[-3]
            feat["cases_lag_4"] = cases_seq[-4]

            feat["rainfall_lag_1"] = rain_seq[-1]
            feat["rainfall_lag_2"] = rain_seq[-2]
            feat["rainfall_lag_3"] = rain_seq[-3]
            feat["rainfall_lag_4"] = rain_seq[-4]

            feat["temp_lag_1"] = temp_seq[-1]
            feat["temp_lag_2"] = temp_seq[-2]
            feat["temp_lag_3"] = temp_seq[-3]
            feat["temp_lag_4"] = temp_seq[-4]

            feat["humidity_lag_1"] = hum_seq[-1]
            feat["humidity_lag_2"] = hum_seq[-2]
            feat["humidity_lag_3"] = hum_seq[-3]
            feat["humidity_lag_4"] = hum_seq[-4]

            feat["week"] = cur_week
            feat["week_sin"] = np.sin(2 * np.pi * cur_week / 52.0)
            feat["week_cos"] = np.cos(2 * np.pi * cur_week / 52.0)

            # Predict
            X_f = pd.DataFrame([feat])[feature_cols]
            pred_cases = float(model.predict(X_f)[0])

            # Update sequences for next horizon
            cases_seq.append(pred_cases)
            rain_seq.append(clim_rain)
            temp_seq.append(clim_temp)
            hum_seq.append(clim_hum)

            # Keep only last 4 for next loop
            cases_seq = cases_seq[-4:]
            rain_seq = rain_seq[-4:]
            temp_seq = temp_seq[-4:]
            hum_seq = hum_seq[-4:]

            # Historical baseline for this district/week
            base_c = baseline_cases[
                (baseline_cases["district"] == district) &
                (baseline_cases["week"] == cur_week)
            ]
            if base_c.empty:
                hist_avg = np.nan
            else:
                hist_avg = float(base_c["historical_avg_cases"].iloc[0])

            beds = estimate_beds_needed(pred_cases)
            blood = estimate_blood_units_needed(pred_cases)
            risk = classify_risk(pred_cases, hist_avg)

            forecasts.append({
                "district": district,
                "target_year": cur_year,
                "target_week": cur_week,
                "horizon_weeks_ahead": horizon,
                "predicted_cases": pred_cases,
                "historical_avg_cases": hist_avg,
                "rainfall_mm_used": clim_rain,
                "temp_avg_c_used": clim_temp,
                "humidity_pct_used": clim_hum,
                "beds_needed": beds,
                "blood_units_needed": blood,
                "risk": risk,
            })

    if not forecasts:
        print("No forecasts produced (check data).")
        return

    forecast_df = pd.DataFrame(forecasts)

    # Generate recommended_actions text
    forecast_df["recommended_actions"] = forecast_df.apply(suggest_actions, axis=1)

    forecast_df.to_csv(OUT_MULTI, index=False)
    print(f"\n🎉 Saved next-2-week multi-district forecast to:\n  {OUT_MULTI}")


if __name__ == "__main__":
    main()
