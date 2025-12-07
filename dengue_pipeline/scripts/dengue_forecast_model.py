import os
import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------------------------------
# Path setup
# ----------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_FILE = os.path.join(PROJECT_ROOT, "data_processed", "weekly_district_dataset_2021_2025.csv")
OUT_FORECAST = os.path.join(PROJECT_ROOT, "data_processed", "forecast_with_capacity.csv")

# ----------------------------------------------------
# Capacity & risk config (you can tune these)
# ----------------------------------------------------

HOSPITALISATION_RATE = 0.15           # 15% of cases need admission
AVERAGE_LENGTH_OF_STAY_DAYS = 4       # average days in hospital
BLOOD_NEED_RATE = 0.05                # 5% of hospitalised need blood/platelets
BLOOD_UNITS_PER_PATIENT = 4           # units per patient (packed cells/platelets etc.)

HIGH_RISK_MULTIPLIER = 1.5            # >1.5x historical average = high
CRITICAL_RISK_MULTIPLIER = 2.0        # >2.0x historical average = critical


# ----------------------------------------------------
# Helper functions
# ----------------------------------------------------

def estimate_beds_needed(predicted_cases: float) -> float:
    """Estimate average number of beds needed for one week."""
    hospitalised = predicted_cases * HOSPITALISATION_RATE
    bed_days = hospitalised * AVERAGE_LENGTH_OF_STAY_DAYS
    beds_needed = bed_days / 7.0
    return beds_needed


def estimate_blood_units_needed(predicted_cases: float) -> float:
    """Estimate total blood units needed for one week."""
    hospitalised = predicted_cases * HOSPITALISATION_RATE
    patients_needing_blood = hospitalised * BLOOD_NEED_RATE
    units_needed = patients_needing_blood * BLOOD_UNITS_PER_PATIENT
    return units_needed


def classify_risk(predicted_cases: float, historical_avg: float) -> str:
    """Classify risk based on forecast vs historical baseline."""
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
    """Generate simple recommended actions string for MoH based on risk & capacity."""
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
            "with focused dengue prevention messages."
        )
        actions.append(
            "Alert PHI and MOH teams to increase field inspections and case follow-up."
        )
        if risk == "critical":
            actions.append(
                "Activate the district dengue emergency response plan and hold "
                "regular situation review meetings (e.g. twice per week)."
            )

    actions.append(
        f"Prepare inpatient capacity for approximately {beds_needed:.0f} beds "
        "for dengue patients (including step-down wards if required)."
    )
    actions.append(
        f"Coordinate with the blood bank to ensure around {blood_needed:.0f} "
        "blood/platelet units are available for dengue-related needs."
    )
    actions.append(
        "Enhance triage and early warning at OPD and emergency units, with "
        "clear referral/transfer pathways for severe cases."
    )

    return " ".join(actions)


# ----------------------------------------------------
# Feature engineering
# ----------------------------------------------------

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features for cases and weather, plus seasonal week-of-year features.
    Input df must at least contain: district, year, week, cases, rainfall_mm, temp_avg_c, humidity_pct.
    """
    df = df.sort_values(["district", "year", "week"]).copy()

    # Case lags
    for lag in [1, 2, 3, 4]:
        df[f"cases_lag_{lag}"] = df.groupby("district")["cases"].shift(lag)

    # Weather lags
    for lag in [1, 2, 3, 4]:
        df[f"rainfall_lag_{lag}"] = df.groupby("district")["rainfall_mm"].shift(lag)
        df[f"temp_lag_{lag}"] = df.groupby("district")["temp_avg_c"].shift(lag)
        df[f"humidity_lag_{lag}"] = df.groupby("district")["humidity_pct"].shift(lag)

    # Seasonal features
    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52.0)

    # Drop rows that don't have enough lag history
    df = df.dropna().reset_index(drop=True)
    return df


def compute_historical_baseline(df: pd.DataFrame, train_years: list) -> pd.DataFrame:
    """
    Compute historical average cases by district & week (using only training years).
    """
    base = (
        df[df["year"].isin(train_years)]
        .groupby(["district", "week"])["cases"]
        .mean()
        .reset_index()
        .rename(columns={"cases": "historical_avg_cases"})
    )
    return base


# ----------------------------------------------------
# Main
# ----------------------------------------------------

def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Combined dataset not found: {DATA_FILE}")

    print(f"Loading data from {DATA_FILE} ...")
    df = pd.read_csv(DATA_FILE)

    # Basic cleaning
    df["district"] = df["district"].astype(str).str.strip()
    df["year"] = df["year"].astype(int)
    df["week"] = df["week"].astype(int)
    df["cases"] = df["cases"].astype(float)

    # Remove rows with missing weather
    df = df.dropna(subset=["rainfall_mm", "temp_avg_c", "humidity_pct"])

    print("Adding lag and seasonal features...")
    df_feat = add_lag_features(df)

    target_col = "cases"

    feature_cols = [
        "rainfall_mm", "temp_avg_c", "humidity_pct",
        "cases_lag_1", "cases_lag_2", "cases_lag_3", "cases_lag_4",
        "rainfall_lag_1", "rainfall_lag_2", "rainfall_lag_3", "rainfall_lag_4",
        "temp_lag_1", "temp_lag_2", "temp_lag_3", "temp_lag_4",
        "humidity_lag_1", "humidity_lag_2", "humidity_lag_3", "humidity_lag_4",
        "week_sin", "week_cos",
    ]

    df_feat = df_feat.dropna(subset=feature_cols + [target_col])

    # ----------------- Train/test split by year ----------------- #

    all_years = sorted(df_feat["year"].unique())
    print("Years in dataset after feature engineering:", all_years)

    # Prefer to use 2024 as test year if available; otherwise use the latest year
    if 2024 in all_years:
        test_year = 2024
    else:
        test_year = max(all_years)

    train_years = [y for y in all_years if y < test_year]

    if not train_years:
        raise RuntimeError("No training years available (all data is in a single year).")

    train_mask = df_feat["year"].isin(train_years)
    test_mask = df_feat["year"] == test_year

    train = df_feat[train_mask].copy()
    test = df_feat[test_mask].copy()

    if train.empty:
        raise RuntimeError("Train set is empty after splitting. Check your data.")
    if test.empty:
        raise RuntimeError(f"Test set for year {test_year} is empty. Check your data.")

    X_train = train[feature_cols]
    y_train = train[target_col]
    X_test = test[feature_cols]
    y_test = test[target_col]

    print(f"Training on years: {train_years}, testing on year: {test_year}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # ----------------- Train model ----------------- #

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=16,
        random_state=42,
        n_jobs=-1
    )

    print("Training model...")
    model.fit(X_train, y_train)

    # ----------------- Evaluate ----------------- #

    print(f"Evaluating on {test_year}...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    print(f"MAE ({test_year})  = {mae:.2f} cases")
    print(f"RMSE ({test_year}) = {rmse:.2f} cases")

    test = test.copy()
    test["predicted_cases"] = y_pred

    # ----------------- Baseline & capacity ----------------- #

    baseline = compute_historical_baseline(df, train_years=train_years)

    test = test.merge(
        baseline,
        on=["district", "week"],
        how="left"
    )

    test["beds_needed"] = test["predicted_cases"].apply(estimate_beds_needed)
    test["blood_units_needed"] = test["predicted_cases"].apply(estimate_blood_units_needed)

    test["risk"] = test.apply(
        lambda r: classify_risk(r["predicted_cases"], r.get("historical_avg_cases", np.nan)),
        axis=1
    )

    test["recommended_actions"] = test.apply(suggest_actions, axis=1)

    # ----------------- Save & print summary ----------------- #

    cols_out = [
        "district", "year", "week",
        "cases", "predicted_cases",
        "historical_avg_cases",
        "rainfall_mm", "temp_avg_c", "humidity_pct",
        "beds_needed", "blood_units_needed",
        "risk", "recommended_actions",
    ]

    # Ensure only columns that actually exist
    cols_out = [c for c in cols_out if c in test.columns]

    test[cols_out].to_csv(OUT_FORECAST, index=False)
    print(f"\nSaved detailed forecast for year {test_year} with capacity to:")
    print(f"  {OUT_FORECAST}")

    # Latest week per district in the test year
    latest_per_district = (
        test.sort_values(["district", "year", "week"])
        .groupby("district")
        .tail(1)
        .sort_values("district")
    )

    print(f"\n=== Latest available week per district ({test_year}) ===")
    for _, row in latest_per_district.iterrows():
        print(f"\nDistrict: {row['district']}")
        print(f"  Year: {int(row['year'])}, Week: {int(row['week'])}")
        print(f"  Observed cases: {row['cases']:.1f}")
        print(f"  Predicted cases: {row['predicted_cases']:.1f}")
        ha = row.get("historical_avg_cases", np.nan)
        if not np.isnan(ha):
            print(f"  Historical avg (same week): {ha:.1f}")
        print(f"  Risk level: {row['risk']}")
        print(f"  Beds needed (estimate): {row['beds_needed']:.1f}")
        print(f"  Blood units needed (estimate): {row['blood_units_needed']:.1f}")
        print(f"  Recommended actions: {row['recommended_actions']}")


if __name__ == "__main__":
    main()
