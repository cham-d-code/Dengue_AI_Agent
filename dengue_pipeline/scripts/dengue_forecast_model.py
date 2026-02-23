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

DATA_FILE = os.path.join(PROJECT_ROOT, "data_processed", "weekly_district_dataset_2021_2024.csv")
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


# ----------------------------------------------------
# National-Scale Configuration
# ----------------------------------------------------

# Ecological Zone Classification (Climate-based clustering)
ECOLOGICAL_ZONES = {
    # Wet Zone - High rainfall sensitivity
    'wet': ['Colombo', 'Gampaha', 'Kalutara', 'Galle', 'Matara', 'Ratnapura', 'Kegalle'],
    
    # Dry Zone - Water storage & humidity critical
    'dry': ['Jaffna', 'Kilinochchi', 'Mannar', 'Vavuniya', 'Mullaitivu', 
            'Trincomalee', 'Batticaloa', 'Ampara', 'Hambantota', 'Puttalam', 'Anuradhapura', 'Polonnaruwa'],
    
    # Hill Country - Temperature-limited
    'hill': ['Kandy', 'Matale', 'Nuwara Eliya', 'Badulla', 'Monaragala']
}

# District Neighbors (for spatial lag calculation)
DISTRICT_NEIGHBORS = {
    'Colombo': ['Gampaha', 'Kalutara'],
    'Gampaha': ['Colombo', 'Kurunegala', 'Kegalle'],
    'Kalutara': ['Colombo', 'Galle', 'Ratnapura'],
    'Galle': ['Kalutara', 'Matara', 'Ratnapura'],
    'Matara': ['Galle', 'Hambantota'],
    'Hambantota': ['Matara', 'Monaragala', 'Ratnapura'],
    'Kandy': ['Matale', 'Kegalle', 'Nuwara Eliya', 'Badulla'],
    'Matale': ['Kandy', 'Anuradhapura', 'Polonnaruwa'],
    'Nuwara Eliya': ['Kandy', 'Badulla', 'Ratnapura'],
    'Badulla': ['Kandy', 'Nuwara Eliya', 'Monaragala', 'Ampara'],
    'Monaragala': ['Badulla', 'Hambantota', 'Ampara'],
    'Ratnapura': ['Kalutara', 'Galle', 'Hambantota', 'Kegalle', 'Nuwara Eliya'],
    'Kegalle': ['Gampaha', 'Kurunegala', 'Kandy', 'Ratnapura'],
    'Kurunegala': ['Gampaha', 'Kegalle', 'Matale', 'Puttalam', 'Anuradhapura'],
    'Puttalam': ['Kurunegala', 'Anuradhapura'],
    'Anuradhapura': ['Puttalam', 'Kurunegala', 'Matale', 'Polonnaruwa', 'Vavuniya', 'Mannar'],
    'Polonnaruwa': ['Anuradhapura', 'Matale', 'Batticaloa', 'Ampara', 'Trincomalee'],
    'Trincomalee': ['Polonnaruwa', 'Batticaloa', 'Vavuniya'],
    'Batticaloa': ['Polonnaruwa', 'Trincomalee', 'Ampara'],
    'Ampara': ['Polonnaruwa', 'Batticaloa', 'Badulla', 'Monaragala'],
    'Jaffna': ['Kilinochchi'],
    'Kilinochchi': ['Jaffna', 'Mullaitivu', 'Vavuniya'],
    'Mullaitivu': ['Kilinochchi', 'Vavuniya'],
    'Vavuniya': ['Kilinochchi', 'Mullaitivu', 'Anuradhapura', 'Trincomalee'],
    'Mannar': ['Anuradhapura']
}

def get_ecological_zone(district: str) -> str:
    """Get the ecological zone for a district"""
    for zone, districts in ECOLOGICAL_ZONES.items():
        if district in districts:
            return zone
    return 'unknown'


def apply_biological_constraints(predicted_cases: float, temp: float) -> float:
    """
    Apply biological hard limits.
    If temp > 34 or temp < 16, survival drops significantly.
    Penalty: reduce prediction by 60%.
    """
    if temp > 34 or temp < 16:
        return predicted_cases * 0.4
    return predicted_cases


def add_spatial_lag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add spatial lag features - cases from neighboring districts.
    This captures the 'spillover effect' from high-transmission areas.
    """
    df = df.sort_values(["district", "year", "week"]).copy()
    
    # Calculate neighbor cases for each district-week
    neighbor_cases = []
    
    for idx, row in df.iterrows():
        district = row['district']
        year = row['year']
        week = row['week']
        
        # Get neighbors
        neighbors = DISTRICT_NEIGHBORS.get(district, [])
        
        if neighbors:
            # Get cases from neighbors in the same week
            neighbor_data = df[
                (df['district'].isin(neighbors)) & 
                (df['year'] == year) & 
                (df['week'] == week)
            ]
            
            if not neighbor_data.empty:
                avg_neighbor_cases = neighbor_data['cases'].mean()
            else:
                avg_neighbor_cases = 0
        else:
            avg_neighbor_cases = 0
        
        neighbor_cases.append(avg_neighbor_cases)
    
    df['neighbor_cases_avg'] = neighbor_cases
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features for cases and weather, plus seasonal week-of-year features.
    Enhanced with spatial and ecological features for national-scale modeling.
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

    # --- ANTIGRAVITY FEATURES ---

    # 1. Rolling Rain 4wk (Cumulative Rain over last 4 weeks)
    df["rain_sum_4w"] = df.groupby("district")["rainfall_mm"].transform(
        lambda x: x.rolling(window=4, min_periods=1).sum()
    )

    # 2. Case Velocity (Momentum)
    df["case_velocity"] = (df["cases_lag_1"] - df["cases_lag_3"]) / (df["cases_lag_3"] + 1e-6)

    # 3. Humidity/Rain Interaction (Refined "Hydro-Humid" Interaction)
    if "rainfall_lag_2" in df.columns and "humidity_lag_2" in df.columns:
        df["rain_hum_interaction"] = df["rainfall_lag_2"] * (df["humidity_lag_2"] / 100.0)

    # 4. The "Epidemic Trigger" (Source * Opportunity)
    if "cases_lag_1" in df.columns and "rainfall_lag_2" in df.columns:
        df["cases_rain_interaction"] = df["cases_lag_1"] * df["rainfall_lag_2"]

    # --- NATIONAL-SCALE FEATURES ---
    
    # 5. Ecological Zone Encoding
    df['eco_zone'] = df['district'].apply(get_ecological_zone)
    df['is_wet_zone'] = (df['eco_zone'] == 'wet').astype(int)
    df['is_dry_zone'] = (df['eco_zone'] == 'dry').astype(int)
    df['is_hill_zone'] = (df['eco_zone'] == 'hill').astype(int)
    
    # 6. District One-Hot Encoding (for baseline learning)
    district_dummies = pd.get_dummies(df['district'], prefix='dist')
    df = pd.concat([df, district_dummies], axis=1)
    
    # 7. Spatial Lag (Neighbor Effect)
    df = add_spatial_lag(df)
    df['neighbor_cases_lag_1'] = df.groupby("district")["neighbor_cases_avg"].shift(1)
    
    # 8. Ecological Suitability Index (zone-specific weather weighting)
    # Wet zone: rainfall dominant
    # Dry zone: humidity dominant  
    # Hill: temperature dominant
    df['eco_suitability'] = 0.0
    
    wet_mask = df['is_wet_zone'] == 1
    dry_mask = df['is_dry_zone'] == 1
    hill_mask = df['is_hill_zone'] == 1
    
    # Normalize to 0-1 scale for combination
    rain_norm = df['rain_sum_4w'] / (df['rain_sum_4w'].max() + 1)
    hum_norm = df['humidity_pct'] / 100.0
    temp_norm = (df['temp_avg_c'] - 16) / (34 - 16)  # Optimal range 16-34
    temp_norm = temp_norm.clip(0, 1)
    
    df.loc[wet_mask, 'eco_suitability'] = 0.6 * rain_norm[wet_mask] + 0.3 * hum_norm[wet_mask] + 0.1 * temp_norm[wet_mask]
    df.loc[dry_mask, 'eco_suitability'] = 0.3 * rain_norm[dry_mask] + 0.5 * hum_norm[dry_mask] + 0.2 * temp_norm[dry_mask]
    df.loc[hill_mask, 'eco_suitability'] = 0.2 * rain_norm[hill_mask] + 0.2 * hum_norm[hill_mask] + 0.6 * temp_norm[hill_mask]

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

    print("Adding lag, seasonal, and new derived features...")
    df_feat = add_lag_features(df)

    target_col = "cases"

    feature_cols = [
        "rainfall_mm", "temp_avg_c", "humidity_pct",
        "cases_lag_1", "cases_lag_2", "cases_lag_3", "cases_lag_4",
        "rainfall_lag_1", "rainfall_lag_2", "rainfall_lag_3", "rainfall_lag_4",
        "temp_lag_1", "temp_lag_2", "temp_lag_3", "temp_lag_4",
        "humidity_lag_1", "humidity_lag_2", "humidity_lag_3", "humidity_lag_4",
        "week_sin", "week_cos",
        "rain_sum_4w", "case_velocity", "rain_hum_interaction",
    ]

    df_feat = df_feat.dropna(subset=feature_cols + [target_col])

    # ----------------- Train on ALL years (2021-2024) ----------------- #

    all_years = sorted(df_feat["year"].unique())
    print("Years in dataset after feature engineering:", all_years)

    # Train on ALL available years for maximum data
    train_years = all_years  # Use all data for training
    
    # For evaluation reporting, use 2024 if available
    if 2024 in all_years:
        eval_year = 2024
    else:
        eval_year = max(all_years)

    train = df_feat.copy()  # Train on ALL data
    eval_set = df_feat[df_feat["year"] == eval_year].copy()  # Eval subset for metrics

    if train.empty:
        raise RuntimeError("Train set is empty. Check your data.")

    X_train = train[feature_cols]
    y_train = train[target_col]
    X_eval = eval_set[feature_cols]
    y_eval = eval_set[target_col]

    print(f"Training on ALL years: {train_years}")
    print(f"Training samples: {len(X_train)}, Eval samples ({eval_year}): {len(X_eval)}")


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

    print(f"Evaluating on {eval_year}...")
    y_pred_raw = model.predict(X_eval)
    
    # Apply biological constraints
    X_eval_temp = X_eval["temp_avg_c"].values 
    y_pred = []
    for pred, temp in zip(y_pred_raw, X_eval_temp):
        y_pred.append(apply_biological_constraints(pred, temp))
    y_pred = np.array(y_pred)

    mae = mean_absolute_error(y_eval, y_pred)
    mse = mean_squared_error(y_eval, y_pred)
    rmse = mse ** 0.5

    print(f"MAE ({eval_year})  = {mae:.2f} cases")
    print(f"RMSE ({eval_year}) = {rmse:.2f} cases")

    eval_set = eval_set.copy()
    eval_set["predicted_cases"] = y_pred

    # ----------------- Baseline & capacity ----------------- #

    baseline = compute_historical_baseline(df, train_years=train_years)

    eval_set = eval_set.merge(
        baseline,
        on=["district", "week"],
        how="left"
    )

    eval_set["beds_needed"] = eval_set["predicted_cases"].apply(estimate_beds_needed)
    eval_set["blood_units_needed"] = eval_set["predicted_cases"].apply(estimate_blood_units_needed)

    eval_set["risk"] = eval_set.apply(
        lambda r: classify_risk(r["predicted_cases"], r.get("historical_avg_cases", np.nan)),
        axis=1
    )

    eval_set["recommended_actions"] = eval_set.apply(suggest_actions, axis=1)

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
    cols_out = [c for c in cols_out if c in eval_set.columns]

    eval_set[cols_out].to_csv(OUT_FORECAST, index=False)
    print(f"\nSaved detailed forecast for year {eval_year} with capacity to:")
    print(f"  {OUT_FORECAST}")

    # Latest week per district in the eval year
    latest_per_district = (
        eval_set.sort_values(["district", "year", "week"])
        .groupby("district")
        .tail(1)
        .sort_values("district")
    )

    print(f"\n=== Latest available week per district ({eval_year}) ===")
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
