import os
import sys
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FORECAST_MULTI = os.path.join(PROJECT_ROOT, "data_processed", "multiweek_forecast_next_2_weeks.csv")
ALERTS_OUT = os.path.join(PROJECT_ROOT, "data_processed", "alerts_next_2_weeks.csv")


def main():
    if not os.path.exists(FORECAST_MULTI):
        raise FileNotFoundError(
            f"Multi-week forecast file not found: {FORECAST_MULTI}\n"
            f"Run forecast_next_two_weeks.py first."
        )

    df = pd.read_csv(FORECAST_MULTI)

    # We care only about high/critical risk
    alerts = df[df["risk"].isin(["high", "critical"])].copy()

    if alerts.empty:
        print("✅ No high/critical alerts in the next 2 weeks.")
        return

    # Sort by horizon then by predicted cases
    alerts = alerts.sort_values(
        ["horizon_weeks_ahead", "predicted_cases"],
        ascending=[True, False]
    )

    # Save to CSV (for dashboards, emails, etc.)
    alerts.to_csv(ALERTS_OUT, index=False)
    print(f"🎉 Saved alerts to:\n  {ALERTS_OUT}")

    # Pretty print summary
    print("\n=== EARLY WARNING: High/Critical dengue risk (next 2 weeks) ===")
    for _, row in alerts.iterrows():
        print("\n------------------------------")
        print(f"District      : {row['district']}")
        print(f"Target year   : {int(row['target_year'])}")
        print(f"Target week   : {int(row['target_week'])}")
        print(f"Horizon       : {int(row['horizon_weeks_ahead'])} week(s) ahead")
        print(f"Predicted cases : {row['predicted_cases']:.1f}")
        ha = row.get("historical_avg_cases")
        if pd.notna(ha):
            print(f"Historical avg : {ha:.1f}")
        print(f"Risk level       : {row['risk']}")
        print(f"Beds needed      : {row['beds_needed']:.1f}")
        print(f"Blood units need : {row['blood_units_needed']:.1f}")
        print("Recommended actions:")
        print(f"  {row['recommended_actions']}")
    print("\n===========================================================\n")


if __name__ == "__main__":
    main()
