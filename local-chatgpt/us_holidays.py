import pandas as pd
from datetime import datetime
import holidays

# Set up US holidays for 2023–2025
us_holidays = holidays.US(years=[2023, 2024, 2025])

# Define date range
dates = pd.date_range(start="2023-01-01", end="2025-12-31")
calendar_df = pd.DataFrame({"date": dates})

# Base features
calendar_df["day_of_week"] = calendar_df["date"].dt.day_name()
calendar_df["is_weekend"] = calendar_df["day_of_week"].isin(["Saturday", "Sunday"])
calendar_df["is_holiday"] = calendar_df["date"].isin(us_holidays)
calendar_df["holiday_name"] = calendar_df["date"].map(us_holidays).fillna("")

# Add known logistics/retail events
prime_days = {
    "2023-07-11": "Amazon Prime Day",
    "2023-07-12": "Amazon Prime Day",
    "2023-10-10": "Prime Big Deal Days",
    "2023-10-11": "Prime Big Deal Days",
    "2024-07-16": "Amazon Prime Day",
    "2024-07-17": "Amazon Prime Day",
    "2024-10-08": "Prime Big Deal Days",
    "2024-10-09": "Prime Big Deal Days",
    "2025-07-15": "Amazon Prime Day",
    "2025-07-16": "Amazon Prime Day",
    "2025-10-07": "Prime Big Deal Days",
    "2025-10-08": "Prime Big Deal Days",
}
calendar_df["special_event"] = calendar_df["date"].astype(str).map(prime_days).fillna("")

# Combined event label
calendar_df["event"] = calendar_df[["holiday_name", "special_event"]].agg(
    lambda x: ", ".join(filter(None, x)), axis=1
)

# Optional: drop intermediate columns
calendar_df.drop(columns=["holiday_name", "special_event"], inplace=True)

# Save to CSV
calendar_df.to_csv("calendar.csv", index=False)
print("Saved calendar.csv")
