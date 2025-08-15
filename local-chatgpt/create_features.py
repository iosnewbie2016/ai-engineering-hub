def create_features(df):
    df = df.sort_values(["AppName", "datetime"]).copy()

    # ========================
    # Hourly-level features
    # ========================
    for lag in [1, 24, 168]:  # 1 hour, 1 day, 1 week
        df[f"lag_{lag}h"] = df.groupby("AppName")["transactions"].shift(lag)

    for window in [3, 6, 24, 72, 168]:  # hours
        df[f"roll_mean_{window}h"] = (
            df.groupby("AppName")["transactions"]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )
        df[f"roll_std_{window}h"] = (
            df.groupby("AppName")["transactions"]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .std()
        )
        df[f"ema_{window}h"] = (
            df.groupby("AppName")["transactions"]
            .shift(1)
            .ewm(span=window, adjust=False)
            .mean()
        )

    # ========================
    # Daily-level aggregates
    # ========================
    df["date"] = df["datetime"].dt.date
    daily = (
        df.groupby(["AppName", "date"])["transactions"]
        .sum()
        .reset_index(name="transactions_daily")
    )
    # Daily lags
    for lag in [1, 7, 14]:
        daily[f"lag_{lag}d"] = (
            daily.groupby("AppName")["transactions_daily"].shift(lag)
        )

    # Daily rolling
    for window in [3, 7, 14]:
        daily[f"roll_mean_{window}d"] = (
            daily.groupby("AppName")["transactions_daily"]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )
        daily[f"ema_{window}d"] = (
            daily.groupby("AppName")["transactions_daily"]
            .shift(1)
            .ewm(span=window, adjust=False)
            .mean()
        )

    # Merge daily features back to hourly
    df = df.merge(daily, on=["AppName", "date"], how="left")

    # ========================
    # Weekly-level aggregates
    # ========================
    df["week"] = df["datetime"].dt.isocalendar().week
    df["year"] = df["datetime"].dt.year
    weekly = (
        df.groupby(["AppName", "year", "week"])["transactions"]
        .sum()
        .reset_index(name="transactions_weekly")
    )
    for lag in [1, 4, 8]:
        weekly[f"lag_{lag}w"] = (
            weekly.groupby("AppName")["transactions_weekly"].shift(lag)
        )
    for window in [4, 8]:
        weekly[f"roll_mean_{window}w"] = (
            weekly.groupby("AppName")["transactions_weekly"]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )

    # Merge weekly features back
    df = df.merge(weekly, on=["AppName", "year", "week"], how="left")

    # Drop helper cols
    df.drop(columns=["date", "week", "year"], inplace=True)

    return df
