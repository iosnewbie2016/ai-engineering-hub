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


## V2 
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

def historical_fill(series, hist_mask):
    """Fill NaNs with ffill/bfill+median based only on historical (True in hist_mask)."""
    hist_values = series[hist_mask]
    median_val = hist_values.median()
    filled = series.copy()
    # Fill using only past data
    filled[hist_mask] = hist_values.ffill().bfill().fillna(median_val)
    # Leave inference NaNs untouched here — model can handle or you fill differently
    return filled

def safe_seasonal_decompose(series, period):
    """Compute seasonal decomposition on past-only data."""
    if series.isna().all() or len(series) < period*2:
        return pd.Series([np.nan]*len(series), index=series.index), \
               pd.Series([np.nan]*len(series), index=series.index), \
               pd.Series([np.nan]*len(series), index=series.index)
    series_filled = series.ffill().bfill()
    decomp = seasonal_decompose(series_filled, model='additive', period=period, extrapolate_trend='freq')
    return decomp.trend, decomp.seasonal, decomp.resid

def create_features(df, date_col='datetime', lag_days=4, last_known_date=None):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(['AppName', date_col])

    # Merge holiday info
    us_calendar = pd.read_csv('calendar.csv', parse_dates=['date'])
    df = df.merge(us_calendar.rename(columns={'date': date_col})[[date_col,'is_holiday']],
                  on=date_col, how='left')

    # Date parts
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day_of_month'] = df[date_col].dt.day
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
    df['is_weekend_or_holiday'] = ((df['is_weekend']==1)|(df['is_holiday']==1)).astype(int)
    df['is_weekday_afternoon_peak'] = ((df['day_of_week'] < 5) &
                                       (df[date_col].dt.hour >= 16) &
                                       (df[date_col].dt.hour < 18)).astype(int)

    # Mask for historical data
    if last_known_date is not None:
        hist_mask = df[date_col] <= last_known_date
    else:
        hist_mask = pd.Series(True, index=df.index)

    # Seasonal decomposition per AppName (past only for inference)
    trend_all, seasonal_all, resid_all = [], [], []
    for app, group in df.groupby('AppName', group_keys=False):
        if last_known_date is not None:
            gmask = (group[date_col] <= last_known_date)
            hist_trend, hist_seasonal, hist_resid = safe_seasonal_decompose(group.loc[gmask, 'transactions'], period=24)
            trend = pd.Series(np.nan, index=group.index)
            seasonal = pd.Series(np.nan, index=group.index)
            resid = pd.Series(np.nan, index=group.index)
            trend.loc[gmask] = hist_trend
            seasonal.loc[gmask] = hist_seasonal
            resid.loc[gmask] = hist_resid
        else:
            trend, seasonal, resid = safe_seasonal_decompose(group['transactions'], period=24)
        trend_all.append(trend.interpolate())
        seasonal_all.append(seasonal.interpolate())
        resid_all.append(resid.interpolate())
    df['trend'] = pd.concat(trend_all).sort_index()
    df['seasonal'] = pd.concat(seasonal_all).sort_index()
    df['residual'] = pd.concat(resid_all).sort_index()

    # Lags
    for i in range(1, lag_days + 1):
        df[f'lag_{i}'] = df.groupby('AppName')['transactions'].shift(i*24)
        df[f'lag_peak_{i}'] = df.groupby('AppName')['transactions'].shift(i*24).where(df['is_weekday_afternoon_peak']==1)
    for i in [7, 14]:
        df[f'lag_{i}'] = df.groupby('AppName')['transactions'].shift(i*24)

    # Rolling & EMA
    for win in [7,14,21]:
        df[f'rolling_mean_{win}'] = df.groupby('AppName')['transactions'].shift(1).rolling(window=win*24, min_periods=1).mean()
        df[f'rolling_std_{win}']  = df.groupby('AppName')['transactions'].shift(1).rolling(window=win*24, min_periods=1).std()
        df[f'rolling_max_{win}']  = df.groupby('AppName')['transactions'].shift(1).rolling(window=win*24, min_periods=1).max()
        df[f'rolling_min_{win}']  = df.groupby('AppName')['transactions'].shift(1).rolling(window=win*24, min_periods=1).min()
    for win in [7,14,21]:
        df[f'rolling_mean_peak_{win}'] = (
            df.groupby('AppName')['transactions']
              .apply(lambda x: x.where(df['is_weekday_afternoon_peak'] == 1)
                                .shift(1)
                                .rolling(window=win*24, min_periods=1)
                                .mean())
        )
    for span in [3,7]:
        df[f'ema_{span}'] = df.groupby('AppName')['transactions'].shift(1).ewm(span=span*24, adjust=False).mean()

    # Historical filling for model stability (medians computed from history only)
    for col in df.columns:
        if col.startswith(('lag_', 'rolling_', 'ema_')) and not col.endswith('_peak'):  # skip special cases or leave in if needed
            df[col] = historical_fill(df[col], hist_mask)

    return df


def multi_date_parity_test(full_df, test_dates, feature_cols):
    results = {}

    for test_date in test_dates:
        last_actual_date = test_date - pd.Timedelta(days=1)

        # 1. As in validation
        features_val = create_features(full_df)
        val_day = features_val[features_val['datetime'].dt.date == test_date.date()]

        # 2. Simulated inference
        hist_until = full_df[full_df['datetime'] <= last_actual_date]
        apps = hist_until['AppName'].unique()
        future_times = pd.date_range(test_date, periods=24, freq='H')
        placeholder = pd.DataFrame([(t, a, 0) for a in apps for t in future_times],
                                   columns=['datetime', 'AppName', 'transactions'])
        sim_input = pd.concat([hist_until, placeholder], ignore_index=True)
        infer_all = create_features(sim_input, last_known_date=last_actual_date)
        infer_day = infer_all[infer_all['datetime'].dt.date == test_date.date()]

        # Align compare
        val_day = val_day.sort_values(['AppName','datetime'])[feature_cols].reset_index(drop=True)
        infer_day = infer_day.sort_values(['AppName','datetime'])[feature_cols].reset_index(drop=True)

        diff = (val_day - infer_day).abs()
        max_diff = diff.max()
        mismatched = max_diff[max_diff > 1e-9]

        results[test_date] = {
            'pass': mismatched.empty,
            'max_diff': max_diff.max(),
            'mismatched_features': list(mismatched.index)
        }

    return results

# Example usage:
test_dates = [
    pd.Timestamp("2025-06-05"),
    pd.Timestamp("2025-06-10"),
    pd.Timestamp("2025-06-15")
]
feature_cols = [c for c in your_model_feature_list]  # use the model's feature columns only
report = multi_date_parity_test(full_df, test_dates, feature_cols)

for date, info in report.items():
    print(f"{date.date()} - {'✅ PASS' if info['pass'] else '❌ FAIL'} - Max diff: {info['max_diff']}")
    if not info['pass']:
        print("  Mismatched features:", info['mismatched_features'])


# V3
def create_features(df, date_col='datetime', lag_days=4, last_known_date=None):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(['AppName', date_col])

    # Merge holiday info
    us_calendar = pd.read_csv('calendar.csv', parse_dates=['date'])
    df = df.merge(us_calendar.rename(columns={'date': date_col})[[date_col, 'is_holiday']],
                  on=date_col, how='left')

    # Date parts
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day_of_month'] = df[date_col].dt.day
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['hour'] = df[date_col].dt.hour
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
    df['is_weekend_or_holiday'] = ((df['is_weekend'] == 1) | (df['is_holiday'] == 1)).astype(int)
    df['is_weekday_afternoon_peak'] = ((df['day_of_week'] < 5) &
                                       (df['hour'] >= 16) & (df['hour'] < 18)).astype(int)

    # Historical mask
    hist_mask = df[date_col] <= last_known_date if last_known_date is not None else pd.Series(True, index=df.index)

    # Seasonal decomposition
    trend_all, seasonal_all, resid_all = [], [], []
    for app, g in df.groupby('AppName', group_keys=False):
        gmask = g[date_col] <= last_known_date if last_known_date is not None else pd.Series(True, index=g.index)
        hist_trend, hist_seasonal, hist_resid = safe_seasonal_decompose(g.loc[gmask, 'transactions'], period=24)

        # Forward-fill seasonal pattern for inference
        extended_seasonal = pd.Series(np.tile(hist_seasonal.dropna().values, int(np.ceil(len(g)/24)))[:len(g)], index=g.index)
        trend_all.append(hist_trend.reindex(g.index).ffill().bfill())
        seasonal_all.append(extended_seasonal)
        resid_all.append(hist_resid.reindex(g.index))

    df['trend'] = pd.concat(trend_all)
    df['seasonal'] = pd.concat(seasonal_all)
    df['residual'] = pd.concat(resid_all)

    # Lags
    for i in range(1, lag_days + 1):
        df[f'lag_{i}'] = df.groupby('AppName', group_keys=False)['transactions'].shift(i*24)
        df[f'lag_peak_{i}'] = df.groupby('AppName', group_keys=False).apply(
            lambda g: g['transactions'].shift(i*24).where(g['is_weekday_afternoon_peak'] == 1)
        )

    for i in [7, 14]:
        df[f'lag_{i}'] = df.groupby('AppName', group_keys=False)['transactions'].shift(i*24)

    # Rolling & EMA
    for win in [7, 14, 21]:
        df[f'rolling_mean_{win}'] = df.groupby('AppName', group_keys=False).apply(
            lambda g: g['transactions'].shift(1).rolling(window=win*24, min_periods=1).mean()
        )
        df[f'rolling_std_{win}'] = df.groupby('AppName', group_keys=False).apply(
            lambda g: g['transactions'].shift(1).rolling(window=win*24, min_periods=1).std()
        )
        df[f'rolling_max_{win}'] = df.groupby('AppName', group_keys=False).apply(
            lambda g: g['transactions'].shift(1).rolling(window=win*24, min_periods=1).max()
        )
        df[f'rolling_min_{win}'] = df.groupby('AppName', group_keys=False).apply(
            lambda g: g['transactions'].shift(1).rolling(window=win*24, min_periods=1).min()
        )

    for span in [3, 7]:
        df[f'ema_{span}'] = df.groupby('AppName', group_keys=False).apply(
            lambda g: g['transactions'].shift(1).ewm(span=span*24, adjust=False).mean()
        )

    # Historical fill
    for col in df.columns:
        if col.startswith(('lag_', 'rolling_', 'ema_')) and not col.endswith('_peak'):
            df[col] = historical_fill(df[col], hist_mask)

    return df


#V4
# Hourly-level features
for lag in [1, 24, 168]:  # 1h, 1d, 1w
    df[f"lag_{lag}h"] = (
        df.groupby("AppName", group_keys=False)
          .apply(lambda g: g["transactions"].shift(lag))
    )

for window in [3, 6, 24, 72, 168]:  # hours
    # Rolling mean
    df[f"roll_mean_{window}h"] = (
        df.groupby("AppName", group_keys=False)
          .apply(lambda g: g["transactions"].shift(1)
                                         .rolling(window=window, min_periods=1)
                                         .mean())
    )
    # Rolling std
    df[f"roll_std_{window}h"] = (
        df.groupby("AppName", group_keys=False)
          .apply(lambda g: g["transactions"].shift(1)
                                         .rolling(window=window, min_periods=1)
                                         .std())
    )
    # EMA
    df[f"ema_{window}h"] = (
        df.groupby("AppName", group_keys=False)
          .apply(lambda g: g["transactions"].shift(1)
                                         .ewm(span=window, adjust=False)
                                         .mean())
    )

# Daily-level features
df["date"] = df["datetime"].dt.date
daily = (
    df.groupby(["AppName", "date"], as_index=False)["transactions"]
      .sum()
      .rename(columns={"transactions": "transactions_daily"})
)

# Daily lags and rolling
for lag in [1, 7, 14]:
    daily[f"lag_{lag}d"] = (
        daily.groupby("AppName", group_keys=False)
             .apply(lambda g: g["transactions_daily"].shift(lag))
    )

for window in [3, 7, 14]:
    daily[f"roll_mean_{window}d"] = (
        daily.groupby("AppName", group_keys=False)
             .apply(lambda g: g["transactions_daily"].shift(1)
                                                 .rolling(window=window, min_periods=1)
                                                 .mean())
    )
    daily[f"ema_{window}d"] = (
        daily.groupby("AppName", group_keys=False)
             .apply(lambda g: g["transactions_daily"].shift(1)
                                                 .ewm(span=window, adjust=False)
                                                 .mean())
    )

# Merge daily features back to hourly
df = df.merge(daily, on=["AppName", "date"], how="left")

# Weekly-level features
df["week"] = df["datetime"].dt.isocalendar().week
df["year"] = df["datetime"].dt.year
weekly = (
    df.groupby(["AppName", "year", "week"], as_index=False)["transactions"]
      .sum()
      .rename(columns={"transactions": "transactions_weekly"})
)

# Weekly lags & rolling
for lag in [1, 4, 8]:
    weekly[f"lag_{lag}w"] = (
        weekly.groupby("AppName", group_keys=False)
              .apply(lambda g: g["transactions_weekly"].shift(lag))
    )
for window in [4, 8]:
    weekly[f"roll_mean_{window}w"] = (
        weekly.groupby("AppName", group_keys=False)
              .apply(lambda g: g["transactions_weekly"].shift(1)
                                                    .rolling(window=window, min_periods=1)
                                                    .mean())
    )

# Merge weekly back
df = df.merge(weekly, on=["AppName", "year", "week"], how="left")


# Final All merged
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

def historical_fill(series, hist_mask):
    """Fill NaNs using ffill/bfill + median, computed from historical portion only."""
    hist_values = series[hist_mask]
    median_val = hist_values.median()
    filled = series.copy()
    filled[hist_mask] = hist_values.ffill().bfill().fillna(median_val)
    return filled

def safe_seasonal_decompose(series, period):
    """Compute STL-like seasonal decomposition on history only."""
    if series.isna().all() or len(series) < period*2:
        return pd.Series(np.nan, index=series.index), pd.Series(np.nan, index=series.index), pd.Series(np.nan, index=series.index)
    filled = series.ffill().bfill()
    decomp = seasonal_decompose(filled, model='additive', period=period, extrapolate_trend='freq')
    return decomp.trend, decomp.seasonal, decomp.resid

def create_features(df, last_known_date=None):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['AppName', 'datetime'])

    # Holiday merge
    us_calendar = pd.read_csv('calendar.csv', parse_dates=['date'])
    df = df.merge(us_calendar.rename(columns={'date':'datetime'})[['datetime','is_holiday']], on='datetime', how='left')

    # Date parts
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day_of_month'] = df['datetime'].dt.day
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
    df['is_weekend_or_holiday'] = ((df['is_weekend']==1) | (df['is_holiday']==1)).astype(int)
    df['is_weekday_afternoon_peak'] = ((df['day_of_week'] < 5) & (df['datetime'].dt.hour>=16) & (df['datetime'].dt.hour<18)).astype(int)

    # Historical mask
    hist_mask = df['datetime'] <= last_known_date if last_known_date is not None else pd.Series(True, index=df.index)

    # Seasonal decomposition per app
    trends, seasonals, residuals = [], [], []
    for app, g in df.groupby('AppName', group_keys=False):
        if last_known_date is not None:
            gmask = g['datetime'] <= last_known_date
            t, s, r = safe_seasonal_decompose(g.loc[gmask,'transactions'], period=24)
            trend = pd.Series(np.nan, index=g.index); trend[gmask] = t
            seas  = pd.Series(np.nan, index=g.index); seas[gmask]  = s
            resid = pd.Series(np.nan, index=g.index); resid[gmask] = r
        else:
            trend, seas, resid = safe_seasonal_decompose(g['transactions'], period=24)
        trends.append(trend.interpolate())
        seasonals.append(seas.interpolate())
        residuals.append(resid.interpolate())
    df['trend'] = pd.concat(trends).sort_index()
    df['seasonal'] = pd.concat(seasonals).sort_index()
    df['residual'] = pd.concat(residuals).sort_index()

    # ===== Hourly features =====
    for lag in [1, 24, 168]:
        df[f'lag_{lag}h'] = df.groupby('AppName', group_keys=False).apply(lambda g: g['transactions'].shift(lag))
    for win in [3, 6, 24, 72, 168]:
        df[f'roll_mean_{win}h'] = df.groupby('AppName', group_keys=False).apply(lambda g: g['transactions'].shift(1).rolling(win, min_periods=1).mean())
        df[f'roll_std_{win}h']  = df.groupby('AppName', group_keys=False).apply(lambda g: g['transactions'].shift(1).rolling(win, min_periods=1).std())
        df[f'ema_{win}h']       = df.groupby('AppName', group_keys=False).apply(lambda g: g['transactions'].shift(1).ewm(span=win, adjust=False).mean())

    # ===== Daily aggregates =====
    df['date'] = df['datetime'].dt.date
    daily = df.groupby(['AppName','date'], as_index=False)['transactions'].sum().rename(columns={'transactions':'transactions_daily'})
    for lag in [1, 7, 14]:
        daily[f'lag_{lag}d'] = daily.groupby('AppName', group_keys=False).apply(lambda g: g['transactions_daily'].shift(lag))
    for win in [3, 7, 14]:
        daily[f'roll_mean_{win}d'] = daily.groupby('AppName', group_keys=False).apply(lambda g: g['transactions_daily'].shift(1).rolling(win,min_periods=1).mean())
        daily[f'ema_{win}d']       = daily.groupby('AppName', group_keys=False).apply(lambda g: g['transactions_daily'].shift(1).ewm(span=win, adjust=False).mean())
    df = df.merge(daily, on=['AppName','date'], how='left')

    # ===== Weekly aggregates =====
    df['week'] = df['datetime'].dt.isocalendar().week
    df['year'] = df['datetime'].dt.year
    weekly = df.groupby(['AppName','year','week'], as_index=False)['transactions'].sum().rename(columns={'transactions':'transactions_weekly'})
    for lag in [1, 4, 8]:
        weekly[f'lag_{lag}w'] = weekly.groupby('AppName', group_keys=False).apply(lambda g: g['transactions_weekly'].shift(lag))
    for win in [4, 8]:
        weekly[f'roll_mean_{win}w'] = weekly.groupby('AppName', group_keys=False).apply(lambda g: g['transactions_weekly'].shift(1).rolling(win,min_periods=1).mean())
    df = df.merge(weekly, on=['AppName','year','week'], how='left')

    # ===== Historical filling =====
    for col in df.columns:
        if any(col.startswith(prefix) for prefix in ('lag_', 'roll_', 'ema_')):
            df[col] = historical_fill(df[col], hist_mask)

    return df

