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

#V6
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
    if series.isna().all() or len(series) < period*2:
        return pd.Series(np.nan, index=series.index), pd.Series(np.nan, index=series.index), pd.Series(np.nan, index=series.index)
    filled = series.ffill().bfill()
    decomp = seasonal_decompose(filled, model='additive', period=period, extrapolate_trend='freq')
    return decomp.trend, decomp.seasonal, decomp.resid

def create_features(df, last_known_date=None):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['AppName', 'datetime'])

    # Merge holiday info
    us_calendar = pd.read_csv('calendar.csv', parse_dates=['date'])
    df = df.merge(us_calendar.rename(columns={'date': 'datetime'})[['datetime','is_holiday']],
                  on='datetime', how='left')

    # Date parts
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day_of_month'] = df['datetime'].dt.day
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
    df['is_weekend_or_holiday'] = ((df['is_weekend']==1) | (df['is_holiday']==1)).astype(int)
    df['is_weekday_afternoon_peak'] = (
        (df['day_of_week'] < 5) &
        (df['datetime'].dt.hour >= 16) &
        (df['datetime'].dt.hour < 18)
    ).astype(int)

    # Historical mask for safe filling
    hist_mask = df['datetime'] <= last_known_date if last_known_date is not None else pd.Series(True, index=df.index)

    # Seasonal decomposition per AppName
    trends, seasonals, residuals = [], [], []
    for app, g in df.groupby('AppName', group_keys=False):
        if last_known_date is not None:
            gmask = g['datetime'] <= last_known_date
            t, s, r = safe_seasonal_decompose(g.loc[gmask, 'transactions'], period=24)
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

    # ===== Hourly Features =====
    for lag in [1, 24, 168]:
        df[f'lag_{lag}h'] = df.groupby('AppName', group_keys=False).apply(lambda g: g['transactions'].shift(lag))
    for win in [3, 6, 24, 72, 168]:
        df[f'roll_mean_{win}h'] = df.groupby('AppName', group_keys=False).apply(lambda g: g['transactions'].shift(1).rolling(win, min_periods=1).mean())
        df[f'roll_std_{win}h']  = df.groupby('AppName', group_keys=False).apply(lambda g: g['transactions'].shift(1).rolling(win, min_periods=1).std())
        df[f'ema_{win}h']       = df.groupby('AppName', group_keys=False).apply(lambda g: g['transactions'].shift(1).ewm(span=win, adjust=False).mean())

    # ===== Peak Lag ===== (in days × 24 hours)
    for i in [1, 2, 3, 4]:
        df[f'lag_peak_{i}'] = df.groupby('AppName', group_keys=False).apply(
            lambda g: g['transactions'].shift(i*24).where(g['is_weekday_afternoon_peak'] == 1)
        )

    # ===== Peak Rolling Mean =====
    for win in [7, 14, 21]:  # days
        df[f'rolling_mean_peak_{win}'] = df.groupby('AppName', group_keys=False).apply(
            lambda g: g['transactions'].where(g['is_weekday_afternoon_peak'] == 1)
                                     .shift(1)
                                     .rolling(window=win*24, min_periods=1)
                                     .mean()
        )

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

    # ===== Historical Filling =====
    for col in df.columns:
        if any(col.startswith(prefix) for prefix in ('lag_', 'roll_', 'ema_')):
            df[col] = historical_fill(df[col], hist_mask)

    return df


#V10
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

def _safe_seasonal_decompose(series, period=24):
    """Decompose on a filled copy; return (trend, seasonal, resid) aligned to original index."""
    if series.dropna().shape[0] < period * 2:
        n = len(series)
        idx = series.index
        z = pd.Series(np.nan, index=idx)
        return z, z, z
    filled = series.ffill().bfill()
    decomp = seasonal_decompose(filled, model="additive", period=period, extrapolate_trend="freq")
    # keep alignment to original index
    return (pd.Series(decomp.trend, index=series.index),
            pd.Series(decomp.seasonal, index=series.index),
            pd.Series(decomp.resid, index=series.index))

def _hist_fill_group(col_s, time_s, last_known_date):
    """
    Fill NaNs using only history for each group slice:
      - for rows <= last_known_date: ffill/bfill → median(hist)
      - for rows  > last_known_date: ffill from last history → median(hist)
    """
    if last_known_date is None:
        hist_mask = pd.Series(True, index=col_s.index)
    else:
        hist_mask = time_s <= last_known_date

    hist_vals = col_s[hist_mask]
    med = hist_vals.median()

    # fill history in-place
    out = col_s.copy()
    out.loc[hist_mask] = hist_vals.ffill().bfill().fillna(med)

    # fill forecast part with last historical value then median
    if last_known_date is not None and (~hist_mask).any():
        fut_idx = out.index[~hist_mask]
        # carry last known historical value forward
        last_hist_val = out.loc[hist_mask].iloc[-1] if hist_mask.any() else med
        out.loc[fut_idx] = out.loc[fut_idx].fillna(method="ffill")  # within future block
        out.loc[fut_idx] = out.loc[fut_idx].fillna(last_hist_val).fillna(med)

    return out

def create_features(
    df,
    date_col="datetime",
    lag_days=(1, 2, 3, 7, 14),
    hourly_lags=(),                 # e.g. (1,2,3,6,12) if you want short-term hourly effects
    rolling_days=(7, 14, 21),
    ema_days=(3, 7),
    peak_hours=range(16, 19),       # 16,17,18 (adjust to 16–18 as you observed)
    peak_weekdays=range(0, 5),      # Mon..Fri
    last_known_date=None,
    calendar_csv="calendar.csv",
):
    """
    Builds features for hourly data with daily-seasonal lags and peak-aware features.
    Pass last_known_date at inference to keep fills leakage-safe.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(["AppName", date_col]).reset_index(drop=True)

    # Merge holiday calendar
    cal = pd.read_csv(calendar_csv, parse_dates=["date"])
    cal = cal.rename(columns={"date": date_col})
    df = df.merge(cal[[date_col, "is_holiday"]], on=date_col, how="left")

    # Time parts
    df["year"]         = df[date_col].dt.year
    df["month"]        = df[date_col].dt.month
    df["day_of_month"] = df[date_col].dt.day
    df["day_of_week"]  = df[date_col].dt.dayofweek
    df["day_of_year"]  = df[date_col].dt.dayofyear
    df["hour"]         = df[date_col].dt.hour

    # Flags
    df["is_holiday"]  = df["is_holiday"].fillna(0).astype(int)
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_weekday_afternoon_peak"] = ((df["day_of_week"].isin(peak_weekdays)) &
                                       (df["hour"].isin(peak_hours))).astype(int)

    # Seasonal decomposition (per AppName) computed on history, extend to future
    trend_all, seas_all, resid_all = [], [], []
    for app, g in df.groupby("AppName", group_keys=False):
        # history mask within the group
        if last_known_date is not None:
            g_hist_mask = g[date_col] <= last_known_date
            hist_series = g.loc[g_hist_mask, "transactions"]
        else:
            g_hist_mask = pd.Series(True, index=g.index)
            hist_series = g["transactions"]

        tr, se, re = _safe_seasonal_decompose(hist_series, period=24)

        # reindex to full group and extend seasonal by repeating 24-hour pattern
        tr_full = tr.reindex(g.index).ffill().bfill()
        # build a 24-hour seasonal template from history
        seas_vals = se.dropna().values
        if len(seas_vals) >= 24:
            # take last complete 24h slice
            last_24 = seas_vals[-24:]
            seas_ext = np.resize(last_24, len(g))
        else:
            # fallback: use whatever we have, tile to length
            seas_ext = np.resize(seas_vals if len(seas_vals) > 0 else np.zeros(24), len(g))
        se_full = pd.Series(seas_ext, index=g.index)
        re_full = re.reindex(g.index)

        trend_all.append(tr_full)
        seas_all.append(se_full)
        resid_all.append(re_full)

    df["trend"]    = pd.concat(trend_all).sort_index()
    df["seasonal"] = pd.concat(seas_all).sort_index()
    df["residual"] = pd.concat(resid_all).sort_index()

    # ----- LAGS -----
    # Hourly lags (optional)
    for h in hourly_lags:
        df[f"lag_hour_{h}"] = df.groupby("AppName", group_keys=False)["transactions"].shift(h)

    # Daily lags: previous-day same-hour (your expected behavior)
    for d in lag_days:
        df[f"lag_day_{d}"] = df.groupby("AppName", group_keys=False)["transactions"].shift(d * 24)

    # Peak lags: same as daily lags but only retained at peak hours; NaN elsewhere
    for d in lag_days:
        df[f"lag_peak_day_{d}"] = (
            df.groupby("AppName", group_keys=False)
              .apply(lambda g: g["transactions"].shift(d * 24)
                     .where(g["is_weekday_afternoon_peak"] == 1))
        )

    # ----- ROLLING (day-scale windows) -----
    # These use shift(1) to avoid leakage and run inside each AppName group
    for win_d in rolling_days:
        win = win_d * 24
        df[f"rolling_mean_{win_d}d"] = (
            df.groupby("AppName", group_keys=False)
              .apply(lambda g: g["transactions"].shift(1).rolling(win, min_periods=1).mean())
        )
        df[f"rolling_std_{win_d}d"] = (
            df.groupby("AppName", group_keys=False)
              .apply(lambda g: g["transactions"].shift(1).rolling(win, min_periods=1).std())
        )
        df[f"rolling_max_{win_d}d"] = (
            df.groupby("AppName", group_keys=False)
              .apply(lambda g: g["transactions"].shift(1).rolling(win, min_periods=1).max())
        )
        df[f"rolling_min_{win_d}d"] = (
            df.groupby("AppName", group_keys=False)
              .apply(lambda g: g["transactions"].shift(1).rolling(win, min_periods=1).min())
        )

    # Peak rolling means (day windows over timeline, but only counting peak-hour observations)
    for win_d in rolling_days:
        win = win_d * 24
        df[f"rolling_mean_peak_{win_d}d"] = (
            df.groupby("AppName", group_keys=False)
              .apply(lambda g: g["transactions"]
                     .where(g["is_weekday_afternoon_peak"] == 1)
                     .shift(1).rolling(win, min_periods=1).mean())
        )

    # ----- EMA (day-scale spans) -----
    for span_d in ema_days:
        span = span_d * 24
        df[f"ema_{span_d}d"] = (
            df.groupby("AppName", group_keys=False)
              .apply(lambda g: g["transactions"].shift(1).ewm(span=span, adjust=False).mean())
        )

    # ----- Fill NaNs leakage-safe (per AppName) -----
    feat_like = [c for c in df.columns if c.startswith(("lag_", "rolling_", "ema_", "trend", "seasonal", "residual"))]
    for app, g in df.groupby("AppName"):
        idx = g.index
        for col in feat_like:
            df.loc[idx, col] = _hist_fill_group(df.loc[idx, col], g[date_col], last_known_date)

    return df




#V11 - Error fix
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
    if series.isna().all() or len(series) < period*2:
        return (
            pd.Series(np.nan, index=series.index),
            pd.Series(np.nan, index=series.index),
            pd.Series(np.nan, index=series.index)
        )
    filled = series.ffill().bfill()
    decomp = seasonal_decompose(filled, model='additive', period=period, extrapolate_trend='freq')
    return decomp.trend, decomp.seasonal, decomp.resid

def create_features(df, last_known_date=None):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['AppName', 'datetime'])

    # Merge holiday info
    us_calendar = pd.read_csv('calendar.csv', parse_dates=['date'])
    df = df.merge(us_calendar.rename(columns={'date': 'datetime'})[['datetime','is_holiday']],
                  on='datetime', how='left')

    # Date parts
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day_of_month'] = df['datetime'].dt.day
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
    df['is_weekend_or_holiday'] = ((df['is_weekend']==1) | (df['is_holiday']==1)).astype(int)
    df['is_weekday_afternoon_peak'] = (
        (df['day_of_week'] < 5) &
        (df['datetime'].dt.hour >= 16) &
        (df['datetime'].dt.hour < 18)
    ).astype(int)

    # Historical mask for safe filling
    hist_mask = df['datetime'] <= last_known_date if last_known_date is not None else pd.Series(True, index=df.index)

    # Seasonal decomposition per AppName
    trends, seasonals, residuals = [], [], []
    for app, g in df.groupby('AppName', group_keys=False):
        if last_known_date is not None:
            gmask = g['datetime'] <= last_known_date
            t, s, r = safe_seasonal_decompose(g.loc[gmask, 'transactions'], period=24)
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

    # ===== Hourly Features (transform for alignment safety) =====
    for lag in [1, 24, 168]:
        df[f'lag_{lag}h'] = df.groupby("AppName")["transactions"].transform(lambda x: x.shift(lag))

    for win in [3, 6, 24, 72, 168]:
        df[f'roll_mean_{win}h'] = df.groupby("AppName")["transactions"].transform(lambda x: x.shift(1).rolling(win, min_periods=1).mean())
        df[f'roll_std_{win}h']  = df.groupby("AppName")["transactions"].transform(lambda x: x.shift(1).rolling(win, min_periods=1).std())
        df[f'ema_{win}h']       = df.groupby("AppName")["transactions"].transform(lambda x: x.shift(1).ewm(span=win, adjust=False).mean())

    # ===== Peak Lag (apply for complex condition, reset index) =====
    for i in [1, 2, 3, 4]:
        df[f'lag_peak_{i}'] = (
            df.groupby('AppName')
              .apply(lambda g: g['transactions'].shift(i*24).where(g['is_weekday_afternoon_peak'] == 1))
              .reset_index(level=0, drop=True)
        )

    # ===== Peak Rolling Mean =====
    for win in [7, 14, 21]:  # days
        df[f'rolling_mean_peak_{win}'] = (
            df.groupby('AppName')
              .apply(lambda g: g['transactions'].where(g['is_weekday_afternoon_peak'] == 1)
                                .shift(1)
                                .rolling(window=win*24, min_periods=1)
                                .mean())
              .reset_index(level=0, drop=True)
        )

    # ===== Daily aggregates =====
    df['date'] = df['datetime'].dt.date
    daily = df.groupby(['AppName','date'], as_index=False)['transactions'].sum().rename(columns={'transactions':'transactions_daily'})
    for lag in [1, 7, 14]:
        daily[f'lag_{lag}d'] = daily.groupby("AppName")["transactions_daily"].transform(lambda x: x.shift(lag))
    for win in [3, 7, 14]:
        daily[f'roll_mean_{win}d'] = daily.groupby("AppName")["transactions_daily"].transform(lambda x: x.shift(1).rolling(win, min_periods=1).mean())
        daily[f'ema_{win}d']       = daily.groupby("AppName")["transactions_daily"].transform(lambda x: x.shift(1).ewm(span=win, adjust=False).mean())
    df = df.merge(daily, on=['AppName','date'], how='left')

    # ===== Weekly aggregates =====
    df['week'] = df['datetime'].dt.isocalendar().week
    df['year'] = df['datetime'].dt.year
    weekly = df.groupby(['AppName','year','week'], as_index=False)['transactions'].sum().rename(columns={'transactions':'transactions_weekly'})
    for lag in [1, 4, 8]:
        weekly[f'lag_{lag}w'] = weekly.groupby("AppName")["transactions_weekly"].transform(lambda x: x.shift(lag))
    for win in [4, 8]:
        weekly[f'roll_mean_{win}w'] = weekly.groupby("AppName")["transactions_weekly"].transform(lambda x: x.shift(1).rolling(win, min_periods=1).mean())
    df = df.merge(weekly, on=['AppName','year','week'], how='left')

    # ===== Historical Filling =====
    for col in df.columns:
        if any(col.startswith(prefix) for prefix in ('lag_', 'roll_', 'ema_')):
            df[col] = historical_fill(df[col], hist_mask)

    return df


# Error
# ===== Peak Lag =====
for i in [1, 2, 3, 4]:
    shifted = df.groupby("AppName")["transactions"].transform(lambda x: x.shift(i * 24))
    df[f"lag_peak_{i}"] = shifted.where(df["is_weekday_afternoon_peak"] == 1)

# ===== Peak Rolling Mean =====
for win in [7, 14, 21]:  # win in days
    masked = df["transactions"].where(df["is_weekday_afternoon_peak"] == 1)
    df[f"rolling_mean_peak_{win}"] = (
        masked.groupby(df["AppName"])
              .transform(lambda x: x.shift(1).rolling(window=win * 24, min_periods=1).mean())
    )

# Fix for lag and rolling issues

# Option 1- Recursive hour-by-hour inference utility
# Uses a working copy of transactions to feed predictions back as pseudo-history.
# Recomputes only the minimal feature row for each hour (fast and leakage-safe).
# Assumes create_features is your final leakage-safe function with last_known_date parameter.

import pandas as pd
import numpy as np

def predict_day_recursive(model, history_df, feature_fn, app_list, predict_date, model_features):
    """
    Recursive 24-step inference for one calendar date (00:00..23:00).
    - model: trained LightGBM model
    - history_df: full history up to last_known_ts (no placeholders), columns: ['datetime','AppName','transactions', ...]
    - feature_fn: your create_features(df, last_known_date=...) function
    - app_list: list/array of AppName to predict
    - predict_date: 'YYYY-MM-DD' date (string or Timestamp) to forecast
    - model_features: list of feature column names in the model

    Returns: DataFrame with columns ['datetime','AppName','y_pred']
    """
    predict_date = pd.to_datetime(predict_date).date()
    hours = pd.date_range(pd.Timestamp(predict_date), periods=24, freq="H")
    last_known_ts = history_df['datetime'].max()

    # Working "truth" that we augment with predictions as pseudo-history
    work = history_df.copy()

    preds = []

    for ts in hours:
        # Prepare 1-row inference df for this ts per app
        inf_rows = pd.DataFrame({
            'datetime': np.repeat(ts, len(app_list)),
            'AppName': app_list,
            'transactions': np.zeros(len(app_list))  # placeholder, never used directly by features due to shift(1)
        })

        # Append to work and build features with last_known_date fixed at historical max
        sim_input = pd.concat([work, inf_rows], ignore_index=True)
        feats_all = feature_fn(sim_input, last_known_date=last_known_ts)

        # Extract just the current hour's rows (one per app)
        feats_ts = feats_all[feats_all['datetime'] == ts].copy()

        # IMPORTANT: For recursive mode, we DO allow lag_1h etc. to draw from previous predicted values.
        # Because work contains prior predictions for earlier hours of this day.

        X = feats_ts[model_features]
        y_hat = model.predict(X)
        out = feats_ts[['datetime','AppName']].copy()
        out['y_pred'] = y_hat
        preds.append(out)

        # Write predictions back into work to be used as pseudo-history for next hour
        new_obs = out.rename(columns={'y_pred':'transactions'})[['datetime','AppName','transactions']]
        work = pd.concat([work, new_obs], ignore_index=True)

    preds = pd.concat(preds, ignore_index=True)
    return preds


# option 2 - Single-shot wrapper: null-out intra-day-dependent features
# Use one pass. For hours after 00:00 on the inference date, set features that require within-day history to NaN, so no illegal dependency on unknown values.
# Keep lag_24h, lag_168h, long rollings, daily/weekly aggregates, holidays, seasonality, and cyclical time encodings.
# Define which features depend on within-day history:
# Example names below; adjust based on your feature set.

def predict_day_single_shot(model, history_df, feature_fn, app_list, predict_date, model_features):
    """
    Non-recursive single-shot 24h forecast.
    - Sets intra-day dependent features to NaN for hours > 00:00 on the forecast day.
    """
    predict_date = pd.to_datetime(predict_date).date()
    hours = pd.date_range(pd.Timestamp(predict_date), periods=24, freq="H")
    last_known_ts = history_df['datetime'].max()

    # Build inference rows (placeholders) and compute all features once
    infer_rows = pd.DataFrame(
        {'datetime': np.repeat(hours, len(app_list)),
         'AppName': np.tile(app_list, len(hours)),
         'transactions': 0.0}
    )
    sim_input = pd.concat([history_df, infer_rows], ignore_index=True)
    feats_all = feature_fn(sim_input, last_known_date=last_known_ts)

    # Identify inference day rows
    mask_day = feats_all['datetime'].dt.date == predict_date
    feats_day = feats_all.loc[mask_day].copy()
    feats_day['hour'] = feats_day['datetime'].dt.hour

    # Define intra-day dependent features (examples; tailor to your names)
    intraday_feats = [
        'lag_1h', 'roll_mean_3h', 'roll_std_3h', 'ema_3h',
        'roll_mean_6h', 'roll_std_6h', 'ema_6h',
        'roll_mean_24h', 'roll_std_24h', 'ema_24h'
        # add any others that require within-day past (i.e., need 06-19 00:00 for 01:00, etc.)
    ]
    intraday_cols = [c for c in intraday_feats if c in feats_day.columns]

    # Null-out illegal intra-day features for hours > 0
    feats_day.loc[feats_day['hour'] > 0, intraday_cols] = np.nan

    # Predict with remaining valid features
    X = feats_day[model_features]
    y_hat = model.predict(X)

    out = feats_day[['datetime','AppName']].copy()
    out['y_pred'] = y_hat
    return out.reset_index(drop=True)

# Fix for residuals_by_dow
# Issues:
# residual_by_dow and residual_by_dow_smoothed were computed over the entire frame, so training may have used future info and inference distributions shift.
# The fix: compute baselines from strictly historical data per AppName (and excluding holidays if desired), then map back; apply winsorization only on historical residuals; for inference rows, do not recompute group stats using them.
# A robust approach:
# Compute per-AppName day_of_week robust medians on history only (non-holiday if desired).
# Map to both historical and inference rows by day_of_week (no use of future values).
# Winsorize residuals per-AppName using only historical residuals.
# Do not fill inference residuals using historical fill functions that recompute on entire DF; keep inference NaNs if they occur.

def robust_median(series, mad_threshold=3.0):
    m = series.median()
    mad = (series - m).abs().median()
    if mad == 0 or np.isnan(mad):
        return m
    lo = m - mad_threshold * mad
    hi = m + mad_threshold * mad
    return series.clip(lower=lo, upper=hi).median()

def compute_residual_by_dow(df, last_known_date=None, exclude_holidays=True):
    """
    Leakage-safe residual_by_dow and residual_by_dow_smoothed per AppName.
    - Baseline computed only from history (<= last_known_date if provided)
    - Optionally exclude holidays from baseline estimation
    - Winsorize residuals using historical portion only
    Returns df with residual_by_dow and residual_by_dow_smoothed.
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['AppName','datetime'])

    # Day-of-week
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['datetime'].dt.dayofweek
    if 'is_holiday' not in df.columns:
        df['is_holiday'] = 0

    if last_known_date is not None:
        hist_mask = df['datetime'] <= last_known_date
    else:
        hist_mask = pd.Series(True, index=df.index)

    # Build baselines per (AppName, DOW) from historical non-holiday rows if requested
    baselines = []
    for app, g in df.groupby('AppName'):
        g_hist = g.loc[hist_mask.loc[g.index]]
        if exclude_holidays:
            g_hist = g_hist[g_hist['is_holiday'].astype(bool) == False]

        if g_hist.empty:
            # Fallback: global median if no history
            baseline_map = {dow: g['transactions'].median() for dow in range(7)}
        else:
            baseline_map = {}
            for dow, dsub in g_hist.groupby('day_of_week'):
                if dsub.empty:
                    baseline_map[dow] = g_hist['transactions'].median()
                else:
                    baseline_map[dow] = robust_median(dsub['transactions'])
            # Fill any missing dows
            all_dows = set(range(7))
            missing = all_dows - set(baseline_map.keys())
            if missing:
                global_med = g_hist['transactions'].median()
                for dow in missing:
                    baseline_map[dow] = global_med

        # Create a per-group Series aligned to g.index
        base_series = g['day_of_week'].map(baseline_map)
        base_series.index = g.index
        baselines.append(base_series)

    baseline_all = pd.concat(baselines).sort_index()
    df['residual_by_dow'] = df['transactions'] - baseline_all

    # Winsorize residuals per AppName using history only
    smoothed = []
    for app, g in df.groupby('AppName'):
        g_hist_res = g.loc[hist_mask.loc[g.index], 'residual_by_dow']
        if g_hist_res.empty:
            # No history: leave as-is
            smoothed.append(pd.Series(g['residual_by_dow'].values, index=g.index))
            continue
        lo = g_hist_res.quantile(0.05)
        hi = g_hist_res.quantile(0.95)
        clipped = g['residual_by_dow'].clip(lower=lo, upper=hi)
        smoothed.append(clipped)

    df['residual_by_dow_smoothed'] = pd.concat(smoothed).sort_index()
    return df

from statsmodels.tsa.seasonal import seasonal_decompose

def decompose_trailing(series, period=24, window=56*24):
    """
    Returns trend, seasonal, resid where each value for t is computed from a trailing window: [t-window, t-1]
    """
    N = len(series)
    trend = np.full(N, np.nan)
    seasonal = np.full(N, np.nan)
    resid = np.full(N, np.nan)

    series_filled = series.ffill().bfill()

    for i in range(window, N):
        win_slice = series_filled[i-window:i]  # strictly past data
        # Decompose only if enough non-NaN
        if win_slice.isnull().mean() < 0.2 and win_slice.count() > period*2:
            dec = seasonal_decompose(win_slice, model='additive', period=period, extrapolate_trend='freq')
            # Assign latest value (for time i)
            trend[i] = dec.trend.iloc[-1]
            seasonal[i] = dec.seasonal.iloc[-1]
            resid[i] = dec.resid.iloc[-1]
        # if insufficient history, keep output as NaN
    return pd.Series(trend, index=series.index), pd.Series(seasonal, index=series.index), pd.Series(resid, index=series.index)


for app, g in df.groupby('AppName'):
    t, s, r = decompose_trailing(g['transactions'], period=24, window=56*24)
    df.loc[g.index, 'trend'] = t
    df.loc[g.index, 'seasonal'] = s
    df.loc[g.index, 'residual'] = r

def group_trailing_mean(df, key_cols, value_col, trailing_hours):
    means = []
    for app, g in df.groupby('AppName'):
        # Only historical portion if needed (prior to last_known_date for inference)
        g = g.sort_values('datetime')
        rolling_means = g.set_index('datetime').groupby(key_cols)[value_col].rolling(window=trailing_hours, min_periods=1).mean().reset_index()
        # Merge back on key_cols + datetime for safe alignment
        means.append(rolling_means)
    # Concatenate all groups and merge with df as needed
    return pd.concat(means, ignore_index=True)



def safe_daily_aggregate(df, last_known_date):
    """
    Compute leakage-safe daily transaction aggregates.

    Parameters:
     - df: DataFrame with columns ['datetime', 'AppName', 'transactions'] and inference placeholders with zero transactions
     - last_known_date: Timestamp up to which data is real (e.g., 2025-06-18 23:00:00)

    Returns:
     - df with a new column 'transactions_daily' with NaN for inference day, real daily sums for history
    """

    # Extract date for grouping
    df['date'] = df['datetime'].dt.date

    # Split into history and inference placeholder parts
    hist_df = df[df['datetime'] <= last_known_date]
    infer_dates = df[df['datetime'] > last_known_date]['date'].unique()

    # Compute daily sum only on history
    daily_hist = (
        hist_df.groupby(['AppName', 'date'], as_index=False)['transactions']
        .sum()
        .rename(columns={'transactions': 'transactions_daily'})
    )

    # Build null daily rows for inference dates
    apps = hist_df['AppName'].unique()
    null_daily = pd.DataFrame({
        'AppName': np.repeat(apps, len(infer_dates)),
        'date': np.tile(infer_dates, len(apps)),
        'transactions_daily': np.nan
    })

    # Combine history + null inference daily sums
    daily_all = pd.concat([daily_hist, null_daily], ignore_index=True)

    # Merge back to original df
    df = df.merge(daily_all, on=['AppName', 'date'], how='left')

    return df


# 
def safe_daily_features(df, last_known_date):
    """
    Compute daily transaction aggregates + daily lag/rolling features leakage-safe.

    Parameters:
     - df: DataFrame with ['datetime','AppName','transactions'].
     - last_known_date: Timestamp or None. Cut-off for historical data to avoid leakage in inference.

    Returns:
     - df with daily features merged:
       transactions_daily, daily lags, rolling means/stds, EMAs
    """

    df = df.copy()
    df['date'] = df['datetime'].dt.date

    # Split history & inference placeholder days
    if last_known_date is not None:
        hist_df = df[df['datetime'] <= last_known_date]
        infer_dates = df[df['datetime'] > last_known_date]['date'].unique()
    else:
        hist_df = df
        infer_dates = []

    # Compute daily sum on history only
    daily = (
        hist_df.groupby(['AppName','date'], as_index=False)['transactions']
        .sum()
        .rename(columns={'transactions':'transactions_daily'})
    )

    # Create null daily sums for inference days
    if len(infer_dates) > 0:
        apps = hist_df['AppName'].unique()
        null_daily = pd.DataFrame({
            'AppName': np.repeat(apps, len(infer_dates)),
            'date': np.tile(infer_dates, len(apps)),
            'transactions_daily': np.nan
        })
        daily = pd.concat([daily, null_daily], ignore_index=True)

    # Compute daily lags on historical + null entries -- use transform to align by index
    for lag in [1,7,14]:
        daily[f'lag_{lag}d'] = (
            daily.groupby('AppName')['transactions_daily']
            .transform(lambda x: x.shift(lag))
        )

    # Daily rolling mean and std
    for window in [3,7,14]:
        daily[f'roll_mean_{window}d'] = (
            daily.groupby('AppName')['transactions_daily']
            .transform(lambda x: x.shift(1).rolling(window=window,min_periods=1).mean())
        )
        daily[f'roll_std_{window}d'] = (
            daily.groupby('AppName')['transactions_daily']
            .transform(lambda x: x.shift(1).rolling(window=window,min_periods=1).std())
        )

    # Daily EMA
    for window in [3,7,14]:
        daily[f'ema_{window}d'] = (
            daily.groupby('AppName')['transactions_daily']
            .transform(lambda x: x.shift(1).ewm(span=window, adjust=False).mean())
        )

    # Merge daily features back to original df
    df = df.merge(daily, on=['AppName','date'], how='left')

    return df

# Usage
# For training/validation (no last_known_date needed)
df_train = safe_daily_features(df_train, last_known_date=None)

# For inference (pass last known actual timestamp)
last_known_date = pd.Timestamp("2025-06-18 23:00:00")
df_infer = safe_daily_features(df_infer, last_known_date=last_known_date)


# Changes to weekly logic
import pandas as pd
import numpy as np

def safe_weekly_features(df, last_known_date=None):
    """
    Compute leakage-safe weekly aggregates and weekly lag/rolling/EMA features.

    Inputs:
      df: DataFrame with at least ['datetime','AppName','transactions'].
          Can include inference placeholder rows (e.g., future dates with 0 transactions).
      last_known_date: Timestamp or None.
        - None for training/validation: compute weekly stats on all provided rows.
        - Timestamp for inference: compute weekly stats only from rows with datetime <= last_known_date.
    
    Returns:
      df with new weekly features merged:
        - transactions_weekly
        - lag_1w, lag_2w, lag_4w (weekly lags)
        - roll_mean_4w, roll_mean_8w (weekly rolling means)
        - ema_4w, ema_8w (weekly EMAs)
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    # ISO week/year for stable weekly grouping
    iso = df['datetime'].dt.isocalendar()
    df['iso_year'] = iso.year.astype(int)
    df['iso_week'] = iso.week.astype(int)

    # Split history vs inference by timestamp
    if last_known_date is not None:
        hist_mask = df['datetime'] <= last_known_date
        hist_df = df.loc[hist_mask]
        # Identify any (year, week) combinations that exist after last_known_date (inference weeks)
        infer_weeks = (
            df.loc[~hist_mask, ['AppName','iso_year','iso_week']]
              .drop_duplicates()
              .to_numpy()
        )
    else:
        hist_df = df
        infer_weeks = np.empty((0,3), dtype=object)

    # Weekly aggregate on history only
    weekly = (
        hist_df.groupby(['AppName','iso_year','iso_week'], as_index=False)['transactions']
               .sum()
               .rename(columns={'transactions':'transactions_weekly'})
    )

    # Create NaN placeholders for inference weeks so they merge but don’t use placeholders
    if infer_weeks.size > 0:
        apps = hist_df['AppName'].unique()
        wk_rows = []
        # If not all apps are in infer weeks entries, we still want NaN rows for all apps present in history
        for app in apps:
            mask_app = infer_weeks[:,0] == app
            # If no explicit rows for this app were found after cutoff, skip
            # (merge will simply not produce weekly rows for those weeks)
            for _, y, w in infer_weeks[mask_app]:
                wk_rows.append((app, int(y), int(w), np.nan))
        if wk_rows:
            null_weekly = pd.DataFrame(wk_rows, columns=['AppName','iso_year','iso_week','transactions_weekly'])
            weekly = pd.concat([weekly, null_weekly], ignore_index=True)

    # Weekly lags
    for lag in [1, 2, 4]:
        weekly[f'lag_{lag}w'] = (
            weekly.groupby('AppName')['transactions_weekly']
                  .transform(lambda x: x.shift(lag))
        )

    # Weekly rolling means (trailing, past-only with shift(1))
    for win in [4, 8]:
        weekly[f'roll_mean_{win}w'] = (
            weekly.groupby('AppName')['transactions_weekly']
                  .transform(lambda x: x.shift(1).rolling(window=win, min_periods=1).mean())
        )
        weekly[f'ema_{win}w'] = (
            weekly.groupby('AppName')['transactions_weekly']
                  .transform(lambda x: x.shift(1).ewm(span=win, adjust=False).mean())
        )

    # Merge weekly features back to hourly frame
    df = df.merge(weekly, on=['AppName','iso_year','iso_week'], how='left')

    return df

# historical_fill
if not isinstance(series, pd.Series):
    series = pd.Series(series)
# Align mask to the series index
if isinstance(hist_mask, pd.Series):
    hist_mask_aligned = hist_mask.reindex(series.index, fill_value=False)
else:
    # If mask is a scalar/bool or None
    hist_mask_aligned = pd.Series(bool(hist_mask), index=series.index)

hist_values = series[hist_mask_aligned]
median_val = hist_values.median()

filled = series.copy()
# Fill only historical rows; leave future rows untouched (NaN ok for model)
filled.loc[hist_mask_aligned] = (
    hist_values.ffill().bfill().fillna(median_val)
)
return filled


# CHANGED: Build a hist_mask always aligned to df index
if last_known_date is not None:
    hist_mask = (df[date_col] <= last_known_date)
else:
    hist_mask = pd.Series(True, index=df.index)

# ===========================================
# CHANGED: Historical filling with reindexing
# ===========================================
# Optional: ensure df index is clean before fill
df = df.reset_index(drop=True)
# Re-assert hist_mask aligned to df after merges
if isinstance(hist_mask, pd.Series):
    hist_mask = hist_mask.reindex(df.index, fill_value=False)
else:
    hist_mask = pd.Series(bool(hist_mask), index=df.index)

# Sanity check alignment
assert hist_mask.index.equals(df.index), "hist_mask not aligned post-merge"

# Fill only lag_/roll_/ema_ (skip _peak if you want)
for col in df.columns:
    if col.startswith(('lag_', 'roll_', 'ema_')) and not col.endswith('_peak'):
        s = df[col]
        # Reindex series to df to guarantee same index
        if not s.index.equals(df.index):
            s = s.reindex(df.index)
        df[col] = historical_fill(s, hist_mask)   



# Fix for duplicate rows
def safe_daily_features(df, last_known_date=None,
lag_days=(1,7,14),
roll_days=(3,7,14),
ema_days=(3,7,14)):
    """
    Leakage-safe daily aggregates and daily lag/rolling/std/EMA features with duplicate protection.
    - df columns required: ['datetime','AppName','transactions']
    - last_known_date:
    None -> training/validation; compute on all provided dates
    Timestamp -> inference; compute daily stats from rows with datetime <= last_known_date
    and set inference-day(s) daily aggregates to NaN
    Returns:
    df with columns merged:
    transactions_daily, lag_{d}d, roll_mean_{d}d, roll_std_{d}d, ema_{d}d
    Also guarantees no duplicated (AppName, datetime) records.
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['AppName'] = df['AppName'].astype(str)
    df['date'] = df['datetime'].dt.date

    text
    # Split history vs future
    if last_known_date is not None:
        hist_mask = df['datetime'] <= last_known_date
        hist_df = df.loc[hist_mask, ['AppName','date','transactions']]
        future_dates = df.loc[~hist_mask, 'date'].unique()
    else:
        hist_df = df[['AppName','date','transactions']]
        future_dates = []

    # Daily sum on history only
    daily_hist = (
        hist_df.groupby(['AppName','date'], as_index=False)['transactions']
            .sum()
            .rename(columns={'transactions':'transactions_daily'})
    )
    daily_hist['AppName'] = daily_hist['AppName'].astype(str)
    daily_hist['date'] = pd.to_datetime(daily_hist['date']).dt.date

    # NaN placeholders for future dates
    if len(future_dates) > 0:
        apps = daily_hist['AppName'].unique()
        if len(apps) == 0:
            apps = df['AppName'].unique()
        daily_null = pd.DataFrame({
            'AppName': np.repeat(apps, len(future_dates)),
            'date': np.tile(future_dates, len(apps)),
            'transactions_daily': np.nan
        })
        daily_null['AppName'] = daily_null['AppName'].astype(str)
        daily_null['date'] = pd.to_datetime(daily_null['date']).dt.date
        daily_all = pd.concat([daily_hist, daily_null], ignore_index=True)
    else:
        daily_all = daily_hist

    # Normalize and deduplicate daily index
    daily_all['AppName'] = daily_all['AppName'].astype(str)
    daily_all['date'] = pd.to_datetime(daily_all['date']).dt.date
    daily_all = (
        daily_all.sort_values(['AppName','date'])
                .drop_duplicates(subset=['AppName','date'], keep='first')
                .reset_index(drop=True)
    )

    # Build daily lag/rolling/std/EMA on the daily frame per AppName
    def _build_daily_stats(g):
        g = g.sort_values('date').copy()
        # Lags
        for d in lag_days:
            g[f'lag_{d}d'] = g['transactions_daily'].shift(d)
        # Rolling (past-only with shift(1))
        for d in roll_days:
            g[f'roll_mean_{d}d'] = g['transactions_daily'].shift(1).rolling(window=d, min_periods=1).mean()
            g[f'roll_std_{d}d']  = g['transactions_daily'].shift(1).rolling(window=d, min_periods=1).std()
        # EMA
        for d in ema_days:
            g[f'ema_{d}d'] = g['transactions_daily'].shift(1).ewm(span=d, adjust=False).mean()
        return g

    daily_all = (
        daily_all.groupby('AppName', group_keys=False)
                .apply(_build_daily_stats)
                .reset_index(drop=True)
    )

    # Merge back to hourly (many hourly rows to one daily row)
    df = df.merge(
        daily_all,
        on=['AppName','date'],
        how='left',
        validate='many_to_one'
    )

    # Guarantee uniqueness of (AppName, datetime) rows
    df = df.drop_duplicates(subset=['AppName','datetime']).reset_index(drop=True)
return df

# Safe weekly features (aggregate + lags/rolling/EMA) with de-duplication
def safe_weekly_features(df, last_known_date=None,
    lag_weeks=(1,2,4),
    roll_weeks=(4,8),
    ema_weeks=(4,8)):
    """
    Leakage-safe weekly aggregates and weekly lag/rolling/EMA features with duplicate protection.
    - df columns required: ['datetime','AppName','transactions']
    - last_known_date:
    None -> training/validation; compute on all provided weeks
    Timestamp -> inference; compute weekly stats from rows with datetime <= last_known_date
    and set inference-week aggregates to NaN
    Returns:
    df with columns merged:
    transactions_weekly, lag_{w}w, roll_mean_{w}w, ema_{w}w
    Also guarantees no duplicated (AppName, datetime) records.
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['AppName'] = df['AppName'].astype(str)

    text
    iso = df['datetime'].dt.isocalendar()
    df['iso_year'] = iso.year.astype(int)
    df['iso_week'] = iso.week.astype(int)

    # Split history vs future
    if last_known_date is not None:
        hist_mask = df['datetime'] <= last_known_date
        hist_week = df.loc[hist_mask, ['AppName','iso_year','iso_week','transactions']]
        future_keys = (
            df.loc[~hist_mask, ['AppName','iso_year','iso_week']]
            .drop_duplicates()
        )
    else:
        hist_week = df[['AppName','iso_year','iso_week','transactions']]
        future_keys = pd.DataFrame(columns=['AppName','iso_year','iso_week'])

    # Weekly sum on history only
    weekly_hist = (
        hist_week.groupby(['AppName','iso_year','iso_week'], as_index=False)['transactions']
                .sum()
                .rename(columns={'transactions':'transactions_weekly'})
    )
    weekly_hist['AppName'] = weekly_hist['AppName'].astype(str)
    weekly_hist['iso_year'] = weekly_hist['iso_year'].astype(int)
    weekly_hist['iso_week'] = weekly_hist['iso_week'].astype(int)

    # NaN placeholders for future weeks
    if not future_keys.empty:
        wk_null = future_keys.copy()
        wk_null['AppName'] = wk_null['AppName'].astype(str)
        wk_null['iso_year'] = wk_null['iso_year'].astype(int)
        wk_null['iso_week'] = wk_null['iso_week'].astype(int)
        wk_null['transactions_weekly'] = np.nan
        weekly_all = pd.concat([weekly_hist, wk_null], ignore_index=True)
    else:
        weekly_all = weekly_hist

    # Normalize and deduplicate weekly index
    weekly_all['AppName'] = weekly_all['AppName'].astype(str)
    weekly_all['iso_year'] = weekly_all['iso_year'].astype(int)
    weekly_all['iso_week'] = weekly_all['iso_week'].astype(int)
    weekly_all = (
        weekly_all.sort_values(['AppName','iso_year','iso_week'])
                .drop_duplicates(subset=['AppName','iso_year','iso_week'], keep='first')
                .reset_index(drop=True)
    )

    # Build weekly lag/rolling/EMA on the weekly frame per AppName
    def _build_weekly_stats(g):
        g = g.sort_values(['iso_year','iso_week']).copy()
        # Lags
        for w in lag_weeks:
            g[f'lag_{w}w'] = g['transactions_weekly'].shift(w)
        # Rolling (past-only)
        for w in roll_weeks:
            g[f'roll_mean_{w}w'] = g['transactions_weekly'].shift(1).rolling(window=w, min_periods=1).mean()
        # EMA
        for w in ema_weeks:
            g[f'ema_{w}w'] = g['transactions_weekly'].shift(1).ewm(span=w, adjust=False).mean()
        return g

    weekly_all = (
        weekly_all.groupby('AppName', group_keys=False)
                .apply(_build_weekly_stats)
                .reset_index(drop=True)
    )

    # Merge back to hourly (many hourly rows to one weekly row)
    df = df.merge(
        weekly_all,
        on=['AppName','iso_year','iso_week'],
        how='left',
        validate='many_to_one'
    )

    # Guarantee uniqueness of (AppName, datetime) rows
    df = df.drop_duplicates(subset=['AppName','datetime']).reset_index(drop=True)
    return df


# safe daily features at prediction
def safe_daily_features_with_trailing_at_prediction(df, last_known_date=None,
lag_days=(1,7,14),
roll_days=(3,7,14),
ema_days=(3,7,14)):
"""
Build daily aggregate features safely and ensure that prediction-day rows
receive trailing daily features (lags/rolls/emas) based strictly on history.
- transactions_daily for the prediction day remains NaN.
- lag_d / roll_d / ema_*d for the prediction day are populated from history.
"""
df = df.copy()
df['datetime'] = pd.to_datetime(df['datetime'])
df['AppName'] = df['AppName'].astype(str)
df['date'] = df['datetime'].dt.date

text
# Partition history vs future
if last_known_date is not None:
    last_known_date = pd.to_datetime(last_known_date)
    hist_mask = df['datetime'] <= last_known_date
    df_hist = df.loc[hist_mask, ['AppName','date','transactions']].copy()
    pred_dates = df.loc[~hist_mask, 'date'].unique()  # may include 2025-06-19
else:
    df_hist = df[['AppName','date','transactions']].copy()
    pred_dates = []

# Ensure date dtype is date (not datetime64)
df_hist['date'] = pd.to_datetime(df_hist['date']).dt.date

# 1) Daily totals from history only
daily = (df_hist.groupby(['AppName','date'], as_index=False)['transactions']
         .sum()
         .rename(columns={'transactions':'transactions_daily'}))
daily['AppName'] = daily['AppName'].astype(str)
daily['date'] = pd.to_datetime(daily['date']).dt.date

# 2) Build daily derived features (lags/rolling/ema) on the daily history frame
def _build_stats(g):
    g = g.sort_values('date').copy()
    for d in lag_days:
        g[f'lag_{d}d'] = g['transactions_daily'].shift(d)
    for d in roll_days:
        g[f'roll_mean_{d}d'] = g['transactions_daily'].shift(1).rolling(d, min_periods=1).mean()
        g[f'roll_std_{d}d']  = g['transactions_daily'].shift(1).rolling(d, min_periods=1).std()
    for d in ema_days:
        g[f'ema_{d}d'] = g['transactions_daily'].shift(1).ewm(span=d, adjust=False).mean()
    return g

daily = (daily.groupby('AppName', group_keys=False)
              .apply(_build_stats)
              .reset_index(drop=True))

# 3) Trailing features at prediction-day rows
# For inference mode: attach the last available daily row per App to prediction-day rows
if last_known_date is not None and len(pred_dates) > 0:
    # last fully observed day
    last_day = last_known_date.date()

    last_rows = (daily[daily['date'] == last_day]
                 .copy()
                 .drop(columns=['transactions_daily']))  # keep only derived features
    # Suffix to distinguish these from true daily columns if needed
    # Here we carry them into the prediction day under the same names for convenience
    # because they are valid trailing values at t=prediction_day
    carry = last_rows[['AppName'] + [c for c in last_rows.columns if c not in ('AppName','date')]]

    # Merge daily history features to historical hourly rows (standard many_to_one)
    df = df.merge(
        daily,
        on=['AppName','date'],
        how='left',
        validate='many_to_one'
    )

    # Now, attach trailing features for prediction-day rows by AppName (no date key)
    pred_mask = df['date'].isin(pred_dates)
    if carry.shape > 0:
        df = df.merge(
            carry,
            on='AppName',
            how='left',
            suffixes=('','_carry')
        )
        # For prediction-day rows, fill NaN derived daily features from carry
        derived_cols = [c for c in carry.columns if c != 'AppName']
        for c in derived_cols:
            base_col = c  # same name present from daily merge
            if base_col in df.columns:
                df.loc[pred_mask, base_col] = df.loc[pred_mask, base_col].fillna(df.loc[pred_mask, c])
        # Optionally drop carry columns afterwards
        df.drop(columns=[c for c in df.columns if c.endswith('_carry')], inplace=True)
    # Leave transactions_daily as NaN for prediction days (no fill)
else:
    # Training/validation: straight merge of daily features
    df = df.merge(
        daily,
        on=['AppName','date'],
        how='left',
        validate='many_to_one'
    )

# Enforce uniqueness of (AppName, datetime)
df = df.drop_duplicates(subset=['AppName','datetime']).reset_index(drop=True)
return df



# Updated safe daily
def safe_daily_features_with_trailing_at_prediction(df, last_known_date=None,
lag_days=(1,7,14),
roll_days=(3,7,14),
ema_days=(3,7,14)):
df = df.copy()
df['datetime'] = pd.to_datetime(df['datetime']) # ensure consistent dtype
df['date'] = df['datetime'].dt.date # date key (not datetime)
# Do NOT force AppName dtype changes in training parity runs; keep as-is unless inconsistent

text
if last_known_date is not None:
    last_known_date = pd.to_datetime(last_known_date)  # robust cutoff[1]
    hist_mask = df['datetime'] <= last_known_date
    df_hist = df.loc[hist_mask, ['AppName','date','transactions']].copy()
    pred_dates = df.loc[~hist_mask, 'date'].unique()
else:
    df_hist = df[['AppName','date','transactions']].copy()
    pred_dates = []

# Build daily history
daily = (df_hist.groupby(['AppName','date'], as_index=False)['transactions']
         .sum().rename(columns={'transactions':'transactions_daily'}))  # history-only[1]

# Derived features on the daily history frame
def _build_stats(g):
    g = g.sort_values('date').copy()  # stable order[1]
    for d in lag_days:
        g[f'lag_{d}d'] = g['transactions_daily'].shift(d)
    for d in roll_days:
        g[f'roll_mean_{d}d'] = g['transactions_daily'].shift(1).rolling(d, min_periods=1).mean()
        g[f'roll_std_{d}d']  = g['transactions_daily'].shift(1).rolling(d, min_periods=1).std()
    for d in ema_days:
        g[f'ema_{d}d'] = g['transactions_daily'].shift(1).ewm(span=d, adjust=False).mean()
    return g

daily = daily.groupby('AppName', group_keys=False).apply(_build_stats).reset_index(drop=True)  # per-app stats[1]

# Merge history daily features to all rows (history rows match; prediction rows will be NaN)[1]
df = df.merge(daily, on=['AppName','date'], how='left', validate='many_to_one')  # right frame unique per key[1]

# Carry trailing daily features into prediction-day rows without filling transactions_daily[1]
if last_known_date is not None and len(pred_dates) > 0:
    last_day = last_known_date.date()
    last_rows = daily[daily['date'] == last_day].copy()
    if last_rows.shape == 0:
        # If last_day is missing (e.g., wrong cutoff), fall back to the latest available historical day per app[1]
        idx = daily.groupby('AppName')['date'].idxmax()
        last_rows = daily.loc[idx].copy()
    # Only derived columns (not transactions_daily)
    carry_cols = [c for c in last_rows.columns if c not in ('AppName','date','transactions_daily')]
    carry = last_rows[['AppName'] + carry_cols].copy()
    if carry.shape > 0:  # FIX: compare row count[3]
        pred_mask = df['date'].isin(pred_dates)
        # Fill NaNs for derived daily features at prediction rows from carry[1]
        df = df.merge(carry, on='AppName', how='left', suffixes=('','_carry'))  # join by AppName[1]
        for c in carry_cols:
            df.loc[pred_mask, c] = df.loc[pred_mask, c].fillna(df.loc[pred_mask, f'{c}_carry'])  # safe fill[1]
        df.drop(columns=[f'{c}_carry' for c in carry_cols], inplace=True)

# Ensure uniqueness on (AppName, datetime)[1]
df = df.drop_duplicates(subset=['AppName','datetime']).reset_index(drop=True)  # guard[1]
return df
