import pandas as pd
import numpy as np

def compare_inference_impact(df, feature_creation_fn, target_date):
    """
    Compare features generated with real past data vs placeholder-0 inference style.
    
    Parameters:
    - df: original dataframe with datetime, AppName, transactions
    - feature_creation_fn: your function that takes df and returns df with engineered features
    - target_date: datetime.date to test (should be inside your test period with real data)
    
    Returns:
    - diff_df: DataFrame of features with large deviation between real vs placeholder runs
    """
    
    # Ensure datetime is datetime type
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # === First run: real data ===
    real_df = feature_creation_fn(df.copy())
    real_features = real_df[real_df['datetime'].dt.date == target_date]
    
    # === Second run: placeholder inference style ===
    placeholder_df = df.copy()
    placeholder_df.loc[placeholder_df['datetime'].dt.date == target_date, 'transactions'] = 0
    placeholder_df = feature_creation_fn(placeholder_df)
    placeholder_features = placeholder_df[placeholder_df['datetime'].dt.date == target_date]
    
    # Align feature columns (drop non-feature columns like datetime/AppName/target)
    non_feature_cols = ['datetime', 'AppName', 'transactions']
    feat_cols = [c for c in real_features.columns if c not in non_feature_cols]
    
    # Calculate relative and absolute differences
    diffs = []
    for col in feat_cols:
        abs_diff = np.abs(real_features[col].values - placeholder_features[col].values)
        rel_diff = abs_diff / (np.abs(real_features[col].values) + 1e-9)
        
        diffs.append({
            'feature': col,
            'mean_abs_diff': abs_diff.mean(),
            'max_abs_diff': abs_diff.max(),
            'mean_rel_diff': rel_diff.mean(),
            'max_rel_diff': rel_diff.max()
        })
    
    diff_df = pd.DataFrame(diffs).sort_values('mean_abs_diff', ascending=False)
    
    # Flag suspicious features
    diff_df['flag_suspicious'] = (diff_df['mean_rel_diff'] > 0.05) & (diff_df['mean_abs_diff'] > 1e-6)
    
    return diff_df

# Example usage:
# diff_report = compare_inference_impact(df, feature_creation_pipeline, target_date=pd.to_datetime("2025-06-10").date())
# print(diff_report[diff_report['flag_suspicious']])

# Prior one
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.preprocessing import LabelEncoder

def feature_drift_and_health(train_df, inference_df, threshold_ks=0.1):
    report = []

    for col in train_df.columns:
        if col not in inference_df.columns:
            continue

        train_col = train_df[col].dropna()
        infer_col = inference_df[col].dropna()

        # Skip if empty
        if train_col.empty or infer_col.empty:
            continue

        if np.issubdtype(train_col.dtype, np.number):
            # 1. Drift test (KS test)
            ks_stat, ks_p = ks_2samp(train_col, infer_col)
            drift_flag = ks_stat > threshold_ks

            # 2. Broken detection
            all_constant_flag = infer_col.nunique() <= 1
            extreme_range_flag = (
                infer_col.max() > train_col.max() * 5 or
                infer_col.min() < train_col.min() * 5
            )

        else:
            # Encode categoricals
            le = LabelEncoder()
            train_enc = le.fit_transform(train_col.astype(str))
            infer_enc = le.transform(infer_col.astype(str))
            ks_stat, ks_p = ks_2samp(train_enc, infer_enc)
            drift_flag = ks_stat > threshold_ks

            all_constant_flag = infer_col.nunique() <= 1
            extreme_range_flag = False  # not numeric

        # 3. Risk score
        risk_score = (
            (drift_flag * 2) + 
            (all_constant_flag * 3) + 
            (extreme_range_flag * 1)
        )

        report.append({
            "feature": col,
            "ks_stat": ks_stat,
            "ks_pvalue": ks_p,
            "drift_flag": drift_flag,
            "all_constant_flag": all_constant_flag,
            "extreme_range_flag": extreme_range_flag,
            "risk_score": risk_score
        })

    report_df = pd.DataFrame(report)
    return report_df.sort_values("risk_score", ascending=False)

# Example usage
# train_features: your feature matrix for training
# infer_features: your feature matrix for inference date
report_df = feature_drift_and_health(train_features, infer_features)
print(report_df)

# If we combine this with your delta RMSE from single-pass feature elimination, you can flag:
#This final_flag gives you features that are both high-risk in drift/brokenness AND harmful to prediction, which is your top priority to drop or fix.
merged = report_df.merge(delta_rmse_df, on="feature", how="left")
merged["rmse_drop_flag"] = merged["delta_rmse"] < 0
merged["final_flag"] = merged["risk_score"] >= 2 & merged["rmse_drop_flag"]

# Prior
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestRegressor

def population_stability_index(expected, actual, bins=10):
    """Calculate PSI between two numeric arrays."""
    expected_perc, _ = np.histogram(expected, bins=bins)
    actual_perc, _ = np.histogram(actual, bins=bins)

    expected_perc = expected_perc / len(expected)
    actual_perc = actual_perc / len(actual)

    psi = np.sum((actual_perc - expected_perc) * np.log((actual_perc + 1e-6) / (expected_perc + 1e-6)))
    return psi

def feature_drift_report(train_df, test_df, inference_df):
    """
    Compare feature drift & data quality issues between train, test, and inference datasets.
    All must have the same columns.
    """
    report = []
    for col in train_df.columns:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            # KS statistic between train & inference
            ks_stat, ks_p = ks_2samp(train_df[col].dropna(), inference_df[col].dropna())
            # PSI between train & inference
            psi = population_stability_index(train_df[col].dropna(), inference_df[col].dropna())
            # Data quality
            train_na = train_df[col].isna().mean()
            inf_na = inference_df[col].isna().mean()
            train_unique = train_df[col].nunique()
            inf_unique = inference_df[col].nunique()

            report.append({
                "feature": col,
                "ks_stat": ks_stat,
                "ks_p_value": ks_p,
                "psi": psi,
                "train_na_pct": train_na,
                "inference_na_pct": inf_na,
                "train_unique": train_unique,
                "inference_unique": inf_unique,
                "constant_in_inference": inf_unique == 1
            })
    return pd.DataFrame(report).sort_values("psi", ascending=False)

def feature_importance_stability(train_df, test_df, y_train, y_test):
    """Train RF on train and on train+test, compare feature rankings."""
    rf1 = RandomForestRegressor(n_estimators=200, random_state=42)
    rf1.fit(train_df, y_train)
    imp1 = pd.Series(rf1.feature_importances_, index=train_df.columns)

    combined_X = pd.concat([train_df, test_df], axis=0)
    combined_y = pd.concat([y_train, y_test], axis=0)

    rf2 = RandomForestRegressor(n_estimators=200, random_state=42)
    rf2.fit(combined_X, combined_y)
    imp2 = pd.Series(rf2.feature_importances_, index=train_df.columns)

    stability = (imp2 - imp1).abs()
    return pd.DataFrame({"importance_train": imp1, "importance_train_test": imp2, "abs_change": stability}).sort_values("abs_change", ascending=False)

# Example usage
# drift_df = feature_drift_report(X_train, X_test, X_inference)
# stability_df = feature_importance_stability(X_train, X_test, y_train, y_test)

# After feature creation
X_train_final = train_features.drop(columns=['transactions'])
X_test_final = test_features.drop(columns=['transactions'])
X_inf_final = inference_features.drop(columns=['transactions'])

# Run drift check
drift_df = feature_drift_report(X_train_final, X_test_final, X_inf_final)
print(drift_df.head(20))  # Top 20 biggest drifts
# KS test: p < 0.05 → distribution differs significantly.
# PSI: > 0.2 → moderate drift, > 0.5 → severe drift.
# constant_in_inference = True → feature might be broken.

stability_df = feature_importance_stability(X_train_final, X_test_final, y_train, y_test)
print(stability_df.head(20))
# Large abs_change = feature’s predictive power is unstable.


