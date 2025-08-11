import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def time_series_feature_elimination(X_train, y_train, X_valid, y_valid, categorical_features=None):
    """
    Sequential backward elimination for time series data.
    Drops one feature at a time and measures RMSE change.
    """

    # Baseline model with all features
    model_all = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        random_state=42
    )
    model_all.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='rmse',
        categorical_feature=categorical_features,
        verbose=False
    )

    y_pred_all = model_all.predict(X_valid)
    baseline_rmse = mean_squared_error(y_valid, y_pred_all, squared=False)
    print(f"\nBaseline RMSE with all features: {baseline_rmse:.4f}")

    results = []

    for feature in X_train.columns:
        print(f"Testing without feature: {feature}")

        X_train_reduced = X_train.drop(columns=[feature])
        X_valid_reduced = X_valid.drop(columns=[feature])

        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            random_state=42
        )
        model.fit(
            X_train_reduced, y_train,
            eval_set=[(X_valid_reduced, y_valid)],
            eval_metric='rmse',
            categorical_feature=[f for f in categorical_features if f != feature] if categorical_features else None,
            verbose=False
        )

        y_pred = model.predict(X_valid_reduced)
        rmse = mean_squared_error(y_valid, y_pred, squared=False)
        delta = rmse - baseline_rmse

        results.append({
            "feature": feature,
            "rmse": rmse,
            "delta_rmse": delta
        })

    results_df = pd.DataFrame(results).sort_values("delta_rmse")
    return results_df

# Example usage:
# Assume df has MultiIndex (datetime, AppName) and we have already split into train/test
# Drop index safely
X = df.drop(columns=["transactions"])  # target column
y = df["transactions"]

# Split preserving time order
train_size = int(len(df) * 0.8)
X_train, X_valid = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_valid = y.iloc[:train_size], y.iloc[train_size:]

categorical_features = ["hour_of_day", "day_of_week"]  # your known cat columns

results_df = time_series_feature_elimination(X_train, y_train, X_valid, y_valid, categorical_features)
print(results_df)

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate_model(X_train, y_train, X_valid, y_valid, categorical_features):
    """Train LightGBM and return RMSE on validation."""
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='rmse',
        categorical_feature=[f for f in categorical_features if f in X_train.columns],
        verbose=False
    )
    y_pred = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, y_pred, squared=False)
    return rmse


def single_pass_feature_elimination(X_train, y_train, X_valid, y_valid, categorical_features):
    """One pass — test removing each feature individually."""
    baseline_rmse = evaluate_model(X_train, y_train, X_valid, y_valid, categorical_features)
    print(f"\nBaseline RMSE with all features: {baseline_rmse:.4f}")

    results = []
    for feature in X_train.columns:
        X_train_reduced = X_train.drop(columns=[feature])
        X_valid_reduced = X_valid.drop(columns=[feature])

        rmse = evaluate_model(X_train_reduced, y_train, X_valid_reduced, y_valid, categorical_features)
        delta = rmse - baseline_rmse

        results.append({
            "feature": feature,
            "rmse": rmse,
            "delta_rmse": delta
        })

    results_df = pd.DataFrame(results).sort_values("delta_rmse")
    return results_df


def sequential_backward_elimination(X_train, y_train, X_valid, y_valid, categorical_features):
    """Full loop — remove worst offender iteratively until RMSE stops improving."""
    current_features = list(X_train.columns)
    baseline_rmse = evaluate_model(X_train, y_train, X_valid, y_valid, categorical_features)
    print(f"\nInitial RMSE: {baseline_rmse:.4f} with {len(current_features)} features")

    improved = True
    history = []

    while improved and len(current_features) > 1:
        improved = False
        best_rmse = baseline_rmse
        worst_feature = None

        for feature in current_features:
            X_train_reduced = X_train[current_features].drop(columns=[feature])
            X_valid_reduced = X_valid[current_features].drop(columns=[feature])

            rmse = evaluate_model(X_train_reduced, y_train, X_valid_reduced, y_valid, categorical_features)

            if rmse < best_rmse - 1e-4:  # tiny tolerance to avoid random fluctuation
                best_rmse = rmse
                worst_feature = feature

        if worst_feature:
            print(f"Removing '{worst_feature}' improved RMSE from {baseline_rmse:.4f} → {best_rmse:.4f}")
            current_features.remove(worst_feature)
            baseline_rmse = best_rmse
            improved = True
            history.append((worst_feature, best_rmse))
        else:
            print("No further improvement found.")
            break

    return current_features, history


# ==== Example Usage ====
# Drop index columns before passing to this script
# df.index is MultiIndex (datetime, AppName)
# target = df["transactions"]
# features = df.drop(columns=["transactions"])

# Split preserving order
# train_size = int(len(df) * 0.8)
# X_train, X_valid = features.iloc[:train_size], features.iloc[train_size:]
# y_train, y_valid = target.iloc[:train_size], target.iloc[train_size:]

# Known categorical columns
# categorical_features = ["hour_of_day", "day_of_week"]

# Step 1 — single pass check
# single_pass_results = single_pass_feature_elimination(X_train, y_train, X_valid, y_valid, categorical_features)
# print(single_pass_results)

# Step 2 — full backward elimination
# final_features, elimination_history = sequential_backward_elimination(X_train, y_train, X_valid, y_valid, categorical_features)
# print("\nFinal selected features:", final_features)
# print("Elimination history:", elimination_history)



from itertools import combinations

def multi_step_backward_elimination(
    X_train, y_train, X_valid, y_valid,
    categorical_features,
    max_step_size=2
):
    """
    Multi-step lookahead backward elimination.
    Can remove 1..max_step_size features per step.
    """
    current_features = list(X_train.columns)
    baseline_rmse = evaluate_model(X_train, y_train, X_valid, y_valid, categorical_features)
    print(f"\nInitial RMSE: {baseline_rmse:.4f} with {len(current_features)} features")

    improved = True
    history = []

    while improved and len(current_features) > 1:
        improved = False
        best_rmse = baseline_rmse
        best_combo = None

        # Try removing 1..max_step_size features at a time
        step_size_limit = min(max_step_size, len(current_features) - 1)
        for step_size in range(1, step_size_limit + 1):
            for combo in combinations(current_features, step_size):
                reduced_features = [f for f in current_features if f not in combo]
                X_train_reduced = X_train[reduced_features]
                X_valid_reduced = X_valid[reduced_features]

                rmse = evaluate_model(X_train_reduced, y_train, X_valid_reduced, y_valid, categorical_features)

                if rmse < best_rmse - 1e-4:
                    best_rmse = rmse
                    best_combo = combo

        if best_combo:
            print(f"Removing {best_combo} improved RMSE from {baseline_rmse:.4f} → {best_rmse:.4f}")
            current_features = [f for f in current_features if f not in best_combo]
            baseline_rmse = best_rmse
            improved = True
            history.append((best_combo, best_rmse))
        else:
            print("No further improvement found.")
            break

    return current_features, history


final_features, history = multi_step_backward_elimination(
    X_train, y_train, X_valid, y_valid,
    categorical_features,
    max_step_size=2
)

import pandas as pd
import lightgbm as lgb

def feature_audit(train_df, test_df, target_col, features):
    # Train on train set
    train_model = lgb.LGBMRegressor(random_state=42)
    train_model.fit(train_df[features], train_df[target_col])
    train_importance = pd.Series(train_model.feature_importances_, index=features)

    # Train on test set (pretend we "know" test target for audit purposes)
    test_model = lgb.LGBMRegressor(random_state=42)
    test_model.fit(test_df[features], test_df[target_col])
    test_importance = pd.Series(test_model.feature_importances_, index=features)

    # Compare
    audit_df = pd.DataFrame({
        "train_importance": train_importance,
        "test_importance": test_importance
    })
    audit_df["importance_diff"] = audit_df["train_importance"] - audit_df["test_importance"]
    audit_df["flag_for_removal"] = (audit_df["train_importance"] > audit_df["train_importance"].mean()) & \
                                   (audit_df["test_importance"] < audit_df["test_importance"].mean())
    return audit_df.sort_values("importance_diff", ascending=False)

# Example usage
audit_results = feature_audit(train_df, test_df, "transactions", features)
print(audit_results[audit_results["flag_for_removal"]])


# Detect and mitigate shift
from scipy.stats import ks_2samp
import numpy as np

def detect_distribution_shift(train_df, test_df, features, threshold=0.1):
    shift_report = {}
    for f in features:
        ks_stat, p_value = ks_2samp(train_df[f].dropna(), test_df[f].dropna())
        shift_report[f] = {"ks_stat": ks_stat, "p_value": p_value, "shifted": p_value < threshold}
    return shift_report

# Example usage
features = ["trend_by_hour", "residual_by_dow", "rolling_24h_mean", "rolling_7d_std"]
shift_report = detect_distribution_shift(train_df, test_df, features)
print({f: r for f, r in shift_report.items() if r["shifted"]})

# Mitigation example: Clip extreme values
for f in features:
    low, high = np.percentile(train_df[f].dropna(), [1, 99])
    train_df[f] = np.clip(train_df[f], low, high)
    test_df[f] = np.clip(test_df[f], low, high)
