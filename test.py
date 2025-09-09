import pandas as pd
import os
from sklearn.neighbors import KNeighborsRegressor



pyFileLoca = os.path.dirname(os.path.realpath(__file__))
print(pyFileLoca)
os.chdir(pyFileLoca)


weather_df = pd.read_csv("weather_data.csv")

times_df = pd.read_csv("waiting_times_train.csv")

train_df = pd.merge(weather_df, times_df, on="DATETIME")



# Adapting columns (Date and attraction):
train_df["DATETIME"] = pd.to_datetime(train_df["DATETIME"])

train_df["year"] = train_df["DATETIME"].dt.year
train_df["month"] = train_df["DATETIME"].dt.month
train_df["day"] = train_df["DATETIME"].dt.weekday
train_df["time"] = train_df["DATETIME"].dt.time
train_df = train_df.drop(columns=["DATETIME"])
train_df["time"] = train_df["time"].apply(lambda t: t.hour * 60 + t.minute)
train_df["Water Ride"] = train_df["ENTITY_DESCRIPTION_SHORT"].apply(lambda x: 1 if x=="Water Ride" else 0)
train_df["Pirate Ship"] = train_df["ENTITY_DESCRIPTION_SHORT"].apply(lambda x: 1 if x=="Pirate Ship" else 0)
train_df = train_df.drop(columns=["ENTITY_DESCRIPTION_SHORT"])
train_df['TIME_TO_PARADE_2'] = train_df['TIME_TO_PARADE_2'].apply(lambda x: 24*60 if pd.isna(x) else x)
train_df['TIME_TO_PARADE_1'] = train_df['TIME_TO_PARADE_1'].apply(lambda x: 24*60 if pd.isna(x) else x)
train_df['TIME_TO_NIGHT_SHOW'] = train_df['TIME_TO_NIGHT_SHOW'].apply(lambda x: 24*60 if pd.isna(x) else x)
train_df['snow_1h'] = train_df['snow_1h'].apply(lambda x: 0 if pd.isna(x) else x)
train_df['rain_1h'] = train_df['rain_1h'].apply(lambda x: 0 if pd.isna(x) else x)


# --- Event-time engineering: nearest event with sign preserved ---
event_cols = [c for c in ['TIME_TO_NIGHT_SHOW', 'TIME_TO_PARADE_1', 'TIME_TO_PARADE_2'] if c in train_df.columns]
if event_cols:
    # Use the event whose absolute minutes is smallest, but keep the original sign (negative means event already passed)
    event_df = train_df[event_cols].copy()
    # Safety: if any residual NaNs slipped through, push them far so real values win
    event_df = event_df.fillna(24*60)
    # Row-wise pick the signed value with minimal absolute magnitude
    nearest_signed = event_df.apply(lambda r: r.loc[r.abs().idxmin()], axis=1)
    train_df['NEAREST_EVENT_MINS'] = nearest_signed

    # Symmetric proximity flags (past or future) based on absolute minutes
    train_df['EVENT_WITHIN_60'] = (train_df['NEAREST_EVENT_MINS'].abs() <= 60).astype(int)
    train_df['EVENT_WITHIN_120'] = (train_df['NEAREST_EVENT_MINS'].abs() <= 120).astype(int)

    # Drop original timers to reduce multicollinearity
    print(f"Engineering NEAREST_EVENT_MINS from columns: {event_cols}")
    train_df = train_df.drop(columns=event_cols)
else:
    print("No event time columns found to engineer.")


# --- Full feature selection diagnostics (across all features) ---
# This block evaluates ALL available features (time, attraction, weather, etc.)
# to determine which columns to keep or discard for predicting WAIT_TIME_IN_2H.
# It computes: missingness/zero-variance checks, Pearson & Spearman correlations,
# univariate F-test, mutual information, Lasso (L1) coefficients, RandomForest
# importances, permutation importance, and VIF for multicollinearity.
# A ranked report is saved to 'feature_selection_report.csv', along with
# 'kept_features.txt' and 'dropped_features.txt'.

import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor

TARGET = 'WAIT_TIME_IN_2H'
assert TARGET in train_df.columns, f"{TARGET} not found in dataframe columns: {train_df.columns.tolist()}"

# Work on a model-ready copy to avoid mutating train_df
df_model = train_df.copy()

# Auto-encode any remaining categorical columns (besides ones already encoded)
cat_cols = df_model.select_dtypes(include=['object', 'category']).columns.tolist()
if cat_cols:
    print(f"One-hot encoding categorical columns: {cat_cols}")
    df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

# Separate features and target
feature_cols = [c for c in df_model.columns if c != TARGET]
X_full = df_model[feature_cols].copy()
y = df_model[TARGET].astype(float)

# Replace infs and handle missingness
X_full = X_full.replace([np.inf, -np.inf], np.nan)

# 0) Drop features with excessive missingness (>40%)
nan_pct = X_full.isna().mean()
too_missing = nan_pct[nan_pct > 0.40].index.tolist()
if too_missing:
    print(f"Dropping features with >40% missingness: {too_missing}")
    X_full = X_full.drop(columns=too_missing)
    feature_cols = [c for c in feature_cols if c not in too_missing]

# Impute remaining missing values with median
X_full = X_full.fillna(X_full.median(numeric_only=True))

# 1) Remove zero/near-zero variance features (<=1 unique value)
nunique = X_full.nunique()
zero_var = nunique[nunique <= 1].index.tolist()
if zero_var:
    print(f"Dropping near/zero-variance features: {zero_var}")
    X_full = X_full.drop(columns=zero_var)
    feature_cols = [c for c in feature_cols if c not in zero_var]

# For all following steps, keep the feature order synced
X_full = X_full[feature_cols]

results = pd.DataFrame(index=feature_cols)

# 2) Pearson & Spearman correlations (absolute value)
try:
    corr_pearson = X_full.join(y).corr(method='pearson')[TARGET].drop(index=[TARGET]).abs()
    results['pearson_r_abs'] = corr_pearson
except Exception as e:
    print(f"Pearson correlation failed: {e}")
    results['pearson_r_abs'] = np.nan

try:
    corr_spearman = X_full.join(y).corr(method='spearman')[TARGET].drop(index=[TARGET]).abs()
    results['spearman_rho_abs'] = corr_spearman
except Exception as e:
    print(f"Spearman correlation failed: {e}")
    results['spearman_rho_abs'] = np.nan

# 3) Univariate F-test (f_regression)
try:
    F, pvals = f_regression(X_full.values, y.values)
    results['f_stat'] = F
    results['neg_log10_p'] = -np.log10(np.maximum(pvals, 1e-300))
except Exception as e:
    print(f"f_regression failed: {e}")
    results['f_stat'] = np.nan
    results['neg_log10_p'] = np.nan

# 4) Mutual Information (nonlinear dependencies)
try:
    # Heuristic: treat ints/bools with small cardinality as discrete
    discrete_mask = X_full.apply(lambda s: (np.issubdtype(s.dtype, np.integer) or s.dtype==bool) and s.nunique() <= 20).values
    mi = mutual_info_regression(X_full.values, y.values, discrete_features=discrete_mask, random_state=42)
    results['mutual_info'] = mi
except Exception as e:
    print(f"mutual_info_regression failed: {e}")
    results['mutual_info'] = np.nan

# 5) Lasso (L1) coefficients with CV
try:
    lasso_pipe = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=5000))
    lasso_pipe.fit(X_full.values, y.values)
    lasso_coefs = np.abs(lasso_pipe.named_steps['lassocv'].coef_)
    results['lasso_coef_abs'] = lasso_coefs
    print(f"Lasso chosen alpha: {lasso_pipe.named_steps['lassocv'].alpha_:.6f}")
except Exception as e:
    print(f"LassoCV failed: {e}")
    results['lasso_coef_abs'] = np.nan

# 6) RandomForest feature importance (impurity-based)
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
try:
    rf.fit(X_full.values, y.values)
    results['rf_importance'] = rf.feature_importances_
except Exception as e:
    print(f"RandomForest fit failed: {e}")
    results['rf_importance'] = np.nan

# 7) Permutation importance (model-agnostic)
try:
    perm = permutation_importance(rf, X_full.values, y.values, n_repeats=5, random_state=42, n_jobs=-1)
    results['permutation_importance'] = perm.importances_mean
except Exception as e:
    print(f"Permutation importance failed: {e}")
    results['permutation_importance'] = np.nan

# 8) VIF (multicollinearity among features)
try:
    X_scaled_for_vif = pd.DataFrame(StandardScaler().fit_transform(X_full.values), columns=X_full.columns)
    vif_vals = []
    for i in range(X_scaled_for_vif.shape[1]):
        vif_vals.append(variance_inflation_factor(X_scaled_for_vif.values, i))
    results['vif'] = vif_vals
except Exception as e:
    print(f"VIF calculation failed: {e}")
    results['vif'] = np.nan

# 9) Normalize positive-direction scores to 0-1 for aggregation
score_cols = ['pearson_r_abs', 'spearman_rho_abs', 'f_stat', 'neg_log10_p', 'mutual_info', 'lasso_coef_abs', 'rf_importance', 'permutation_importance']
for col in score_cols:
    if col in results.columns:
        col_vals = results[col].astype(float)
        m, M = np.nanmin(col_vals), np.nanmax(col_vals)
        results[col + '_norm'] = (col_vals - m) / (M - m + 1e-12)

norm_cols = [c for c in results.columns if c.endswith('_norm')]
results['keep_score'] = results[norm_cols].mean(axis=1)

# 10) Recommend drop candidates: high VIF AND low predictive score
VIF_THRESHOLD = 10.0
KEEP_FRACTION = 0.50  # keep the top 50% by aggregate score

# Low score threshold at the (1 - KEEP_FRACTION) quantile
low_score_cut = results['keep_score'].quantile(1 - KEEP_FRACTION)

high_vif = results['vif'] > VIF_THRESHOLD if 'vif' in results else pd.Series(False, index=results.index)
low_score = results['keep_score'] <= low_score_cut
results['drop_flag'] = high_vif & low_score

recommended_keep = results.index[~results['drop_flag']].tolist()
recommended_drop = results.index[results['drop_flag']].tolist()

# 11) Print concise summary
print("\n=== Feature Selection Summary (top 30 by keep_score) ===")
summary_cols = ['keep_score', 'pearson_r_abs', 'spearman_rho_abs', 'mutual_info', 'lasso_coef_abs', 'rf_importance', 'permutation_importance', 'vif']
existing_summary_cols = [c for c in summary_cols if c in results.columns]
print(results.sort_values('keep_score', ascending=False)[existing_summary_cols].head(30))
print(f"\nRecommended KEEP count: {len(recommended_keep)}")
print(f"Recommended DROP features (high VIF & low score): {recommended_drop}")

# 12) Persist detailed report & lists
results.sort_values('keep_score', ascending=False).to_csv('feature_selection_report.csv', index=True)
with open('kept_features.txt', 'w') as f:
    for c in recommended_keep:
        f.write(c + '\n')
with open('dropped_features.txt', 'w') as f:
    for c in recommended_drop:
        f.write(c + '\n')

# 13) (Optional) a ready-to-use matrix of selected features for modeling
X_selected = df_model[recommended_keep].copy()
print(f"X_selected shape: {X_selected.shape}")
