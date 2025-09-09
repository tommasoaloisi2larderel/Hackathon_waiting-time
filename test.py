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


# --- Weather column selection diagnostics ---
# The following block tests which weather-related columns are informative and not redundant.
# It includes standardization, correlation, VIF, Ridge regression, and permutation importance.

# 1. Standardize numeric weather columns for fair comparison and VIF computation
from sklearn.preprocessing import StandardScaler

weather_cols = ['temp', 'dew_point', 'feels_like', 'pressure', 'humidity', 'wind_speed', 'rain_1h', 'snow_1h', 'clouds_all']
weather_cols = [col for col in weather_cols if col in train_df.columns]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(train_df[weather_cols]), columns=weather_cols)

# 2. Pearson correlation with WAIT_TIME_IN_2H to assess linear relationships
print("\nPearson correlation with WAIT_TIME_IN_2H:")
print(train_df[weather_cols + ['WAIT_TIME_IN_2H']].corr()['WAIT_TIME_IN_2H'].sort_values())

# 3. Variance Inflation Factor (VIF) to detect multicollinearity/redundancy among weather features
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

print("\nVariance Inflation Factor (VIF):")
vif_data = pd.DataFrame()
vif_data['feature'] = weather_cols
X_vif = X_scaled.replace([np.inf, -np.inf], np.nan).dropna()
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print(vif_data.sort_values(by='VIF', ascending=False))

# 4. Ridge regression to assess feature coefficients while controlling for overfitting
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score

X = train_df[weather_cols].fillna(0)
y = train_df['WAIT_TIME_IN_2H']

ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
ridge.fit(X, y)
print("\nRidge coefficients:")
for feature, coef in zip(weather_cols, ridge.coef_):
    print(f"{feature}: {coef:.4f}")

# 5. Permutation importance with RandomForestRegressor for non-linear feature importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

rf = RandomForestRegressor(n_estimators=50, random_state=42)
rf.fit(X, y)
result = permutation_importance(rf, X, y, n_repeats=10, random_state=42)

print("\nPermutation importance (Random Forest):")
for i in result.importances_mean.argsort()[::-1]:
    print(f"{weather_cols[i]}: {result.importances_mean[i]:.4f}")


# These steps help detect overfitting risk or redundancy among weather features before modeling.

# --- Apply final feature selection on weather columns ---
# Keep low-VIF, informative features; drop highly collinear or negligible ones
_kept = ['snow_1h', 'rain_1h', 'pressure', 'wind_speed']
_drop = ['temp', 'feels_like', 'dew_point', 'humidity', 'clouds_all']

kept_weather_cols = [c for c in _kept if c in train_df.columns]
drop_weather_cols = [c for c in _drop if c in train_df.columns]

train_df = train_df.drop(columns=drop_weather_cols)
