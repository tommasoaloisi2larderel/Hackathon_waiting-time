import pandas as pd
import os
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, ParameterGrid
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error


weather_df = pd.read_csv("weather_data.csv")
train_df = pd.read_csv("waiting_times_train.csv")
test_df = pd.read_csv("waiting_times_X_test_val.csv")



def format_file(df):
       df = pd.merge(weather_df, df, on="DATETIME")
       # Datetime expansion
       df["DATETIME"] = pd.to_datetime(df["DATETIME"])
       df["year"] = df["DATETIME"].dt.year
       df["month"] = df["DATETIME"].dt.month
       df["dayofweek"] = df["DATETIME"].dt.weekday
       df["hour"] = df["DATETIME"].dt.hour
       df["minute"] = df["DATETIME"].dt.minute
       df["time_minutes"] = df["hour"] * 60 + df["minute"]

       # Cyclical encodings for time-based features
       df["sin_hour"] = np.sin(2 * np.pi * (df["time_minutes"] / (24 * 60)))
       df["cos_hour"] = np.cos(2 * np.pi * (df["time_minutes"] / (24 * 60)))
       df["sin_dow"] = np.sin(2 * np.pi * (df["dayofweek"] / 7))
       df["cos_dow"] = np.cos(2 * np.pi * (df["dayofweek"] / 7))
       df["sin_month"] = np.sin(2 * np.pi * (df["month"] / 12))
       df["cos_month"] = np.cos(2 * np.pi * (df["month"] / 12))

       # Handle missing values in show time columns by pushing them far in the future
       for col in ["TIME_TO_PARADE_2", "TIME_TO_PARADE_1", "TIME_TO_NIGHT_SHOW"]:
              if col in df.columns:
                     df[col] = df[col].fillna(24 * 60)

       # Precipitation NaNs to 0; other weather NaNs will be imputed later in the pipeline
       for col in ["snow_1h", "rain_1h"]:
              if col in df.columns:
                     df[col] = df[col].fillna(0)

       # Drop raw datetime (we keep ENTITY_DESCRIPTION_SHORT for one-hot encoding later)
       df = df.drop(columns=["DATETIME"])

       # Remove previous manual dummies and keep the raw category instead
       # (We intentionally do NOT create 'Water Ride'/'Pirate Ship' columns here.)
       return df

train_df = format_file(train_df)
test_df = format_file(test_df)

# =============================
# Model selection to minimize RMSE
# =============================

# Prepare training matrices
TARGET_COL = "WAIT_TIME_IN_2H"
X_train = train_df.drop(columns=[TARGET_COL])
Y_train = train_df[TARGET_COL]

# Identify feature types automatically to be robust to schema changes
numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

preprocess = ColumnTransformer(
       transformers=[
              ("num", Pipeline(steps=[
                     ("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()),
              ]), numeric_features),
              ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
       ]
)

pipe = Pipeline(steps=[
       ("preprocess", preprocess),
       ("reg", KNeighborsRegressor()),  # placeholder; will be swapped in grid
])

param_grid = [
       {
              "reg": [KNeighborsRegressor()],
              "reg__n_neighbors": [3, 5, 10, 20, 40],
              "reg__weights": ["uniform", "distance"],
              "reg__p": [1, 2],
       },
       {
              "reg": [HistGradientBoostingRegressor(random_state=42)],
              "reg__learning_rate": [0.05, 0.1, 0.2],
              "reg__max_depth": [None, 3, 7],
              "reg__max_leaf_nodes": [31, 127],
       },
       {
              "reg": [RandomForestRegressor(random_state=42)],
              "reg__n_estimators": [300, 600],
              "reg__max_depth": [None, 10, 20],
              "reg__min_samples_leaf": [1, 2, 5],
       },
]

# Time-aware CV (preserves ordering)
cv = TimeSeriesSplit(n_splits=5)

grid = GridSearchCV(
       estimator=pipe,
       param_grid=param_grid,
       scoring="neg_root_mean_squared_error",
       cv=cv,
       n_jobs=-1,
       verbose=2,
)

# Optional progress bar for GridSearchCV (uses tqdm_joblib if installed)
try:
       from tqdm.auto import tqdm  # type: ignore
       from tqdm_joblib import tqdm_joblib  # pip install tqdm tqdm-joblib
       total_candidates = sum(len(list(ParameterGrid(g))) for g in param_grid)
       total_fits = total_candidates * cv.get_n_splits()
       with tqdm_joblib(tqdm(desc="Grid search", total=total_fits)):
              grid.fit(X_train, Y_train)
except Exception:
       # Fallback: run without progress bar
       grid.fit(X_train, Y_train)

best_rmse = -grid.best_score_
print(f"Best CV RMSE: {best_rmse:.3f}")
print("Best model:", grid.best_estimator_)

# Fit the best model on the full training data
best_model = grid.best_estimator_
best_model.fit(X_train, Y_train)

# Predict on the test frame
X_test = test_df.copy()
predictions = best_model.predict(X_test)
if hasattr(predictions, "ravel"):
       predictions = predictions.ravel()

# Attach and save predictions
test_df["PREDICTED_WAIT_TIME_IN_2H"] = predictions
print(test_df[["PREDICTED_WAIT_TIME_IN_2H"]].head())

test_df.to_csv("waiting_times_with_predictions.csv", index=False)

# Challenge file in the expected shape/name
# (keep the same name even if the winning model is not KNN)
df = pd.read_csv("waiting_times_X_test_val.csv")
df_copy = df.copy(deep=True)
cols_to_drop = [
       "ADJUST_CAPACITY",
       "DOWNTIME",
       "CURRENT_WAIT_TIME",
       "TIME_TO_PARADE_1",
       "TIME_TO_PARADE_2",
       "TIME_TO_NIGHT_SHOW",
]
df_copy = df_copy.drop(columns=[c for c in cols_to_drop if c in df_copy.columns])
df_copy["y_pred"] = predictions
# Avoid hard-coding length; fill KEY appropriately
df_copy["KEY"] = "Validation"

df_copy.to_csv("TEST_waiting_times_KNeighborsRegressor.csv", index=False)
