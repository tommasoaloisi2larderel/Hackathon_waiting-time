import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

# -------------------------
# Load and merge datasets
# -------------------------
weather_df = pd.read_csv("data/weather_data.csv")

train_df = pd.read_csv("data/waiting_times_train.csv")
train_df = pd.merge(weather_df, train_df, on="DATETIME")

test_df = pd.read_csv("data/waiting_times_X_test_val.csv")
test_df = pd.merge(weather_df, test_df, on="DATETIME")

# -------------------------
# Feature engineering
# -------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["DATETIME"])

    # keep a simple, non-cyc hour feature since you don't span midnight
    df["year"] = dt.dt.year
    df["time"] = dt.dt.hour * 60 + dt.dt.minute  # 0..1439 but within open hours

    # cyclical week & month (adjacency Sun↔Mon, Dec↔Jan)
    dow = dt.dt.weekday                 # 0..6
    month0 = dt.dt.month - 1            # 0..11

    df["sin_dow"]   = np.sin(2*np.pi * (dow/7))
    df["cos_dow"]   = np.cos(2*np.pi * (dow/7))
    df["sin_month"] = np.sin(2*np.pi * (month0/12))
    df["cos_month"] = np.cos(2*np.pi * (month0/12))

    # you can drop the ordinal versions to avoid the model overusing them
    for col in ["day", "month"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df

train_df = add_time_features(train_df)
test_df = add_time_features(test_df)

# Columns for the pipeline
# Numeric: all numeric columns except the target
# Categorical: just 'ENTITY_DESCRIPTION_SHORT' per spec
numeric_features = train_df.select_dtypes(include=["number"]).columns.tolist()
if "WAIT_TIME_IN_2H" in numeric_features:
    numeric_features.remove("WAIT_TIME_IN_2H")

categorical_features = ["ENTITY_DESCRIPTION_SHORT"]

# -------------------------
# Preprocessing + Model pipeline
# -------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

# Use dense output so the downstream regressor always receives a dense array
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)

regressor = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", regressor),
])

# -------------------------
# Fit
# -------------------------
X_train = train_df[numeric_features + categorical_features]
y_train = train_df["WAIT_TIME_IN_2H"]

model.fit(X_train, y_train)

# -------------------------
# Predict (test/validation)
# -------------------------
X_test = test_df[numeric_features + categorical_features]
pred = model.predict(X_test)

# -------------------------
# Save predictions
# -------------------------
out = test_df[["DATETIME", "ENTITY_DESCRIPTION_SHORT"]].copy()
out["y_pred"] = pred
out["KEY"] = "Validation"
out.to_csv("output/TEST_waiting_times_HGBR.csv", index=False)