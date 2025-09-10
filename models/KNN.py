import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# Load and merge datasets
weather_df = pd.read_csv("data/weather_data.csv")
train_df = pd.read_csv("data/waiting_times_train.csv")
train_df = pd.merge(weather_df, train_df, on="DATETIME")

test_df = pd.read_csv("data/waiting_times_X_test_val.csv")
test_df = pd.merge(weather_df, test_df, on="DATETIME")

# Common preprocessing function
def preprocess(df, is_train=True, cat_columns=None, scaler=None):
    df = df.copy()
    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df["year"] = df["DATETIME"].dt.year
    df["month"] = df["DATETIME"].dt.month
    df["day"] = df["DATETIME"].dt.weekday
    df["time"] = df["DATETIME"].dt.hour * 60 + df["DATETIME"].dt.minute
    df = df.drop(columns=["DATETIME"])

    # One-hot encode attraction type
    df = pd.get_dummies(df, columns=["ENTITY_DESCRIPTION_SHORT"], prefix="ATTRACTION", drop_first=False)

    # Fill missing values in numeric features
    base_cols = ['ADJUST_CAPACITY', 'DOWNTIME', 'CURRENT_WAIT_TIME', 'year', 'month', 'day', 'time']
    df[base_cols] = df[base_cols].fillna(0)

    # For test set: align columns and reuse scaler
    if not is_train:
        for col in cat_columns:
            if col not in df.columns:
                df[col] = 0
        extra = [c for c in df.columns if c.startswith("ATTRACTION_") and c not in cat_columns]
        df = df.drop(columns=extra)

        df[base_cols] = pd.DataFrame(scaler.transform(df[base_cols]), columns=base_cols, index=df.index)
        df = df[base_cols + cat_columns]
        return df

    # For train set: determine final columns and fit scaler
    cat_columns = [c for c in df.columns if c.startswith("ATTRACTION_")]
    scaler = StandardScaler()
    df[base_cols] = pd.DataFrame(scaler.fit_transform(df[base_cols]), columns=base_cols, index=df.index)
    X = df[base_cols + cat_columns]
    y = df["WAIT_TIME_IN_2H"]
    return X, y, cat_columns, scaler

# Process training data
X_train, y_train, cat_columns, scaler = preprocess(train_df, is_train=True)

# Train model
model = KNeighborsRegressor(n_neighbors=25, weights="distance")
model.fit(X_train, y_train)

# Process test data
X_test = preprocess(test_df, is_train=False, cat_columns=cat_columns, scaler=scaler)

# Predict
predictions = model.predict(X_test)

# Prepare output
df_out = test_df.copy()
df_out = df_out[["DATETIME"]].copy()
df_out["ENTITY_DESCRIPTION_SHORT"] = test_df["ENTITY_DESCRIPTION_SHORT"]
df_out["y_pred"] = predictions
df_out["KEY"] = "Validation"
df_out.to_csv("output/TEST_waiting_times_KNN.csv", index=False)

