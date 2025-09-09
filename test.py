import pandas as pd
import os
from sklearn.neighbors import KNeighborsRegressor



pyFileLoca = os.path.dirname(os.path.realpath(__file__))
print(pyFileLoca)
os.chdir(pyFileLoca)

train_df = pd.read_csv("waiting_times_train.csv")
# Adapting columns (Date and attraction):
train_df["DATETIME"] = pd.to_datetime(train_df["DATETIME"])
train_df["year"] = train_df["DATETIME"].dt.year
train_df["month"] = train_df["DATETIME"].dt.month
train_df["day"] = train_df["DATETIME"].dt.dayofweek  # 0=Mon, 6=Sun
train_df["time"] = train_df["DATETIME"].dt.time
train_df = train_df.drop(columns=["DATETIME"])
train_df["time"] = train_df["time"].apply(lambda t: t.hour * 60 + t.minute)

# Harmonize parade column names from raw to standardized if needed
print(train_df["day"].unique())