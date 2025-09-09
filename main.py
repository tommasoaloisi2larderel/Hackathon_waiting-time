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

model = KNeighborsRegressor(n_neighbors=5)

print(train_df.columns)


X = train_df[['feels_like', 'pressure', 'wind_speed',
       'rain_1h', 'snow_1h', 'ADJUST_CAPACITY', 'DOWNTIME',
       'CURRENT_WAIT_TIME', 'TIME_TO_PARADE_1', 'TIME_TO_PARADE_2',
       'TIME_TO_NIGHT_SHOW', 'year', 'month', 'day', 'time',
       'Water Ride', 'Pirate Ship']]

Y = train_df[['WAIT_TIME_IN_2H']]


model.fit(X,Y)


