import pandas as pd
import os
from sklearn.neighbors import KNeighborsRegressor



pyFileLoca = os.path.dirname(os.path.realpath(__file__))
print(pyFileLoca)
os.chdir(pyFileLoca)


weather_df = pd.read_csv("weather_data.csv")

train_df = pd.read_csv("waiting_times_train.csv")
test_df = pd.read_csv("waiting_times_X_test_val.csv")

weather_df = weather_df.drop(columns=["cloud_all"])
weather_df = weather_df.drop(columns=["temp"])
weather_df = weather_df.drop(columns=["dew_point"])
weather_df = weather_df.drop(columns=["humidity"])


def format_file(df):
       df = pd.merge(weather_df, df, on="DATETIME")
       # Adapting columns (Date and attraction):
       df["DATETIME"] = pd.to_datetime(df["DATETIME"])
       df["year"] = df["DATETIME"].dt.year
       df["month"] = df["DATETIME"].dt.month
       df["day"] = df["DATETIME"].dt.weekday
       df["time"] = df["DATETIME"].dt.time
       df = df.drop(columns=["DATETIME"])
       df["time"] = df["time"].apply(lambda t: t.hour * 60 + t.minute)

       df["Water Ride"] = df["ENTITY_DESCRIPTION_SHORT"].apply(lambda x: 1 if x=="Water Ride" else 0)
       df["Pirate Ship"] = df["ENTITY_DESCRIPTION_SHORT"].apply(lambda x: 1 if x=="Pirate Ship" else 0)
       df = df.drop(columns=["ENTITY_DESCRIPTION_SHORT"])

       df['TIME_TO_PARADE_2'] = df['TIME_TO_PARADE_2'].apply(lambda x: 24*60 if pd.isna(x) else x)
       df['TIME_TO_PARADE_1'] = df['TIME_TO_PARADE_1'].apply(lambda x: 24*60 if pd.isna(x) else x)
       df['TIME_TO_NIGHT_SHOW'] = df['TIME_TO_NIGHT_SHOW'].apply(lambda x: 24*60 if pd.isna(x) else x)

       df['snow_1h'] = df['snow_1h'].apply(lambda x: 0 if pd.isna(x) else x)
       df['rain_1h'] = df['rain_1h'].apply(lambda x: 0 if pd.isna(x) else x)
       return df

train_df = format_file(train_df)
test_df = format_file(test_df)

model = KNeighborsRegressor(n_neighbors=5)

print(train_df.columns)


X = train_df[['feels_like', 'pressure', 'wind_speed',
       'rain_1h', 'snow_1h', 'ADJUST_CAPACITY', 'DOWNTIME',
       'CURRENT_WAIT_TIME', 'TIME_TO_PARADE_1', 'TIME_TO_PARADE_2',
       'TIME_TO_NIGHT_SHOW', 'year', 'month', 'day', 'time',
       'Water Ride', 'Pirate Ship']]

Y = train_df[['WAIT_TIME_IN_2H']]


model.fit(X,Y)


# S'assurer que toutes les colonnes nécessaires sont présentes
X_test = test_df[['feels_like', 'pressure', 'wind_speed',
       'rain_1h', 'snow_1h', 'ADJUST_CAPACITY', 'DOWNTIME',
       'CURRENT_WAIT_TIME', 'TIME_TO_PARADE_1', 'TIME_TO_PARADE_2',
       'TIME_TO_NIGHT_SHOW', 'year', 'month', 'day', 'time',
       'Water Ride', 'Pirate Ship']]

# Faire la prédiction
predictions = model.predict(X_test)

# Ajouter la prédiction au DataFrame pour consultation ou sauvegarde
test_df['PREDICTED_WAIT_TIME_IN_2H'] = predictions

# Optionnel : afficher ou sauvegarder les résultats
print(test_df[['PREDICTED_WAIT_TIME_IN_2H']].head())

# Sauvegarder dans un fichier CSV
test_df.to_csv("waiting_times_with_predictions.csv", index=False)

# Predictions 
df = pd.read_csv("waiting_times_X_test_val.csv")
df_copy = df.copy(deep=True)
df_copy = df_copy.drop(columns=["ADJUST_CAPACITY"])
df_copy = df_copy.drop(columns=["DOWNTIME"])
df_copy = df_copy.drop(columns=["CURRENT_WAIT_TIME"])
df_copy = df_copy.drop(columns=["TIME_TO_PARADE_1"])
df_copy = df_copy.drop(columns=["TIME_TO_PARADE_2"])
df_copy = df_copy.drop(columns=["TIME_TO_NIGHT_SHOW"])
LSTpredictions = predictions.ravel().tolist()
df_copy["y_pred"] = LSTpredictions

key_=["Validation" for i in range (0,2444)]
df_copy["KEY"]=key_
df_copy

df_copy.to_csv("TEST_waiting_times_KNeighborsRegressor.csv", index=False)

