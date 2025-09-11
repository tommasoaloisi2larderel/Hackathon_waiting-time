import pandas as pd
import numpy as np
import holidays


def load_and_merge_on(path_1, path_2, column="DATETIME"):
    """Load two files and merge on column."""
    file_1 = pd.read_csv(path_1)
    file_2 = pd.read_csv(path_2)
    merged_df = pd.merge(file_1, file_2, on=column, how="left")
    merged_df["DATETIME"] = pd.to_datetime(merged_df["DATETIME"])
    return merged_df

def time(data_df):
    data_df['day'] = (data_df['DATETIME'].dt.hour - 15) / 6
    data_df['week'] = data_df['DATETIME'].dt.dayofweek 
    data_df['year'] = data_df['DATETIME'].dt.year - 2019
    german_holidays = holidays.Germany(years=range(2018, 2023))  # adjust years as needed
    french_holidays = holidays.France(years=range(2018, 2023))  # adjust years as needed
    swiss_holidays = holidays.Switzerland(years=range(2018, 2023))  # adjust years as needed
    data_df['is_holiday'] = data_df['DATETIME'].dt.date.isin(german_holidays) * 0.5 + data_df['DATETIME'].dt.date.isin(french_holidays) * 0.3 + data_df['DATETIME'].dt.date.isin(swiss_holidays) * 0.2
    return data_df

def final_adjustments(data_df):
    cols_to_drop = ['DOWNTIME','ADJUST_CAPACITY','pressure','temp','dew_point','feels_like','humidity','wind_speed','rain_1h','snow_1h','clouds_all','TIME_TO_PARADE_1','TIME_TO_PARADE_2','TIME_TO_NIGHT_SHOW']
    return data_df.drop(columns=cols_to_drop, errors='ignore')

def various(data_df):
    data_df['is_pirate_ship'] = (data_df['ENTITY_DESCRIPTION_SHORT'] == 'Pirate Ship').astype(int)
    data_df['is_flying_coaster'] = (data_df['ENTITY_DESCRIPTION_SHORT'] == 'Flying Coaster').astype(int)
    data_df['capacity_normalized'] = data_df.groupby('ENTITY_DESCRIPTION_SHORT')['ADJUST_CAPACITY'].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x - x.mean())
    data_df['has_downtime'] = (data_df['DOWNTIME'] > 0).astype(int)
    event_cols = ['TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW']
    data_df['time_to_next_event'] = data_df[event_cols].fillna(240).clip(upper=240).min(axis=1)
    return data_df

def weather(data_df):
    # Weather comfort score (0-1, higher = more comfortable)
    temp_comfort = 1 - np.abs(data_df['temp'] - 20) / 25
    temp_comfort = np.clip(temp_comfort, 0, 1)
    
    humidity_comfort = 1 - np.abs(data_df['humidity'] - 50) / 50
    humidity_comfort = np.clip(humidity_comfort, 0, 1)
    
    rain_penalty = np.where(data_df['rain_1h'].fillna(0) > 0, 0.3, 1)
    
    data_df['weather_comfort'] = temp_comfort * humidity_comfort * rain_penalty
    
    return data_df

def run():
    train_df = load_and_merge_on("data/waiting_times_train.csv", "data/weather_data.csv")
    test_df = load_and_merge_on("data/waiting_times_X_test_final.csv", "data/weather_data.csv")

    list_of_dfs = [train_df, test_df]
    list_of_functions = [time, various, weather, final_adjustments]

    for func in list_of_functions:
        for i, df in enumerate(list_of_dfs):
            list_of_dfs[i] = func(df)

    train_df, test_df = list_of_dfs
    train_df.to_csv("data/FINAL_TRAIN.csv", index=False)
    test_df.to_csv("data/FINAL_TEST.csv", index=False)

run()