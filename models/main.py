import pandas as pd
import os
from sklearn.neighbors import KNeighborsRegressor

model_1 = pd.read_csv("output/TEST_waiting_times_HGBR.csv")
model_2 = pd.read_csv("output/TEST_waiting_times_KNN.csv")
model_3 = pd.read_csv("output/decision_tree_regression_predictions.csv")
model_4 = pd.read_csv("output/random_forest_regression_predictions.csv")
model_5 = pd.read_csv("output/xgboost_regression_predictions.csv")
model_6 = pd.read_csv("output/linear_regression_predictions.csv")
model = model_1.copy()

model["y_pred"] =  model_1["y_pred"] + model_2["y_pred"] + model_4["y_pred"] + model_5["y_pred"] + model_6["y_pred"]
model["y_pred"] = model["y_pred"] / 5

model.to_csv("output/TEST_waiting_times_ENSEMBLE.csv", index=False)