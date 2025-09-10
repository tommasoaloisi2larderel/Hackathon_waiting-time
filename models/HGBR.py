import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

# -------------------------
# Load datasets
# -------------------------
train_df = [pd.read_csv("with_parade1_nightshow.csv"), pd.read_csv("with_all_events.csv"), pd.read_csv("with_no_events.csv")]

test_df = [pd.read_csv("with_parade1_nightshow_test.csv"), pd.read_csv("with_all_events_test.csv"), pd.read_csv("with_no_events_test.csv")]


Hgbr = [HistGradientBoostingRegressor(learning_rate=0.05,max_depth=3,random_state=42) for k in range(3)]
Knn = [KNeighborsRegressor(n_neighbors=25, weights="distance") for k in range(3)]

PredHgbr = []
PredKnn = []

# -------------------------
# Fit
# -------------------------
for i in range(3):
    y_train = train_df[i]['WAIT_TIME_IN_2H']
    X_train = train_df[i].drop(['WAIT_TIME_IN_2H', 'DATETIME','ENTITY_DESCRIPTION_SHORT'])

    Hgbr[i].fit(X_train, y_train)
    Knn[i].fit(X_train, y_train)

# -------------------------
# Predict (test/validation)
# -------------------------
for i in range(3):
    X_test = test_df[i].drop('DATETIME','ENTITY_DESCRIPTION_SHORT')

    PredHgbr.append(Hgbr[i].predict(X_train))
    PredKnn.append(Knn[i].predict(X_train))

# combining preds with the test_df
PredictedTestHgbr = [test_df for k in range(3)]
PredictedTestKnn = PredictedTestHgbr.copy()

for i in range(3):
    PredictedTestHgbr[i]['y_pred'] = PredHgbr[i]
    PredictedTestKnn[i]['y_pred'] = PredKnn[i]

mergedTestPredictionHgbr = pd.concat(PredictedTestHgbr)
mergedTestPredictionKnn = pd.concat(PredictedTestKnn)
# -------------------------
# Save predictions
# -------------------------
outHgbr = mergedTestPredictionHgbr[["DATETIME", "ENTITY_DESCRIPTION_SHORT",'y_pred']].copy()
outHgbr["KEY"] = "Validation"
outHgbr.to_csv("output/TEST_waiting_times_HGBR_divided.csv", index=False)

outKnn = mergedTestPredictionKnn[["DATETIME", "ENTITY_DESCRIPTION_SHORT",'y_pred']].copy()
outKnn["KEY"] = "Validation"
outKnn.to_csv("output/TEST_waiting_times_Knn_divided.csv", index=False)