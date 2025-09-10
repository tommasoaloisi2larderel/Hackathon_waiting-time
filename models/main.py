import pandas as pd
from pathlib import Path

# --- Config: list the model prediction files ---
MODEL_FILES = [
    "output/TEST_waiting_times_HGBR.csv",
    "output/TEST_waiting_times_KNN.csv",
    "output/decision_tree_regression_predictions.csv",
    "output/random_forest_regression_predictions.csv",
    "output/xgboost_regression_predictions.csv",
    "output/linear_regression_predictions.csv",
    "waiting_times_predictions_ALL_forest.csv",
]

RMSE = [9.43, 10.03, 11.87, 9.45, 9.57, 10.06, 9.08]

# Keys that must match across models
KEY_COLS = ["DATETIME", "ENTITY_DESCRIPTION_SHORT", "KEY"]
YPRED = "y_pred"

# Where to save the ensemble
OUT_PATH = Path("output/TEST_waiting_times_ENSEMBLE.csv")


def main():
    print("📦 Loading model files…")

    # Read the first file as the base
    base = pd.read_csv(MODEL_FILES[0])[KEY_COLS + [YPRED]].copy()
    base = base.rename(columns={YPRED: "y_pred_1"})

    # Iteratively merge the rest, keeping only the keys and y_pred
    for idx, path in enumerate(MODEL_FILES[1:], start=2):
        df = pd.read_csv(path)[KEY_COLS + [YPRED]].copy()
        df = df.rename(columns={YPRED: f"y_pred_{idx}"})
        base = base.merge(df, on=KEY_COLS, how="inner")

    # Keep validation rows (as per your workflow)
    base = base[base["KEY"] == "Validation"].copy()

    # Sum and average predictions across models
    import numpy as np

    # Compute weights: inverse of RMSE (lower RMSE = higher weight)
    weights = np.array([1 / (r - 8)**3 for r in RMSE])
    weights /= weights.sum()  # normalize to sum to 1
    y_matrix = base[[f"y_pred_{i+1}" for i in range(len(RMSE))]].values
    base["y_pred"] = (y_matrix * weights).sum(axis=1)

    # Output
    out = base[KEY_COLS + ["y_pred"]]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"✅ Saved ensemble to {OUT_PATH} with shape {out.shape}")


if __name__ == "__main__":
    main()
