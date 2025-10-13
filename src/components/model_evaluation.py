import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_predictions(
    predictions_csv: str,
    metrics_json: str = None,
    plot: bool = False
) -> dict:
    df = pd.read_csv(predictions_csv)
    if "RUL_true" not in df.columns or "RUL_pred" not in df.columns:
        raise ValueError("Predictions CSV must include columns RUL_true and RUL_pred.")
    df = df.dropna(subset=["RUL_true", "RUL_pred"])
    y_true = df["RUL_true"].values
    y_pred = df["RUL_pred"].values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    results = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "n_samples": int(len(y_true)),
    }
    print(f"RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nR²: {r2:.3f}, N: {len(y_true)}")
    if metrics_json is not None:
        with open(metrics_json, "w") as f:
            json.dump(results, f, indent=2)
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "--k", lw=2)
        plt.xlabel("True RUL")
        plt.ylabel("Predicted RUL")
        plt.title("Predicted vs True RUL")
        plt.show()
        plt.figure()
        plt.hist(y_true - y_pred, bins=30, alpha=0.7)
        plt.xlabel("Error (True - Predicted RUL)")
        plt.ylabel("Frequency")
        plt.title("Prediction Errors")
        plt.show()
    return results

if __name__ == "__main__":

    base_dir = "D:/My Projects/Predictive Maintainability RUL/artifacts/processed_tabular"
    predictions_csv = os.path.join(base_dir, "xgb_tabular_predictions.csv")
    metrics_json = os.path.join(base_dir, "xgb_tabular_eval_metrics.json")
    evaluate_predictions(predictions_csv, metrics_json, plot=False)

