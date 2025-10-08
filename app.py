import os
import json
import traceback
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import joblib

# Optional: if placing app.py at repo root, adjust import path as needed
from src.components.data_transformation_tabular import DataTransformationTabular, TabularTransformConfig

# Configure paths to existing artifacts
BASE_DIR = "D:/My Projects/Predictive Maintainability RUL/artifacts/processed_tabular"
SCALER_PATH = os.path.join(BASE_DIR, "tabular_scaler.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "xgb_tabular_model.pkl")

app = Flask(__name__)

# Load artifacts once
artifacts = {}
def load_artifacts():
    global artifacts
    scaler_obj = joblib.load(SCALER_PATH)  # contains {"scaler": StandardScaler, "feature_cols": [...]}
    feature_cols = scaler_obj["feature_cols"]
    model = joblib.load(MODEL_PATH)
    artifacts = {
        "feature_cols": feature_cols,
        "model": model,
        "scaler_obj": scaler_obj,
    }

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict-json", methods=["POST"])
def predict_json():
    """
    Expects JSON with a field 'instances' which is a list of dicts.
    Each dict must have all feature columns in artifacts['feature_cols'].
    Example:
    {
      "instances": [
        {"sensor2": 0.11, "sensor3": -0.08, ..., "window_len": 50},
        {"sensor2": 0.05, "sensor3": 0.01, ..., "window_len": 80}
      ]
    }
    """
    try:
        payload = request.get_json(force=True)
        if payload is None or "instances" not in payload:
            return jsonify({"error": "JSON must include 'instances' (list of feature dicts)."}), 400

        feature_cols = artifacts["feature_cols"]
        model = artifacts["model"]

        # Convert instances into DataFrame with the exact feature schema
        rows = payload["instances"]
        X_df = pd.DataFrame(rows)
        # Fill any missing required columns with 0.0; reorder columns
        for c in feature_cols:
            if c not in X_df.columns:
                X_df[c] = 0.0
        X_df = X_df[feature_cols]
        X = X_df.to_numpy(dtype=float)

        y_pred = model.predict(X).astype(float).tolist()
        return jsonify({"predictions": y_pred, "count": len(y_pred)}), 200
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/predict-from-raw", methods=["POST"])
def predict_from_raw():
    """
    Accepts JSON with:
    {
      "raw_test_path": "D:/.../test_FD001.txt",
      "rul_test_path": "D:/.../RUL_FD001.txt"   # optional, only needed if you also want to return RUL_true
    }
    It will:
    - run the existing DataTransformationTabular to compute features for the test set
    - load the saved model and output predictions per unit_number
    """
    try:
        data = request.get_json(force=True)
        raw_test_path = data.get("raw_test_path")
        rul_test_path = data.get("rul_test_path", None)

        if not raw_test_path or not os.path.exists(raw_test_path):
            return jsonify({"error": "Valid 'raw_test_path' is required"}), 400

        # Set up transformation config pointing to the raw files' directory
        processed_dir = BASE_DIR
        cfg = TabularTransformConfig(
            raw_dir=os.path.dirname(raw_test_path),
            test_raw=os.path.basename(raw_test_path),
            rul_raw=os.path.basename(rul_test_path) if rul_test_path else "RUL_FD001.txt",
            processed_dir=processed_dir,
            include_ops=False
        )
        transformer = DataTransformationTabular(cfg)
        # initiate() will rebuild train/test processed files; we only need test here, but it's fine to run both
        _, test_out_path = transformer.initiate()

        test_df = pd.read_csv(test_out_path)
        feature_cols = artifacts["feature_cols"]
        model = artifacts["model"]

        X = test_df[feature_cols].to_numpy(dtype=float)
        y_pred = model.predict(X).astype(float)
        out = test_df[["unit_number"]].copy()
        out["RUL_pred"] = y_pred

        # If ground truth is present (when RUL file supplied), include it
        if "RUL" in test_df.columns:
            out["RUL_true"] = test_df["RUL"].to_numpy(dtype=float)

        # Aggregate to one row per unit_number
        out = out.groupby("unit_number", as_index=False).agg(
            RUL_pred=("RUL_pred", "mean"),
            RUL_true=("RUL_true", "max")
        )

        # Return top few rows and summary stats to keep response small
        preview = out.head(10).to_dict(orient="records")
        response = {
            "preview": preview,
            "n_units": int(out.shape),
        }
        # Optionally compute quick metrics if labels exist
        if "RUL_true" in out.columns and out["RUL_true"].notna().any():
            try:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                valid = out.dropna(subset=["RUL_true"])
                y_true = valid["RUL_true"].to_numpy(dtype=float)
                y_hat = valid["RUL_pred"].to_numpy(dtype=float)
                rmse = float(np.sqrt(mean_squared_error(y_true, y_hat)))
                mae = float(mean_absolute_error(y_true, y_hat))
                r2 = float(r2_score(y_true, y_hat))
                response["metrics"] = {"rmse": rmse, "mae": mae, "r2": r2, "n_eval_units": int(valid.shape)}
            except Exception:
                # metrics are optional; ignore failures
                pass

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    load_artifacts()
    # For local demo use debug=True; turn off in production
    app.run(host="0.0.0.0", port=5000, debug=True)
