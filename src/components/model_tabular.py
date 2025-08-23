import os
import sys
import json
import math
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, List, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.logger import logging
from src.exception import CustomException
from src.utils import perform_xgb_grid_search


@dataclass
class TabularModelConfig:
    processed_dir: str = "D:/My Projects/Predictive Maintainability RUL/artifacts/processed_tabular"
    train_csv: str = "train_tabular.csv"
    test_csv: str = "test_tabular.csv"
    scaler_path: str = "tabular_scaler.pkl"
    model_path: str = "xgb_tabular_model.pkl"
    predictions_csv: str = "xgb_tabular_predictions.csv"
    metrics_json: str = "xgb_tabular_metrics.json"
    feature_meta_json: str = "tabular_features.json"
    param_grid: Dict[str, List[Any]] = None


class ModelTabular:
    def __init__(self, cfg: TabularModelConfig = None):
        self.cfg = cfg or TabularModelConfig()
        self.train_path = os.path.join(self.cfg.processed_dir, self.cfg.train_csv)
        self.test_path = os.path.join(self.cfg.processed_dir, self.cfg.test_csv)
        self.scaler_path = os.path.join(self.cfg.processed_dir, self.cfg.scaler_path)
        self.model_out = os.path.join(self.cfg.processed_dir, self.cfg.model_path)
        self.preds_out = os.path.join(self.cfg.processed_dir, self.cfg.predictions_csv)
        self.metrics_out = os.path.join(self.cfg.processed_dir, self.cfg.metrics_json)
        self.meta_path = os.path.join(self.cfg.processed_dir, self.cfg.feature_meta_json)
        os.makedirs(self.cfg.processed_dir, exist_ok=True)

        if self.cfg.param_grid is None:
            self.cfg.param_grid = {
                "n_estimators": [800, 1200],
                "learning_rate": [0.02, 0.035, 0.05],
                "max_depth": [5, 6, 7],
            }

    def _load_artifacts(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
        try:
            if not os.path.exists(self.train_path):
                raise CustomException(f"Train CSV not found at: {self.train_path}", sys)
            if not os.path.exists(self.test_path):
                raise CustomException(f"Test CSV not found at: {self.test_path}", sys)
            if not os.path.exists(self.scaler_path):
                raise CustomException(f"Scaler not found at: {self.scaler_path}", sys)
            if not os.path.exists(self.meta_path):
                raise CustomException(f"Feature meta JSON not found at: {self.meta_path}", sys)

            train_df = pd.read_csv(self.train_path)
            test_df = pd.read_csv(self.test_path)
            scaler_obj = joblib.load(self.scaler_path)
            with open(self.meta_path, "r") as f:
                meta = json.load(f)

            feature_cols = scaler_obj.get("feature_cols")
            if feature_cols is None:
                raise CustomException("Scaler object missing 'feature_cols' key.", sys)

            # Basic checks omitted for brevity, copy from earlier as needed.

            return train_df, test_df, scaler_obj, meta
        except Exception as e:
            raise CustomException(e, sys)

    def _prepare_data(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            X_train = train_df[feature_cols].to_numpy(dtype=float)
            y_train = train_df["RUL"].to_numpy(dtype=float)

            mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
            X_train = X_train[mask]
            y_train = y_train[mask]
            if X_train.shape[0] == 0:
                raise CustomException("No valid training rows after filtering non-finite values.", sys)

            X_test = test_df[feature_cols].to_numpy(dtype=float)
            if not np.isfinite(X_test).all():
                col_means = np.nanmean(np.where(np.isfinite(X_train), X_train, np.nan), axis=0)
                inds = ~np.isfinite(X_test)
                X_test[inds] = np.take(col_means, np.where(inds)[1])

            return X_train, y_train, X_test
        except Exception as e:
            raise CustomException(e, sys)

    def train_and_predict(self) -> Tuple[str, str]:
        try:
            train_df, test_df, scaler_obj, _ = self._load_artifacts()
            feature_cols = scaler_obj["feature_cols"]

            X_train, y_train, X_test = self._prepare_data(train_df, test_df, feature_cols)

            model, best_params = perform_xgb_grid_search(X_train, y_train, self.cfg.param_grid)

            joblib.dump(model, self.model_out)
            logging.info(f"Saved best model at: {self.model_out}")
            logging.info(f"Best hyperparameters found: {best_params}")

            y_pred = model.predict(X_test).astype(float)
            out_df = test_df[["unit_number"]].copy()
            out_df["RUL_pred"] = y_pred
            out_df["RUL_true"] = test_df["RUL"].to_numpy(dtype=float)

            out_df = out_df.groupby("unit_number", as_index=False).agg(
                RUL_true=("RUL_true", "max"),
                RUL_pred=("RUL_pred", "mean"),
            ).sort_values("unit_number")

            eval_df = out_df.dropna(subset=["RUL_true"])
            if not eval_df.empty:
                y_true = eval_df["RUL_true"].to_numpy(dtype=float)
                y_hat = eval_df["RUL_pred"].to_numpy(dtype=float)
                rmse = math.sqrt(mean_squared_error(y_true, y_hat))
                mae = mean_absolute_error(y_true, y_hat)
                from sklearn.metrics import r2_score

                r2 = r2_score(y_true, y_hat)
                metrics = {
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "r2": float(r2),
                    "n_eval_units": int(eval_df.shape[0]),
                }
            else:
                metrics = {"rmse": None, "mae": None, "r2": None, "n_eval_units": 0}

            out_df.to_csv(self.preds_out, index=False)
            logging.info(f"Saved predictions at: {self.preds_out}")

            with open(self.metrics_out, "w") as f:
                json.dump(metrics, f, indent=2)
            logging.info(f"Saved metrics at: {self.metrics_out}")

            return self.preds_out, self.metrics_out
        except Exception as e:
            logging.error("Error during training/prediction")
            raise CustomException(e, sys)


if __name__ == "__main__":
    runner = ModelTabular()
    runner.train_and_predict()
