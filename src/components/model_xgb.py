import os
import sys
import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from src.logger import logging
from src.exception import CustomException


class ModelXGB:
    def __init__(
        self,
        train_path,
        test_path,
        scaler_path="D:/My Projects/Predictive Maintainability RUL/artifacts/processed/scaler.pkl",
        model_path="D:/My Projects/Predictive Maintainability RUL/artifacts/processed/xgb_model.pkl",
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.scaler_path = scaler_path
        self.model_path = model_path


        sensors = [2, 3, 4, 7, 8, 9, 11, 12, 13, 15, 17, 20, 21]
        self.selected_sensors = [f"sensor{i}" for i in sensors]
        self.feature_cols = []
        for sensor in self.selected_sensors:
    
            self.feature_cols.extend([
                sensor,
                f"{sensor}_mean",
                f"{sensor}_std",
                f"{sensor}_min",
                f"{sensor}_max"
            ])

    def train_xgb(self):
        try:
            logging.info("Loading processed train and test data...")

            train_df = pd.read_csv(self.train_path)
            test_df = pd.read_csv(self.test_path)

            scaler = joblib.load(self.scaler_path)

        
            X_train = train_df[self.feature_cols]
            y_train = train_df["RUL"]


            test_features_present = [col for col in self.feature_cols if col in test_df.columns]
            X_test = test_df[test_features_present]
            y_test = test_df["RUL"]

            logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")


            xgb = XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
                eval_metric="rmse"
            )


            param_grid = {
                "n_estimators": [300, 800],
                "max_depth": [10],
                "learning_rate": [0.05],
                "subsample": [0.7, 0.9],
                "colsample_bytree": [0.7, 0.9]
            }

            logging.info("Starting GridSearchCV for XGBoost...")
            grid_search = GridSearchCV(
                estimator=xgb,
                param_grid=param_grid,
                cv=3,
                scoring="neg_root_mean_squared_error",
                verbose=2,
                n_jobs=-1
            )

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            logging.info(f"Best Params: {grid_search.best_params_}")


            y_pred = best_model.predict(X_test)


            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            logging.info("XGBoost Performance:")
            logging.info(f"   Best Params: {grid_search.best_params_}")
            logging.info(f"   Test RMSE: {rmse:.4f}")
            logging.info(f"   Test R2: {r2:.4f}")
            logging.info(f"   Test MAE: {mae:.4f}")


            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, "wb") as f:
                pickle.dump(best_model, f)
            logging.info(f"XGBoost model saved at {self.model_path}")


            metrics_path = os.path.join(os.path.dirname(self.model_path), "xgb_metrics.txt")
            with open(metrics_path, "w") as f:
                f.write(f"Best Params: {grid_search.best_params_}\n")
                f.write(f"Test RMSE: {rmse:.4f}\n")
                f.write(f"Test R2: {r2:.4f}\n")
                f.write(f"Test MAE: {mae:.4f}\n")

            logging.info(f"Metrics saved at {metrics_path}")

        except Exception as e:
            logging.error("Exception occurred in XGBoost training")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = ModelXGB(
        train_path="D:/My Projects/Predictive Maintainability RUL/artifacts/processed/train.csv",
        test_path="D:/My Projects/Predictive Maintainability RUL/artifacts/processed/test.csv"
    )
    obj.train_xgb()
