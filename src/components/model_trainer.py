import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException

class ModelTrainerConfig:
    train_path = os.path.join("D:/My Projects/Predictive Maintainability RUL/artifacts", "processed", "train.csv")
    test_path = os.path.join("D:/My Projects/Predictive Maintainability RUL/artifacts", "processed", "test.csv")
    model_save_path = os.path.join("D:/My Projects/Predictive Maintainability RUL/artifacts", "best_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def load_data(self):
        """Load processed train & test data."""
        train_df = pd.read_csv(self.config.train_path)
        test_df = pd.read_csv(self.config.test_path)

        X_train = train_df.drop(columns=["RUL", "unit_number"], errors="ignore")
        y_train = train_df["RUL"]

        X_test = test_df.drop(columns=["RUL", "unit_number"], errors="ignore")
        y_test = test_df["RUL"]

        return X_train, X_test, y_train, y_test

    def get_model_and_params(self, model_name):
        """Return model and parameter grid based on name."""
        if model_name == "RandomForest":
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5]
            }

        elif model_name == "XGBoost":
            model = XGBRegressor(objective="reg:squarederror", random_state=42)
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.01, 0.1, 0.2]
            }

        elif model_name == "Lasso":
            model = Lasso(random_state=42, max_iter=5000)
            param_grid = {
                "alpha": [0.001, 0.01, 0.1, 1, 10]
            }

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        return model, param_grid

    def train_and_evaluate(self, model_name):
        try:
            logging.info(f"Loading data for training {model_name}")
            X_train, X_test, y_train, y_test = self.load_data()

            model, param_grid = self.get_model_and_params(model_name)

            logging.info(f"Starting GridSearchCV for {model_name}")
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1, verbose=2)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            logging.info(f"Best Params for {model_name}: {grid_search.best_params_}")

            # Evaluate on test
            y_pred = best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            logging.info(f"{model_name} Test RMSE: {rmse:.4f}, R2: {r2:.4f}")
            print(f"\n {model_name} Performance:")
            print(f"   Best Params: {grid_search.best_params_}")
            print(f"   Test RMSE: {rmse:.4f}")
            print(f"   Test R2: {r2:.4f}")

            # Save model
            os.makedirs(os.path.dirname(self.config.model_save_path), exist_ok=True)
            joblib.dump(best_model, self.config.model_save_path)
            logging.info(f"Model saved at {self.config.model_save_path}")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model to train: RandomForest | XGBoost | Lasso")
    args = parser.parse_args()

    trainer = ModelTrainer()
    trainer.train_and_evaluate(args.model)
