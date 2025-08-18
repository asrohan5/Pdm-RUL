import os
import sys
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.exception import CustomException
from src.logger import logging


class ModelEvaluator:
    def __init__(self, model_path="D:/My Projects/Predictive Maintainability RUL/artifacts/best_model.pkl",
                 preprocessor_path="D:/My Projects/Predictive Maintainability RUL/artifacts/preprocessor.pkl",
                 test_data_path="D:/My Projects/Predictive Maintainability RUL/artifacts/processed/processed_test.csv"):
        try:
            self.model = self.load_object(model_path)
            self.preprocessor = self.load_object(preprocessor_path)
            self.test_data_path = test_data_path
            logging.info("ModelEvaluator initialized successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def load_object(self, file_path):
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate(self):
        try:
            # Load processed test dataset (has engineered features)
            logging.info(f"Loading test dataset from {self.test_data_path}")
            test_df = pd.read_csv(self.test_data_path)

            # Separate features & target
            X_test_df = test_df[self.preprocessor.feature_names_in_]
            y_test = test_df["RUL"]

            logging.info(f"Shape of X_test: {X_test_df.shape}, y_test: {y_test.shape}")

            # Transform test features
            X_test = self.preprocessor.transform(X_test_df)

            # Predictions
            y_pred = self.model.predict(X_test)

            # Evaluation metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
            logging.info(f"Evaluation Metrics: {metrics}")

            return metrics

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate()
        print("Final Evaluation Metrics:", metrics)
    except Exception as ex:
        raise CustomException(ex, sys)
