import os
import numpy as np
import pandas as pd
import sys

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainConfig:
    trained_model_file_path = os.path.join('artifacts', 'best_model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainConfig()
    
    def evaluate_model(self, model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        result1 = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, result1))
        mae = mean_absolute_error(y_test, result1)
        r2 = r2_score(y_test, result1)
        
        return {'model': model, 'rmse': rmse, 'mae': mae, 'r2':r2}

    def initate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting X and y')
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                'LinearRegression': LinearRegression(),
                'RandomForest': RandomForestRegressor(),
                'GradientBoosting': GradientBoostingRegressor(),
                'XGBoost': XGBRegressor()
            }

            params = {
                'LinearRegression': {},
                'RandomForest':{
                    'n_estimators': [100, 500, 1000],
                    'max_depth': [None, 10, 20],
                },

                'GradientBoosting':{
                    'learning_rate': [0.1, 0.05],
                    'n_estimators':[300, 400, 900],
                    'subsample':[1.0],
                },

                'XGBoost':{
                    'learning_rate': [00.1, 0.05],
                    'n_estimators':[350, 500, 1000],
                }
            }

            model_report: dict = evaluate_models (
                X_train = X_train, y_train = y_train,
                X_test = X_test, y_test = y_test,
                models = models, param =params
            )


            best_model_name, best_model_metrics = max(model_report.items(), key=lambda x: x[1]['r2'])
            best_model = best_model_metrics['model']
            logging.info(f"Best Model: {best_model_name} | R2: {best_model['r2']:.4f}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model['model'] 
            )


            return best_model_metrics
            


        except Exception as e:
            raise CustomException(e,sys)
