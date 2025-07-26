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
    trained_model_file_path = os.path.join('artificats', 'best_model.pkl')

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
            X_test, y_test = test_array[:, :-1], train_array[:, -1]

            models = {
                'LinearRegression': LinearRegression(),
                'RandomForest': RandomForestRegressor(),
                'GradientBoosting': GradientBoostingRegressor(),
                'XGBoost': XGBRegressor()
            }

            params = {
                'LinearRegression': {},
                'RandomForest':{
                    'n_estimators': [50, 400, 1000],
                    'max_depth': [None, 10, 20],
                },

                'GradientBoosting':{
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators':{50, 100, 350},
                    'subsample':[0.8, 1.0],
                },

                'XGBoost':{
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators':{50, 100, 350},
                }
            }

            model_report: dict = evaluate_models (
                X_train = X_train, y_train = y_train,
                X_test = X_test, y_test = y_test,
                models = models, param =params
            )


            best_model_score = max(model_report.value())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score <0.6:
                raise CustomException('No Model met performance threshold')
            
            logging.info(f'Best Model: {best_model_name} with R2 score: {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            y_pred = best_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            return r2
            


        except Exception as e:
            raise CustomException(e,sys)