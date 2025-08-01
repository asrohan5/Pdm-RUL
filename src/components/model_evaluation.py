import sys
import pandas as pd
import numpy as np
import os

from src.utils import load_object
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import compute_RUL

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def read_raw_test_file(file_path):
    try:
        column_names = [
            'unit_number', 'time_in_cycles', 
            'op_setting_1', 'op_setting_2', 'op_setting_3'
        ] + [f'sensor_{i}' for i in range(1,22)]

        df = pd.read_csv(file_path, sep = "\s+", header = None)
        df.columns = column_names
        return df
    except Exception as e:
        raise CustomException(e,sys)



class ModelEvaluation:
    def __init__(self, model_path: str, preprocessor_path: str):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path

    def evaluate(self):
        try:
            logging.info('Loading trained model and preprocessor')
            
            test_path = os.path.join('artifacts', 'raw', 'test_FD001.txt')
            raw_test_df = read_raw_test_file(test_path)

            raw_test_df = compute_RUL(raw_test_df)
            y_true = raw_test_df.groupby('unit_number')['RUL'].first().values


            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            logging.info('Computing RUL from test data')
            raw_test_df['RUL'] = raw_test_df.groupby('unit_number')['time_in_cycles'].transform('max') - raw_test_df['time_in_cycles']

            
            X_test_df = raw_test_df[preprocessor.feature_names_in_]
            X_test_transformed = preprocessor.transform(X_test_df)

            raw_test_df['predicted_RUL'] = model.predict(X_test_transformed)
            y_pred = raw_test_df.groupby('unit_number')['predicted_RUL'].last().values

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            logging.info(f"Evaluation Complete - RMSE: {rmse}, MAE: {mae}, R2: {r2}")
            print(f" Model Evaluation Results:\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR2 Score: {r2:.4f}")

            return {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    try:
        
        evaluator = ModelEvaluation(
            model_path = 'D:/My Projects/Predictive Maintainability RUL/artifacts/best_model.pkl',
            preprocessor_path='D:/My Projects/Predictive Maintainability RUL/artifacts/preprocessor.pkl'
        
        )

        metrics = evaluator.evaluate()

        print('Final Evaluation Metrics: ', metrics)
    
    except Exception as ex:
        raise CustomException(ex, sys)