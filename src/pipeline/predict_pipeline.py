import os
import sys
import pandas as pd
import numpy as np

from src.utlis import load_object
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import compute_RUL, add_rolling_features

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'best_model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

        def predict(self, input_df):
            try:
                logging.info('Loading model and preprocessor')
                model = load_object(self.model_path)
                preprocessor = load_object(self.preprocessor_path)

                logging.info('computing RUL and rollling features')
                inpute_df = compute_RUL(input_df)

                selected_sensors = ['sensor2', 'sensor3', 'sensor4', 'sensor7', 'sensor8', 'sensor11', 'sensor15']
                input_df = add_rolling_features(input_df, sensor_cols = selected_senors, window = 5)
                
                drop_columns = ['unit_number', 'RUL', 'RUL_bins'] if 'RUL_bin' in input_df.columns else ['unit_number', 'RUL']
                features = input_df.drop(columns = [col for col in drop_columns if col in input_df.columns])

                transformed_features = preprocessor.transform(features)

                logging.info('Making Predictions')
                predictions = model.predict(transformed_features)

                input_df['Predicted_RUL'] = predictions
                return input_df[['unit_number', 'time_in_cycles', 'Predicted_RUL']].groupby('unit_number').last().reset_index()

            except Exception as e:
                raise CustomException(e,sys)