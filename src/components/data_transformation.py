import os
from dataclasses import dataclass
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

from src.exception import CustomException
from src.logger import logging
import pickle
from src.utils import save_object

def compute_RUL(df):
    max_cycle = df.groupby('unit_number')['time_in_cycles'].transform('max')
    df['RUL'] = max_cycle - df['time_in_cycles']
    return df
    
def RUL_bin(df, threshold = 30):
    df['RUL_bin'] = df['RUL'].apply(lambda x: 1 if x <= threshold else 0)
    return df

def add_rolling_features(df, sensor_cols, window=5):
    try:
        for sensor in sensor_cols:
            df[f'{sensor}_rolling_mean'] = df.groupby('unit_number')[sensor].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            df[f'{sensor}_rolling_std'] = df.groupby('unit_number')[sensor].transform(lambda x: x.rolling(window=window, min_periods=1).std().fillna(0))
            df[f'{sensor}_min'] = df.groupby('unit_number')[sensor].transform(lambda x: x.rolling(window=window, min_periods=1).min())
            df[f'{sensor}_max'] = df.groupby('unit_number')[sensor].transform(lambda x: x.rolling(window=window, min_periods=1).max())
        return df
    except Exception as e:
        raise CustomException(e,sys)


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
    
    def get_data_transformer_object(self, feature_cols):

        try:
            
            num_pipeline = Pipeline(steps =[('scaler', StandardScaler())])
            num_preprocessor = ColumnTransformer([('num_tran', num_pipeline, feature_cols)])
            return num_preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    
    

    def initiate_data_transformation(self, train_path, test_path):
        try:

            
            logging.info('Reading train and test data')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df = compute_RUL(train_df)
            train_df = RUL_bin(train_df)

            test_df = compute_RUL(test_df)
            test_df = RUL_bin(test_df)

            selected_sensors = ['sensor2', 'sensor3', 'sensor4', 'sensor7', 'sensor8', 'sensor11', 'sensor15']

            train_df = add_rolling_features(train_df, sensor_cols = selected_sensors, window = 5)
            test_df = add_rolling_features(test_df, sensor_cols= selected_sensors, window=5)

            feature_cols = [col for col in train_df.columns if col not in ['unit_number', 'RUL', 'RUL_bin']]

            logging.info('Separating features adn target')
            y_col = 'RUL'
            drop_column = ['unit_number', 'RUL', 'RUL_bin']

            X_train_df = train_df.drop(columns = drop_column)
            y_train_df = train_df[y_col]

            X_test_df = test_df.drop(columns = drop_column)
            y_test_df = test_df[y_col]

            logging.info('Applying preprocessing')
            preprocessing_obj = self.get_data_transformer_object(feature_cols)
            X_train_arr = preprocessing_obj.fit_transform(X_train_df)
            X_test_arr = preprocessing_obj.transform(X_test_df)

            logging.info('Saving Preprocessing Object')
            save_object(
                file_path = self.config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            train_arr = np.c_[X_train_arr, y_train_df]
            test_arr = np.c_[X_test_arr, y_test_df]

            logging.info('Data Transformation complete')

            return (train_arr, test_arr, self.config.preprocessor_obj_file_path)
        
        
        except Exception as e:
            raise Exception(e,sys)


