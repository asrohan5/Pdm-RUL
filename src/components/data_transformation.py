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



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
    
    def get_data_transformer_object(self):

        try:
            num_cols = [
                'time_in_cycles',
                'op_setting_1', 'op_setting_2', 'op_setting_3',
                'sensor2', 'sensor3', 'sensor4', 'sensor6', 'sensor7',
                'sensor8', 'sensor9', 'sensor11', 'sensor12', 'sensor13',
                'sensor14', 'sensor15', 'sensor17', 'sensor20', 'sensor21',
            ]

            num_pipeline = Pipeline(steps =[('scaler', StandardScaler())])

            num_preprocessor = ColumnTransformer([('num_tran', num_pipeline, num_cols)])

            return num_preprocessor
        
        except Exception as e:
            raise ColumnTransformer(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info('Reading train and test data')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Separating features adn target')
            y_col = 'RUL'
            drop_column = ['unit_number', 'RUL', 'RUL_bin']

            X_train_df = train_df.drop(columns = drop_column)
            y_train_df = train_df[y_col]

            X_test_df = test_df.drop(columns = drop_column)
            y_test_df = test_df[y_col]

            logging.info('Applying preprocessing')
            preprocessing_obj = self.get_data_transformer_object()
            X_train_arr = preprocessing_obj.fit_transform(X_train_df)
            X_test_arr = preprocessing_obj.transform(X_test_df)

            logging.info('Saving Preprocessing Object')
            save_object(
                file_path = self.data_tranformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            train_arr = np.c_(X_train_arr, y_train_df)
            test_arr = np.c_(X_test_arr, y_test_df)

            logging.info('Data Transformation complete')

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
        
        
        except Exception as e:
            raise Exception(e,sys)
        
