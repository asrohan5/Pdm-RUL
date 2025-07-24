import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    raw_data_dir:str = os.path.join("artifacts", "raw")
    processed_data_dir:str = os.path.join("artifacts", "processed")
    train_data_path:str = os.path.join("artifacts", "processed", "train.csv")
    test_data_path:str = os.path.join("artifacts", "processed", "test.csv")
    rul_data_path:str = os.path.join("artifacts", "processed", "rul.csv")


class DataIngestion:
    
    def __init__(self):
        self.config = DataIngestionConfig()
        self.column_names = [
            'unit_number', 'time_in_cycles',
            'op_setting_1', 'op_setting_2', 'op_setting_3'
        ] + [f"sensor_{i}" for i in range(1,22)]

    def read_file(self, file_path):
        df = pd.read_csv(file_path, sep='\s+', header=None)
        df.columns = self.column_names
        return df
    
    def initiate_data_ingestion(self):
        logging.info("Starting Data Ingestion 1")

        try:
            os.makedirs(self.config.processed_data_dir, exist_ok = True)

            train_path = os.path.join(self.config.raw_data_dir, 'train_FD001.txt')
            test_path = os.path.join(self.config.raw_data_dir, 'test_FD001.txt')
            rul_path = os.path.join(self.config.raw_data_dir, "RUL_FD001.txt")

            train_df = self.read_file(train_path)
            test_df = self.read_file(test_path)
            rul_df = pd.read_csv(rul_path, header = None)
            rul_df.columns = ['RUL']

            train_df.to_csv(self.config.train_data_path, index = False)
            test_df.to_csv(self.config.test_data_path, index = False)
            rul_df.to_csv(self.config.rul_data_path, index = False)

            logging.info('Data Ingestion Complete 1')

            return(
                self.config.train_data_path,
                self.config.test_data_path,
                self.config.rul_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)



if __name__ == '__main__':
    ingestion = DataIngestion()
    train, test, rul = ingestion.initiate_data_ingestion()
    print(f"Train: {train}\n Test: {test}\n RUL:{rul}")