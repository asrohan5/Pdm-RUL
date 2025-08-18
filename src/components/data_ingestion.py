import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException

class DataIngestion:
    def __init__(self,
                 raw_data_dir="D:/My Projects/Predictive Maintainability RUL/artifacts/raw",
                 processed_data_dir="D:/My Projects/Predictive Maintainability RUL/artifacts/processed"):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        os.makedirs(self.processed_data_dir, exist_ok=True)

        # C-MAPSS FD001 column names
        self.column_names = (
            ['unit_number', 'time_in_cycles',
             'op_setting_1', 'op_setting_2', 'op_setting_3'] +
            [f'sensor{i}' for i in range(1, 22)]
        )

    def initiate_data_ingestion(self, train_path: str, test_path: str, rul_path: str):
        logging.info("Starting Data Ingestion")

        try:
            # Read raw train & test with correct headers
            logging.info("Reading train and test datasets")
            train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=self.column_names)
            test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=self.column_names)

            # Read RUL file (single column, no header)
            rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=['RUL'])

            # Save processed versions
            train_file = os.path.join(self.processed_data_dir, "train.csv")
            test_file = os.path.join(self.processed_data_dir, "test.csv")
            rul_file = os.path.join(self.processed_data_dir, "rul.csv")

            train_df.to_csv(train_file, index=False)
            test_df.to_csv(test_file, index=False)
            rul_df.to_csv(rul_file, index=False)

            logging.info(f"Train data saved at: {train_file}")
            logging.info(f"Test data saved at: {test_file}")
            logging.info(f"RUL data saved at: {rul_file}")

            return train_file, test_file, rul_file

        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_path = "D:/My Projects/Predictive Maintainability RUL/artifacts/raw/train_FD001.txt"
    test_path = "D:/My Projects/Predictive Maintainability RUL/artifacts/raw/test_FD001.txt"
    rul_path = "D:/My Projects/Predictive Maintainability RUL/artifacts/raw/RUL_FD001.txt"

    obj.initiate_data_ingestion(train_path, test_path, rul_path)
