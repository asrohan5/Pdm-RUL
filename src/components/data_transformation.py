import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.logger import logging
from src.exception import CustomException

class DataTransformationConfig:
    processed_train_path = os.path.join("D:/My Projects/Predictive Maintainability RUL/artifacts", "processed", "train.csv")
    processed_test_path = os.path.join("D:/My Projects/Predictive Maintainability RUL/artifacts", "processed", "test.csv")
    processed_rul_path = os.path.join("D:/My Projects/Predictive Maintainability RUL/artifacts", "processed", "rul.csv")
    preprocessor_path = os.path.join("D:/My Projects/Predictive Maintainability RUL/artifacts", "processed", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def add_rolling_features(self, df, sensors, window=5):

        for sensor in sensors:
            df[f"{sensor}_mean"] = df.groupby("unit_number")[sensor].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f"{sensor}_std"] = df.groupby("unit_number")[sensor].rolling(window=window, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
            df[f"{sensor}_min"] = df.groupby("unit_number")[sensor].rolling(window=window, min_periods=1).min().reset_index(level=0, drop=True)
            df[f"{sensor}_max"] = df.groupby("unit_number")[sensor].rolling(window=window, min_periods=1).max().reset_index(level=0, drop=True)
        return df

    def initiate_data_transformation(self, train_data_path, test_data_path, rul_data_path):
        logging.info("Reading raw train, test, and RUL data")

        try:

            train_df = pd.read_csv(train_data_path, sep=" ", header=None)
            test_df = pd.read_csv(test_data_path, sep=" ", header=None)
            rul_df = pd.read_csv(rul_data_path, header=None)

            train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
            test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)

            col_names = ["unit_number", "time_in_cycles", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"sensor{i}" for i in range(1, 22)]
            train_df.columns = col_names
            test_df.columns = col_names


            train_df["RUL"] = train_df.groupby("unit_number")["time_in_cycles"].transform("max") - train_df["time_in_cycles"]


            rul_df.columns = ["RUL"]
            rul_df["unit_number"] = rul_df.index + 1
            test_df = test_df.merge(rul_df, on="unit_number", how="left")


            sensor_cols = [col for col in train_df.columns if "sensor" in col]
            train_df = self.add_rolling_features(train_df, sensor_cols)
            test_df = self.add_rolling_features(test_df, sensor_cols)


            scaler = StandardScaler()
            feature_cols = [col for col in train_df.columns if col not in ["unit_number", "RUL"]]
            train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
            test_df[feature_cols] = scaler.transform(test_df[feature_cols])

            os.makedirs(os.path.dirname(self.config.processed_train_path), exist_ok=True)
            train_df.to_csv(self.config.processed_train_path, index=False)
            test_df.to_csv(self.config.processed_test_path, index=False)
            rul_df.to_csv(self.config.processed_rul_path, index=False)

            logging.info(f"Train data saved at: {self.config.processed_train_path}")
            logging.info(f"Test data saved at: {self.config.processed_test_path}")
            logging.info(f"RUL data saved at: {self.config.processed_rul_path}")

            return self.config.processed_train_path, self.config.processed_test_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataTransformation()
    obj.initiate_data_transformation(
        train_data_path="D:/My Projects/Predictive Maintainability RUL/artifacts/raw/train_FD001.txt",
        test_data_path="D:/My Projects/Predictive Maintainability RUL/artifacts/raw/test_FD001.txt",
        rul_data_path="D:/My Projects/Predictive Maintainability RUL/artifacts/raw/RUL_FD001.txt"
    )
