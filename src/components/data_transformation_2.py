# src/components/data_transformation_2.py
import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object 

from dataclasses import dataclass
import os
class DataTransformationConfig:
    processed_data_dir = os.path.join("D:/My Projects/Predictive Maintainability RUL/artifacts", "processed")
    processed_train_path = os.path.join(processed_data_dir, "train.csv")
    processed_test_path  = os.path.join(processed_data_dir, "test.csv")
    processed_rul_path   = os.path.join(processed_data_dir, "rul.csv")
    preprocessor_path    = os.path.join(processed_data_dir, "preprocessor.pkl")

class DataTransformation2:
    def __init__(self,
                 variance_threshold: float = 1e-4,
                 lags=(1, 5),
                 roll_windows=(5, 20),
                 cap_rul: bool = True,
                 max_rul_clip: int = 125):
        self.config = DataTransformationConfig()
        self.scaler = StandardScaler()
        self.variance_threshold = variance_threshold
        self.lags = tuple(lags)
        self.roll_windows = tuple(roll_windows)
        self.cap_rul = cap_rul
        self.max_rul_clip = max_rul_clip

        self.manual_low_var = {"sensor1", "sensor5", "sensor10", "sensor16", "sensor18", "sensor19"}


    @staticmethod
    def _normalized_cycle(df):
        mx = df.groupby("unit_number")["time_in_cycles"].transform("max")
        return (df["time_in_cycles"] / mx).rename("cycle_norm")

    @staticmethod
    def _zscore_block(df, cols):
        g = df.groupby("unit_number")
        m = g[cols].transform("mean")
        s = g[cols].transform("std").replace(0, 1)
        z = (df[cols] - m) / s
        z.columns = [f"{c}_z" for c in cols]
        return z

    def add_stat_features(df, window=30):
        sensor_cols = [col for col in df.columns if col.startswith("sensor")]
        for col in sensor_cols:
            df[f"{col}_mean"] = df[col].rolling(window=window, min_periods=1).mean()
            df[f"{col}_std"]  = df[col].rolling(window=window, min_periods=1).std().fillna(0)
            df[f"{col}_min"]  = df[col].rolling(window=window, min_periods=1).min()
            df[f"{col}_max"]  = df[col].rolling(window=window, min_periods=1).max()
        return df


    @staticmethod
    def _rolling_block(df, cols, window):
        g = df.groupby("unit_number")[cols]
        roll = g.rolling(window, min_periods=1)

        m  = roll.mean().reset_index(level=0, drop=True)
        sd = roll.std().reset_index(level=0, drop=True).fillna(0)
        mn = roll.min().reset_index(level=0, drop=True)
        mx = roll.max().reset_index(level=0, drop=True)

        m.columns  = [f"{c}_rmean{window}" for c in cols]
        sd.columns = [f"{c}_rstd{window}"  for c in cols]
        mn.columns = [f"{c}_rmin{window}"  for c in cols]
        mx.columns = [f"{c}_rmax{window}"  for c in cols]

        return pd.concat([m, sd, mn, mx], axis=1)

    @staticmethod
    def _slope_block(df, cols, window):

        diffs = df.groupby("unit_number")[cols].diff()
        sl = diffs.rolling(window, min_periods=1).mean().fillna(0)
        sl.columns = [f"{c}_slope{window}" for c in cols]
        return sl

    @staticmethod
    def _lag_block(df, cols, lags):
        out = []

        first_vals = df.groupby("unit_number")[cols].transform("first")
        for lag in lags:
            lagged = df.groupby("unit_number")[cols].shift(lag)
            lagged = lagged.fillna(first_vals)
            lagged.columns = [f"{c}_lag{lag}" for c in cols]
            out.append(lagged)
        return pd.concat(out, axis=1) if out else pd.DataFrame(index=df.index)

    def _drop_low_variance(self, train_df, test_df, cols):

        low_var = set([c for c in cols if train_df[c].var() < self.variance_threshold])
        low_var |= (self.manual_low_var & set(cols))
        if low_var:
            logging.info(f"Dropping low-variance sensors: {sorted(low_var)}")
            train_df = train_df.drop(columns=sorted(low_var))
            test_df  = test_df.drop(columns=sorted(low_var))
        return train_df, test_df

    def initiate_data_transformation(self, train_data_path, test_data_path, rul_data_path):
        try:
            logging.info("Reading raw train, test, and RUL data")

            train_raw = pd.read_csv(train_data_path, sep=r"\s+", header=None)
            test_raw  = pd.read_csv(test_data_path,  sep=r"\s+", header=None)
            rul_df    = pd.read_csv(rul_data_path,   sep=r"\s+", header=None)


            cols_to_drop = ["setting3"]  
            train_raw.drop(columns=[c for c in cols_to_drop if c in train_raw.columns], inplace=True)
            test_raw.drop(columns=[c for c in cols_to_drop if c in test_raw.columns], inplace=True)


            cols = ["unit_number", "time_in_cycles",
                    "op_setting_1", "op_setting_2", "op_setting_3"] + [f"sensor{i}" for i in range(1, 22)]
            train_raw.columns = cols
            test_raw.columns  = cols


            train_raw["RUL"] = train_raw.groupby("unit_number")["time_in_cycles"].transform("max") - train_raw["time_in_cycles"]

 
            rul_df.columns = ["RUL"]
            rul_df["unit_number"] = rul_df.index + 1
            test_raw = test_raw.merge(rul_df, on="unit_number", how="left")


            train_raw.sort_values(["unit_number", "time_in_cycles"], inplace=True)
            test_raw.sort_values(["unit_number", "time_in_cycles"], inplace=True)
            train_raw.reset_index(drop=True, inplace=True)
            test_raw.reset_index(drop=True, inplace=True)


            sensor_cols = [c for c in cols if c.startswith("sensor")]


            train_raw, test_raw = self._drop_low_variance(train_raw, test_raw, sensor_cols)
            sensor_cols = [c for c in train_raw.columns if c.startswith("sensor")]

            def build_features(df):
                blocks = []

                blocks.append(self._normalized_cycle(df).to_frame())

                blocks.append(self._zscore_block(df, sensor_cols))
 
                for w in self.roll_windows:
                    blocks.append(self._rolling_block(df, sensor_cols, window=w))
                    blocks.append(self._slope_block(df, sensor_cols, window=w))

                blocks.append(self._lag_block(df, sensor_cols, self.lags))
                return pd.concat(blocks, axis=1)

            logging.info("Building engineered features (train)...")
            train_feats = build_features(train_raw)
            logging.info("Building engineered features (test)...")
            test_feats  = build_features(test_raw)


            base_cols = ["unit_number", "time_in_cycles", "op_setting_1", "op_setting_2", "op_setting_3"] + sensor_cols + ["RUL"]
            train_df = pd.concat([train_raw[base_cols].reset_index(drop=True), train_feats.reset_index(drop=True)], axis=1)
            test_df  = pd.concat([test_raw[base_cols].reset_index(drop=True),  test_feats.reset_index(drop=True)],  axis=1)


            if self.cap_rul:
                train_df["RUL"] = np.minimum(train_df["RUL"], self.max_rul_clip)
                test_df["RUL"]  = np.minimum(test_df["RUL"],  self.max_rul_clip)
            


            feature_cols = [c for c in train_df.columns if c not in {"unit_number", "RUL"}]
  
            train_df[feature_cols] = train_df[feature_cols].astype(float)
            test_df[feature_cols] = test_df[feature_cols].astype(float)

            logging.info(f"Scaling {len(feature_cols)} feature columns")
            train_df.loc[:, feature_cols] = self.scaler.fit_transform(train_df[feature_cols])
            test_df.loc[:,  feature_cols] = self.scaler.transform(test_df[feature_cols])

            train_df = self.add_stat_features(train_df)
            test_df  = self.add_stat_features(test_df)



            processed_path = os.path.join('D:/My Projects/Predictive Maintainability RUL/artifacts', 'processed')
            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            train_df.to_csv(self.config.processed_train_path, index=False)
            test_df.to_csv(self.config.processed_test_path,  index=False)
            rul_df.to_csv(self.config.processed_rul_path,    index=False)

            os.makedirs(self.config.processed_data_dir, exist_ok=True)
            scaler_path = os.path.join(self.config.processed_data_dir, "scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            logging.info(f"Scaler saved successfully at {scaler_path}")



            try:
                save_object(self.config.preprocessor_path, self.scaler)
            except Exception:
               
                pass

            logging.info(f"Processed train saved: {self.config.processed_train_path}")
            logging.info(f"Processed test  saved: {self.config.processed_test_path}")
            logging.info(f"RUL file saved      : {self.config.processed_rul_path}")
            return self.config.processed_train_path, self.config.processed_test_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        obj = DataTransformation2(
            variance_threshold=1e-4,
            lags=(1, 5),
            roll_windows=(5, 20),
            cap_rul=True,         
            max_rul_clip=125
        )
        obj.initiate_data_transformation(
            train_data_path="D:/My Projects/Predictive Maintainability RUL/artifacts/raw/train_FD001.txt",
            test_data_path="D:/My Projects/Predictive Maintainability RUL/artifacts/raw/test_FD001.txt",
            rul_data_path="D:/My Projects/Predictive Maintainability RUL/artifacts/raw/RUL_FD001.txt"
        )
    except Exception as ex:
        raise CustomException(ex, sys)
