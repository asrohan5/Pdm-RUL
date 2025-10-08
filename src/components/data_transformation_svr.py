import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.logger import logging
from src.exception import CustomException


@dataclass
class DataTransformationSVRConfig:

    raw_dir: str = "D:/My Projects/Predictive Maintainability RUL/artifacts/raw"
    train_raw: str = "train_FD001.txt"
    test_raw: str = "test_FD001.txt"
    rul_raw: str = "RUL_FD001.txt"


    processed_dir: str = "D:/My Projects/Predictive Maintainability RUL/artifacts/processed_svr"
    processed_train_csv: str = "train_svr.csv"
    processed_test_csv: str = "test_svr.csv"
    vhi_projection_path: str = "vhi_projection.pkl"      
    vhi_scaler_path: str = "vhi_scaler.pkl"              
    library_sequences_path: str = "library_train_sequences.pkl" 
    manifest_path: str = "svr_manifest.json"   


class DataTransformationSVR:
    def __init__(self, config: DataTransformationSVRConfig = None):
        self.cfg = config or DataTransformationSVRConfig()


        self.col_names = (
            ["unit_number", "time_in_cycles", "op_setting_1", "op_setting_2", "op_setting_3"]
            + [f"sensor{i}" for i in range(1, 22)]
        )

        self.selected_sensor_ids = [2, 3, 4, 7, 11, 12, 15]
        self.selected_sensors = [f"sensor{i}" for i in self.selected_sensor_ids]

 
        self.healthy_threshold = 300  
        self.failed_low = 0      
        self.failed_high = 4

        os.makedirs(self.cfg.processed_dir, exist_ok=True)

    def _read_raw(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        try:
            train_path = os.path.join(self.cfg.raw_dir, self.cfg.train_raw)
            test_path = os.path.join(self.cfg.raw_dir, self.cfg.test_raw)
            rul_path = os.path.join(self.cfg.raw_dir, self.cfg.rul_raw)

            logging.info(f"Reading raw train: {train_path}")
            train_df = pd.read_csv(train_path, sep=r"\s+", header=None)

            logging.info(f"Reading raw test: {test_path}")
            test_df = pd.read_csv(test_path, sep=r"\s+", header=None)

            logging.info(f"Reading raw RUL: {rul_path}")
            rul_df = pd.read_csv(rul_path, header=None, names=["RUL"])


            if train_df.shape[1] >= 28:
                train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
            if test_df.shape[1] >= 28:
                test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)


            train_df.columns = self.col_names
            test_df.columns = self.col_names

            missing_train = [c for c in self.selected_sensors if c not in train_df.columns]
            missing_test = [c for c in self.selected_sensors if c not in test_df.columns]
            if missing_train or missing_test:
                raise CustomException(
                    f"Missing expected sensors. Train missing: {missing_train}; Test missing: {missing_test}",
                    sys,
                )

            return train_df, test_df, rul_df

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _compute_train_rul(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        max_cycle = df.groupby("unit_number")["time_in_cycles"].transform("max")
        df["RUL"] = max_cycle - df["time_in_cycles"]
        return df

    @staticmethod
    def _compute_test_last_cycle_with_rul(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
        last = test_df.groupby("unit_number").last().reset_index()
        rul_df = rul_df.copy()
        rul_df["unit_number"] = np.arange(1, len(rul_df) + 1)
        merged = last.merge(rul_df, on="unit_number", how="left")
        return merged

    def _validate_and_clean_features(self, df: pd.DataFrame) -> pd.DataFrame:

        vals = df[self.selected_sensors].to_numpy(dtype=float)
        mask = np.isfinite(vals).all(axis=1)
        cleaned = df.loc[mask].copy()
        if cleaned.empty:
            raise CustomException("All rows invalid after filtering for finite sensor values.", sys)
        return cleaned

    def _prepare_Q_sets(self, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        q0 = train_df[(train_df["RUL"] >= self.failed_low) & (train_df["RUL"] <= self.failed_high)]
        q1 = train_df[train_df["RUL"] > self.healthy_threshold]

        q0 = self._validate_and_clean_features(q0)
        q1 = self._validate_and_clean_features(q1)

        if q0.empty or q1.empty:
            raise CustomException(
                f"Q0/Q1 empty after cleaning: q0={len(q0)}, q1={len(q1)}. "
                f"Consider adjusting thresholds.", sys
            )
        return q0, q1

    def _learn_vhi_projection(self, q0: pd.DataFrame, q1: pd.DataFrame) -> Dict[str, str]:
        
        try:

            X0 = q0[self.selected_sensors].to_numpy(dtype=float)
            X1 = q1[self.selected_sensors].to_numpy(dtype=float)


            if X0.ndim != 2 or X1.ndim != 2:
                raise CustomException(f"Feature arrays must be 2D. Got X0.ndim={X0.ndim}, X1.ndim={X1.ndim}", sys)
            n0 = X0.shape[0]
            n1 = X1.shape[0]
            if n0 == 0 or n1 == 0:
                raise CustomException(f"Q0/Q1 have zero rows: q0={n0}, q1={n1}", sys)


            y0 = np.zeros((n0,), dtype=int)
            y1 = np.ones((n1,), dtype=int)


            X = np.vstack([X0, X1]).astype(float)
            y = np.concatenate([y0, y1], axis=0).astype(int)

            if X.shape[0] < 2 or len(np.unique(y)) < 2:
                raise CustomException("Insufficient class diversity to learn VHI projection.", sys)

            lda = LinearDiscriminantAnalysis(n_components=1)
            lda.fit(X, y)

            vhi_proj_path = os.path.join(self.cfg.processed_dir, self.cfg.vhi_projection_path)
            joblib.dump(lda, vhi_proj_path)
            logging.info(f"Saved VHI projection model at: {vhi_proj_path}")

            return {"type": "lda_model", "path": vhi_proj_path}

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _minmax_fit(v: np.ndarray) -> Dict[str, float]:
        vmin = float(np.min(v))
        vmax = float(np.max(v))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            raise CustomException("Non-finite VHI encountered during scaling fit.", sys)
        if np.isclose(vmax, vmin):
            vmax = vmin + 1e-6
        return {"vmin": vmin, "vmax": vmax}

    @staticmethod
    def _minmax_transform(v: np.ndarray, scaler: Dict[str, float]) -> np.ndarray:
        vmin, vmax = scaler["vmin"], scaler["vmax"]
        return (v - vmin) / (vmax - vmin)

    def _compute_vhi_series(self, df: pd.DataFrame, proj_info: Dict[str, str]) -> np.ndarray:
        lda = joblib.load(proj_info["path"])
        X = df[self.selected_sensors].to_numpy(dtype=float)
        vhi_raw = lda.transform(X).ravel()
        return vhi_raw

    @staticmethod
    def _compute_tau(df: pd.DataFrame) -> np.ndarray:
        max_cycle = df.groupby("unit_number")["time_in_cycles"].transform("max")
        tau = df["time_in_cycles"].to_numpy(dtype=float) - max_cycle.to_numpy(dtype=float)
        return tau

    def initiate_data_transformation(self) -> Tuple[str, str]:

        try:

            train_df, test_df, rul_df = self._read_raw()

            train_df = self._compute_train_rul(train_df)
            test_last = self._compute_test_last_cycle_with_rul(test_df, rul_df)

  
            train_df = self._validate_and_clean_features(train_df)
            test_last = self._validate_and_clean_features(test_last)


            q0, q1 = self._prepare_Q_sets(train_df)
            proj_info = self._learn_vhi_projection(q0, q1)
            vhi_train_raw = self._compute_vhi_series(train_df, proj_info)
            vhi_test_raw = self._compute_vhi_series(test_last, proj_info)


            vhi_scaler = self._minmax_fit(vhi_train_raw)
            vhi_train = self._minmax_transform(vhi_train_raw, vhi_scaler)
            vhi_test = self._minmax_transform(vhi_test_raw, vhi_scaler)

            vhi_scaler_path = os.path.join(self.cfg.processed_dir, self.cfg.vhi_scaler_path)
            joblib.dump(vhi_scaler, vhi_scaler_path)
            logging.info(f"Saved VHI scaler at: {vhi_scaler_path}")


            tau_train = self._compute_tau(train_df)
            tau_test = self._compute_tau(test_last)

            processed_train = pd.DataFrame(
                {
                    "unit_number": train_df["unit_number"].to_numpy(dtype=int),
                    "time_in_cycles": train_df["time_in_cycles"].to_numpy(dtype=int),
                    "tau": tau_train.astype(float),
                    "VHI": vhi_train.astype(float),
                    "RUL": train_df["RUL"].to_numpy(dtype=int),
                }
            )

            processed_test = pd.DataFrame(
                {
                    "unit_number": test_last["unit_number"].to_numpy(dtype=int),
                    "time_in_cycles": test_last["time_in_cycles"].to_numpy(dtype=int),
                    "tau": tau_test.astype(float),
                    "VHI": vhi_test.astype(float),
                    "RUL": test_last["RUL"].to_numpy(dtype=float),
                }
            )

            train_out = os.path.join(self.cfg.processed_dir, self.cfg.processed_train_csv)
            test_out = os.path.join(self.cfg.processed_dir, self.cfg.processed_test_csv)
            processed_train.to_csv(train_out, index=False)
            processed_test.to_csv(test_out, index=False)
            logging.info(f"Saved processed train CSV at: {train_out}")
            logging.info(f"Saved processed test CSV at: {test_out}")


            library: Dict[int, Dict[str, np.ndarray]] = {}
            for uid, grp in processed_train.groupby("unit_number", sort=True):
                library[int(uid)] = {
                    "tau": grp["tau"].to_numpy(dtype=float),
                    "VHI": grp["VHI"].to_numpy(dtype=float),
                }
            lib_out = os.path.join(self.cfg.processed_dir, self.cfg.library_sequences_path)
            joblib.dump(library, lib_out)
            logging.info(f"Saved training reference library at: {lib_out}")


            manifest = {
                "selected_sensors": self.selected_sensors,
                "healthy_threshold": self.healthy_threshold,
                "failed_low": self.failed_low,
                "failed_high": self.failed_high,
                "artifacts": {
                    "processed_train_csv": train_out,
                    "processed_test_csv": test_out,
                    "vhi_projection_model": os.path.join(self.cfg.processed_dir, self.cfg.vhi_projection_path),
                    "vhi_scaler": vhi_scaler_path,
                    "library_sequences": lib_out,
                },
            }
            manifest_out = os.path.join(self.cfg.processed_dir, self.cfg.manifest_path)
            with open(manifest_out, "w") as f:
                json.dump(manifest, f, indent=2)
            logging.info(f"Saved SVR manifest at: {manifest_out}")

            return train_out, test_out

        except Exception as e:
            logging.error("Error in SVR Data Transformation")
            raise CustomException(e, sys)


if __name__ == "__main__":
    transformer = DataTransformationSVR()
    transformer.initiate_data_transformation()
