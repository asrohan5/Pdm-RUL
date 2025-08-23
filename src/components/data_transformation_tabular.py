# src/components/data_transformation_tabular.py

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler
from src.logger import logging
from src.exception import CustomException


@dataclass
class TabularTransformConfig:
    raw_dir: str = "D:/My Projects/Predictive Maintainability RUL/artifacts/raw"
    train_raw: str = "train_FD001.txt"
    test_raw: str = "test_FD001.txt"
    rul_raw: str = "RUL_FD001.txt"
    processed_dir: str = "D:/My Projects/Predictive Maintainability RUL/artifacts/processed_tabular"
    train_csv: str = "train_tabular.csv"
    test_csv: str = "test_tabular.csv"
    feature_meta_json: str = "tabular_features.json"
    scaler_path: str = "tabular_scaler.pkl"
    window_size: int = 80
    train_cycles_per_unit: int = 80
    include_ops: bool = False
    sensor_ids: List[int] = None


class DataTransformationTabular:
    def __init__(self, cfg: TabularTransformConfig = None):
        self.cfg = cfg or TabularTransformConfig()
        if self.cfg.sensor_ids is None:
            self.cfg.sensor_ids = [2, 3, 4, 7, 8, 9, 11, 12, 13, 15, 17, 20, 21]
        self.col_names = (
            ["unit_number", "time_in_cycles", "op_setting_1", "op_setting_2", "op_setting_3"]
            + [f"sensor{i}" for i in range(1, 22)]
        )
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

            return train_df, test_df, rul_df
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _compute_train_rul(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        max_cycle = df.groupby("unit_number")["time_in_cycles"].transform("max")
        df["RUL"] = max_cycle - df["time_in_cycles"]
        df["RUL"] = np.minimum(df["RUL"], 130)  # Cap RUL at 130 for training stability
        return df

    def _merge_test_rul_last_cycle(self, test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
        last = test_df.groupby("unit_number").last().reset_index()
        unique_units = sorted(last["unit_number"].unique().tolist())
        if len(unique_units) != len(rul_df):
            raise CustomException(f"RUL rows ({len(rul_df)}) do not match test units ({len(unique_units)}).", sys)
        aligned_rul = rul_df.copy()
        aligned_rul["unit_number"] = unique_units
        merged = last.merge(aligned_rul, on="unit_number", how="left")
        return merged

    @staticmethod
    def _win_sum(a: np.ndarray, window_size: int) -> np.ndarray:
        out = a.copy()
        if window_size < len(a):
            out[window_size:] = a[window_size:] - a[:-window_size]
        return out

    def _build_window_features_for_group(
        self,
        grp: pd.DataFrame,
        sensors: List[str],
        window_size: int
    ) -> pd.DataFrame:
        g = grp.sort_values("time_in_cycles").reset_index(drop=True).copy()
        n_rows = g.shape[0]

        t = g["time_in_cycles"].to_numpy(dtype=float)
        cycles_since_start = t - t[0]
        window_len = np.minimum(window_size, np.arange(1, n_rows + 1))

        add_cols: Dict[str, np.ndarray] = {
            "cycles_since_start": cycles_since_start,
            "window_len": window_len,
        }

        for s in sensors:
            s_vals = g[s].to_numpy(dtype=float)
            roll = pd.Series(s_vals).rolling(window=window_size, min_periods=1)

            mean_w = roll.mean().to_numpy(dtype=float)
            std_w = roll.std().fillna(0.0).to_numpy(dtype=float)
            min_w = roll.min().to_numpy(dtype=float)
            max_w = roll.max().to_numpy(dtype=float)

            cs_x = np.cumsum(t)
            cs_x2 = np.cumsum(t * t)
            cs_y = np.cumsum(s_vals)
            cs_xy = np.cumsum(t * s_vals)
            n = np.arange(1, n_rows + 1)
            w = np.minimum(window_size, n)

            sx = self._win_sum(cs_x, window_size)
            sx2 = self._win_sum(cs_x2, window_size)
            sy = self._win_sum(cs_y, window_size)
            sxy = self._win_sum(cs_xy, window_size)

            num = w * sxy - sx * sy
            den = w * sx2 - sx * sx
            slope_w = np.zeros_like(num, dtype=float)
            valid = den != 0
            slope_w[valid] = num[valid] / den[valid]

            delta_mean = s_vals - mean_w
            delta_min = s_vals - min_w
            eps = 1e-6
            ratio_mean = s_vals / (mean_w + eps)

            diff_1 = np.zeros_like(s_vals)
            diff_1[1:] = s_vals[1:] - s_vals[:-1]

            diff_5 = np.zeros_like(s_vals)
            for i in range(n_rows):
                if i >= 5:
                    diff_5[i] = np.mean(s_vals[i-4:i+1] - s_vals[i-5:i])
                elif i > 0:
                    diff_5[i] = np.mean(s_vals[1:i+1] - s_vals[0:i])

            add_cols.update({
                f"{s}_mean_w": mean_w,
                f"{s}_std_w": std_w,
                f"{s}_min_w": min_w,
                f"{s}_max_w": max_w,
                f"{s}_slope_w": slope_w,
                f"{s}_delta_mean": delta_mean,
                f"{s}_delta_min": delta_min,
                f"{s}_ratio_mean": ratio_mean,
                f"{s}_diff_1": diff_1,
                f"{s}_diff_5mean": diff_5,
            })

        new_block = pd.DataFrame(add_cols, index=g.index)
        g = pd.concat([g, new_block], axis=1)
        return g

    def _select_train_rows(self, df_unit: pd.DataFrame, k: int) -> pd.DataFrame:
        if df_unit.shape[0] <= k:
            return df_unit.copy()
        return df_unit.iloc[-k:, :].copy()

    def _assemble_feature_matrix(
        self,
        df: pd.DataFrame,
        sensors: List[str],
        include_ops: bool,
        is_train: bool
    ) -> Tuple[pd.DataFrame, List[str]]:
        per_sensor_suffixes = [
            "", "_mean_w", "_std_w", "_min_w", "_max_w",
            "_slope_w", "_delta_mean", "_delta_min", "_ratio_mean", "_diff_1", "_diff_5mean"
        ]
        feature_cols = [f"{s}{suf}" if suf else s for s in sensors for suf in per_sensor_suffixes]
        feature_cols += ["cycles_since_start", "window_len"]
        if include_ops:
            feature_cols += ["op_setting_1", "op_setting_2", "op_setting_3"]
        for c in feature_cols:
            if c not in df.columns:
                df[c] = 0.0
        cols_out = ["unit_number", "time_in_cycles"] + feature_cols
        if is_train:
            cols_out += ["RUL"]
        return df[cols_out].copy(), feature_cols

    def initiate(self) -> Tuple[str, str]:
        try:
            train_df, test_df, rul_df = self._read_raw()
            train_df = self._compute_train_rul(train_df)
            test_last = self._merge_test_rul_last_cycle(test_df, rul_df)
            sensor_cols = [f"sensor{i}" for i in self.cfg.sensor_ids]
            parts_train = []
            for _, grp in train_df.groupby("unit_number", sort=True):
                g = self._build_window_features_for_group(grp, sensor_cols, self.cfg.window_size)
                gk = self._select_train_rows(g, self.cfg.train_cycles_per_unit)
                parts_train.append(gk)
            train_feat = pd.concat(parts_train, axis=0, ignore_index=True)
            parts_test = []
            for _, grp in test_df.groupby("unit_number", sort=True):
                g = self._build_window_features_for_group(grp, sensor_cols, self.cfg.window_size)
                parts_test.append(g.iloc[[-1], :])
            test_feat_full = pd.concat(parts_test, axis=0, ignore_index=True)
            test_feat = test_feat_full.merge(test_last[["unit_number", "RUL"]], on="unit_number", how="left")
            train_final, feature_cols = self._assemble_feature_matrix(train_feat, sensor_cols, self.cfg.include_ops, True)
            test_final, _ = self._assemble_feature_matrix(test_feat, sensor_cols, self.cfg.include_ops, False)
            scaler = StandardScaler()
            X_train = train_final[feature_cols].to_numpy(dtype=float)
            X_test = test_final[feature_cols].to_numpy(dtype=float)
            train_final_scaled = train_final.copy()
            test_final_scaled = test_final.copy()
            train_final_scaled[feature_cols] = scaler.fit_transform(X_train)
            test_final_scaled[feature_cols] = scaler.transform(X_test)
            map_rul = dict(zip(test_last["unit_number"], test_last["RUL"]))
            rul_series = test_final_scaled["unit_number"].map(map_rul).rename("RUL")
            test_final_scaled = pd.concat([test_final_scaled, rul_series], axis=1)
            scaler_path = os.path.join(self.cfg.processed_dir, self.cfg.scaler_path)
            joblib.dump({"scaler": scaler, "feature_cols": feature_cols}, scaler_path)
            meta = {
                "window_size": self.cfg.window_size,
                "train_cycles_per_unit": self.cfg.train_cycles_per_unit,
                "include_ops": self.cfg.include_ops,
                "sensors": self.cfg.sensor_ids,
                "feature_cols": feature_cols,
            }
            meta_path = os.path.join(self.cfg.processed_dir, self.cfg.feature_meta_json)
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            train_out = os.path.join(self.cfg.processed_dir, self.cfg.train_csv)
            test_out = os.path.join(self.cfg.processed_dir, self.cfg.test_csv)
            train_final_scaled.to_csv(train_out, index=False)
            test_final_scaled.to_csv(test_out, index=False)
            logging.info(f"Saved tabular train CSV at: {train_out}")
            logging.info(f"Saved tabular test CSV at: {test_out}")
            return train_out, test_out
        except Exception as e:
            logging.error("Error in Tabular Data Transformation")
            raise CustomException(e, sys)


if __name__ == "__main__":
    transformer = DataTransformationTabular()
    transformer.initiate()
