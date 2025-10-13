import os
import sys
import json
import math
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.logger import logging
from src.exception import CustomException


@dataclass
class SVRConfig:
    
    processed_dir: str = "D:/My Projects/Predictive Maintainability RUL/artifacts/processed_svr"
    train_csv: str = "train_svr.csv"
    test_csv: str = "test_svr.csv"
    library_sequences: str = "library_train_sequences.pkl"

 
    model_path: str = "svr_model.pkl"
    predictions_csv: str = "svr_predictions.csv"
    metrics_json: str = "svr_metrics.json"

   
    C: float = 10.0
    epsilon: float = 0.1
    gamma: float = 0.01 


    window_size: int = 50 
    top_k: int = 10     
    min_overlap: int = 15 

    random_state: int = 42


class ModelSVR:
    def __init__(self, cfg: SVRConfig = None):
        self.cfg = cfg or SVRConfig()
        # Resolve full paths
        self.train_path = os.path.join(self.cfg.processed_dir, self.cfg.train_csv)
        self.test_path = os.path.join(self.cfg.processed_dir, self.cfg.test_csv)
        self.library_path = os.path.join(self.cfg.processed_dir, self.cfg.library_sequences)
        self.model_out = os.path.join(self.cfg.processed_dir, self.cfg.model_path)
        self.preds_out = os.path.join(self.cfg.processed_dir, self.cfg.predictions_csv)
        self.metrics_out = os.path.join(self.cfg.processed_dir, self.cfg.metrics_json)

        os.makedirs(self.cfg.processed_dir, exist_ok=True)

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, Dict[str, np.ndarray]]]:
        try:
            logging.info(f"Loading processed train CSV: {self.train_path}")
            train_df = pd.read_csv(self.train_path)

            logging.info(f"Loading processed test CSV: {self.test_path}")
            test_df = pd.read_csv(self.test_path)

            logging.info(f"Loading reference library: {self.library_path}")
            library = joblib.load(self.library_path)

            for col in ["tau", "VHI"]:
                if col not in train_df.columns:
                    raise CustomException(f"Column '{col}' missing in train CSV.", sys)
                if col not in test_df.columns:
                    raise CustomException(f"Column '{col}' missing in test CSV.", sys)

            for df, name in [(train_df, "train"), (test_df, "test")]:
                for col in ["tau", "VHI"]:
                    vals = df[col].to_numpy(dtype=float)
                    if not np.isfinite(vals).any():
                        raise CustomException(f"No finite values in {name} '{col}' column.", sys)

            return train_df, test_df, library
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _prepare_curve_training_data(train_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
       
        tau = train_df["tau"].to_numpy(dtype=float)
        vhi = train_df["VHI"].to_numpy(dtype=float)
       
        mask = np.isfinite(tau) & np.isfinite(vhi)
        tau = tau[mask]
        vhi = vhi[mask]
        if tau.size == 0:
            raise CustomException("No valid (tau, VHI) pairs for SVR training after filtering.", sys)
        X = tau.reshape(-1, 1)
        y = vhi.reshape(-1,)
        return X, y

    def _fit_svr(self, X: np.ndarray, y: np.ndarray) -> SVR:
        
        try:
            svr = SVR(kernel="rbf", C=self.cfg.C, epsilon=self.cfg.epsilon, gamma=self.cfg.gamma)
            svr.fit(X, y)
            joblib.dump(svr, self.model_out)
            logging.info(f"Saved SVR model at: {self.model_out}")
            return svr
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _ssd(a: np.ndarray, b: np.ndarray) -> float:
        
        m = min(a.shape[0], b.shape)
        if m == 0:
            return np.inf
        diff = a[:m] - b[:m]
      
        mask = np.isfinite(diff)
        if not np.any(mask):
            return np.inf
        d = diff[mask]
        return float(np.dot(d, d))

    def _align_and_estimate_rul_from_reference(
        self,
        test_tau: np.ndarray,
        test_vhi: np.ndarray,
        ref_tau: np.ndarray,
        ref_vhi: np.ndarray
    ) -> Tuple[float, float]:
     
        W = self.cfg.window_size
        min_overlap = self.cfg.min_overlap

        if test_vhi.size == 0 or ref_vhi.size == 0:
            return np.nan, np.inf

      
        w = min(W, test_vhi.shape[0])
        if w < min_overlap:
            return np.nan, np.inf

        test_seg = test_vhi[-w:]
 
        if not np.isfinite(test_seg).any():
            return np.nan, np.inf

     
        best_ssd = np.inf
        best_ref_end_idx: Optional[int] = None

        ref_len = ref_vhi.shape
        if ref_len < min_overlap:
            return np.nan, np.inf

        for end in range(min_overlap, ref_len + 1):
            start = end - w
            if start < 0:
                continue
            ref_seg = ref_vhi[start:end]
        
            m = min(test_seg.shape, ref_seg.shape)
            if m < min_overlap:
                continue


            ssd = self._ssd(test_seg[-m:], ref_seg[-m:])
            if ssd < best_ssd:
                best_ssd = ssd
                best_ref_end_idx = end

        if best_ref_end_idx is None or not np.isfinite(best_ssd):
            return np.nan, np.inf

        aligned_tau = ref_tau[best_ref_end_idx - 1]
        if not np.isfinite(aligned_tau):
            return np.nan, np.inf

        rul_est = float(-aligned_tau)
        if rul_est < 0:
           
            rul_est = 0.0

        return rul_est, best_ssd

    def _predict_test_unit_rul(
        self,
        test_unit_df: pd.DataFrame,
        library: Dict[int, Dict[str, np.ndarray]]
    ) -> float:
    
        test_tau = test_unit_df["tau"].to_numpy(dtype=float)
        test_vhi = test_unit_df["VHI"].to_numpy(dtype=float)

   
        mask = np.isfinite(test_tau) & np.isfinite(test_vhi)
        test_tau = test_tau[mask]
        test_vhi = test_vhi[mask]

        if test_vhi.size == 0:
            return np.nan

     
        estimates: List[Tuple[float, float]] = [] 
        for uid, seq in library.items():
            ref_tau = np.asarray(seq["tau"], dtype=float)
            ref_vhi = np.asarray(seq["VHI"], dtype=float)


            mask_ref = np.isfinite(ref_tau) & np.isfinite(ref_vhi)
            ref_tau = ref_tau[mask_ref]
            ref_vhi = ref_vhi[mask_ref]
            if ref_vhi.size == 0:
                continue

            rul_est, ssd = self._align_and_estimate_rul_from_reference(test_tau, test_vhi, ref_tau, ref_vhi)
            if np.isfinite(rul_est) and np.isfinite(ssd) and ssd > 0:
                estimates.append((rul_est, ssd))

        if len(estimates) == 0:
            return np.nan

        estimates.sort(key=lambda x: x[1])
        top = estimates[: self.cfg.top_k]

    
        weights = np.array([1.0 / e[1] for e in top], dtype=float)
        ruls = np.array([e for e in top], dtype=float)
        wsum = float(np.sum(weights))
        if wsum <= 0:
            return np.nan
        return float(np.dot(weights, ruls) / wsum)

    def train_and_predict(self) -> Tuple[str, str]:
      
        try:
         
            train_df, test_df, library = self._load_data()

            
            X_train, y_train = self._prepare_curve_training_data(train_df)
            _ = self._fit_svr(X_train, y_train)

   
            preds = []
            for unit_id, grp in test_df.groupby("unit_number", sort=True):
                rul_pred = self._predict_test_unit_rul(grp, library)
         
                true_rul = grp["RUL"].dropna().max()
                preds.append(
                    {
                        "unit_number": int(unit_id),
                        "RUL_true": float(true_rul) if pd.notna(true_rul) else np.nan,
                        "RUL_pred": float(rul_pred) if np.isfinite(rul_pred) else np.nan,
                    }
                )

            preds_df = pd.DataFrame(preds).sort_values("unit_number")
            preds_df.to_csv(self.preds_out, index=False)
            logging.info(f"Saved SVR predictions at: {self.preds_out}")

            eval_df = preds_df.dropna(subset=["RUL_true", "RUL_pred"]).copy()
            if not eval_df.empty:
                y_true = eval_df["RUL_true"].to_numpy(dtype=float)
                y_pred = eval_df["RUL_pred"].to_numpy(dtype=float)

                rmse = math.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)


                with np.errstate(divide="ignore", invalid="ignore"):
                    rel_err = (y_pred - y_true) / np.where(y_true == 0, np.nan, y_true)
                mp = float(np.nanmean(rel_err))

                metrics = {"rmse": rmse, "mae": mae, "mp": mp, "n_eval_units": int(eval_df.shape[0])}
            else:
                metrics = {"rmse": None, "mae": None, "mp": None, "n_eval_units": 0}

            with open(self.metrics_out, "w") as f:
                json.dump(metrics, f, indent=2)
            logging.info(f"Saved SVR metrics at: {self.metrics_out}")

            return self.preds_out, self.metrics_out

        except Exception as e:
            logging.error("Error in SVR training/prediction")
            raise CustomException(e, sys)


if __name__ == "__main__":
    runner = ModelSVR()
    runner.train_and_predict()
