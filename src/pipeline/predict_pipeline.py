import os
import sys
import logging
import pandas as pd
import joblib
from src.components.data_transformation_tabular import DataTransformationTabular, TabularTransformConfig
from src.exception import CustomException

def run_predict_pipeline(
    raw_test_path: str,
    rul_test_path: str,
    processed_dir: str,
    model_path: str,
    output_csv: str
):
    try:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        logging.info("Starting Predict Pipeline")

    
        if not os.path.exists(raw_test_path):
            raise CustomException(f"Test file not found: {raw_test_path}", sys)
        if not os.path.exists(rul_test_path):
            logging.warning(f"RUL file for test set not found: {rul_test_path} (predictions will have no ground-truth labels if required)")
        if not os.path.isdir(processed_dir):
            raise CustomException(f"Processed dir missing: {processed_dir}", sys)
        if not os.path.exists(model_path):
            raise CustomException(f"Model artifact not found: {model_path}", sys)

        
        config = TabularTransformConfig(
            raw_dir=os.path.dirname(raw_test_path),
            test_raw=os.path.basename(raw_test_path),
            rul_raw=os.path.basename(rul_test_path),
            processed_dir=processed_dir,
            include_ops=False  
        )
        transformer = DataTransformationTabular(cfg=config)
        _, test_out_path = transformer.initiate()
        logging.info(f"Test set transformed: {test_out_path}")

   
        scaler_obj = joblib.load(os.path.join(processed_dir, "tabular_scaler.pkl"))
        feature_cols = scaler_obj["feature_cols"]
        model = joblib.load(model_path)
        test_df = pd.read_csv(test_out_path)

        X_test = test_df[feature_cols].to_numpy(dtype=float)
        y_pred = model.predict(X_test)

        output_df = test_df[["unit_number", "time_in_cycles"]].copy()
        output_df["RUL_pred"] = y_pred
      
        if "RUL" in test_df.columns:
            output_df["RUL_true"] = test_df["RUL"]

        output_df.to_csv(output_csv, index=False)
        logging.info(f"Predictions saved to: {output_csv}")
        print(f"Prediction pipeline completed successfully. Output file: {output_csv}")

        return output_df

    except Exception as e:
        logging.error(f"Prediction pipeline failed: {str(e)}")
        raise CustomException(e, sys)

if __name__ == "__main__":
 
    base_dir = "D:/My Projects/Predictive Maintainability RUL/artifacts/processed_tabular"
    raw_test_path = "D:/My Projects/Predictive Maintainability RUL/artifacts/raw/test_FD001.txt"
    rul_test_path = "D:/My Projects/Predictive Maintainability RUL/artifacts/raw/RUL_FD001.txt"
    model_path = os.path.join(base_dir, "xgb_tabular_model.pkl")
    output_csv = os.path.join(base_dir, "xgb_tabular_predictions_new.csv")
    run_predict_pipeline(raw_test_path, rul_test_path, base_dir, model_path, output_csv)
