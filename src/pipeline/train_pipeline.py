# src/pipeline/train_pipeline.py

import os
import sys
import logging
from src.components.data_transformation_tabular import DataTransformationTabular, TabularTransformConfig
from src.components.model_tabular import ModelTabular, TabularModelConfig
from src.components.model_evaluation import evaluate_predictions
from src.exception import CustomException

def main():
    try:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

        # STEP 1: Data Ingestion
        # For many projects, ingestion is just copying raw => staged;
        # Add advanced ingestion if using/moving data from multiple sources.
        raw_dir = "D:/My Projects/Predictive Maintainability RUL/artifacts/raw"
        processed_dir = "D:/My Projects/Predictive Maintainability RUL/artifacts/processed_tabular"
        logging.info("Starting Training Pipeline")
        logging.info("Step 1: Data Ingestion")
        if not os.path.isdir(raw_dir):
            raise CustomException(f"Raw data directory not found: {raw_dir}", sys)
        logging.info(f"Raw data found at: {raw_dir}")

        # STEP 2: Data Transformation (Feature Engineering)
        logging.info("Step 2: Data Transformation")
        dt_cfg = TabularTransformConfig(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            include_ops=False  # Or True, depending on your model/experiment
        )
        transformer = DataTransformationTabular(dt_cfg)
        train_csv, test_csv = transformer.initiate()
        logging.info(f"Data transformation complete: {train_csv}, {test_csv}")

        # STEP 3: Model Training
        logging.info("Step 3: Model Training (with GridSearchCV)")
        mdl_cfg = TabularModelConfig(
            processed_dir=processed_dir,
        )
        model_trainer = ModelTabular(mdl_cfg)
        preds_csv, metrics_json = model_trainer.train_and_predict()
        logging.info(f"Training complete. Predictions: {preds_csv}, Metrics: {metrics_json}")

        # STEP 4: Model Evaluation (Optional if already done, but for demonstration)
        logging.info("Step 4: Model Evaluation")
        evaluate_predictions(preds_csv, metrics_json)
        logging.info("Model evaluation stored.")

        logging.info("Training pipeline completed successfully.")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
