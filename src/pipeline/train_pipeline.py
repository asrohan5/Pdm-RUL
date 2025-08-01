from src.components.dat_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.logger import logging

if __name__ = '__main__':
    try:
        logging.info('Training Pipeline Started')

        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()


        transformation = DataTransformation()
        train_array, test_array,_ = transformation.intitate_data_transformation(train_path, test_path)


        trainer = ModelTrainer()
        r2 = trainer.initiate_model_trainer(train_array, test_array)
        logging.info(f'Training complete. Final R2 score: {r2:.4f}')

        evaluator = ModelEvaluation(
            model_path = 'D:/My Projects/Predictive Maintainability RUL/artifacts/best_model.pkl'
            preprocessor_path = 'D:/My Projects/Predictive Maintainability RUL/artifacts/preprocessor.pkl'
        )

        evaluator.evaluate()

        logging.info('Training Pipeline Completed')

    except Exception as e:
        raise CustomeException(e,sys)
