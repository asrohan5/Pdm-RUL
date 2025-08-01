import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline
from src.components.data_ingestion import read_file

if __name__ ='__main__':
    test_file_path = 'D:/My Projects/Predictive Maintainability RUL/artifacts/raw/test_FD001.txt'
    df = read_file('test_file_path')

    pipeline = PredictPipeline()
    result_df = pipeline.predict(df)

    print('Predicted RUL for each unit:')
    print(result_df.to_string(index=False))
    