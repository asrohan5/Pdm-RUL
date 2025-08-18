import pandas as pd
import nunpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os



def run_diagnostics():
    artifacts_dir = 'D:/My Projects/Predictive Maintainability RUL/artifacts'
    model_path = os.path.join(artifacts_dir, 'best_model.pkl')
    train_path = os.path.join(artifacts_dir, 'train.csv')
    test_path = os.path.join(artifacts_dir, 'test.csv')
    rul_path = os.path.join(artifacts_dir, 'RUL.csv')


    model = joblib.load(model_path)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    rul_df = pd.read_csv(rul_path)

    y_train = train_df['RUL'].values
    X_train = train_df.drop(columns = ['RUL'])

    X_test = test_df.copy()
    y_test = rul_df['RUL'].values

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    regression_metrics
    