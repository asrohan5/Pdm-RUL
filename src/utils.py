import os
import sys
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

import logging
from typing import Any, Dict, List, Tuple
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise Exception(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv  = 3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)




def perform_xgb_grid_search(
    X_train,
    y_train,
    param_grid: Dict[str, List[Any]],
    cv: int = 3,
    scoring: str = "neg_root_mean_squared_error",
    n_jobs: int = -1,
    verbose: int = 2,
    random_state: int = 42,
) -> Tuple[XGBRegressor, Dict[str, Any]]:
    """
    Perform grid search with cross-validation for XGBRegressor to find best hyperparameters.

    Returns:
        best_model: trained XGBRegressor estimator with best params.
        best_params: dict of best hyperparameters found.

    Raises:
        ValueError if training data is invalid.
    """
    xgb = XGBRegressor(
        objective="reg:squarederror",
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=random_state,
        n_jobs=n_jobs,
        eval_metric="rmse",
    )

    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
    )
    grid.fit(X_train, y_train)

    logging.info(f"Grid search best params: {grid.best_params_}")
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    return best_model, best_params
