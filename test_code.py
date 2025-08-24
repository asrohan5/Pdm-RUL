from src.components.model_evaluation import evaluate_predictions

pred_csv = "D:/My Projects/Predictive Maintainability RUL/artifacts/processed_tabular/xgb_tabular_predictions_new.csv"
metrics_json = "D:/My Projects/Predictive Maintainability RUL/artifacts/processed_tabular/save_metrics.json"

metrics = evaluate_predictions(pred_csv, metrics_json)
print(metrics)
