# models/regression/evaluate.py

import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm
import matplotlib.pyplot as plt
from Project.features.build_features import build_all_features
from Project.features.imputation import NUM_MEDIAN, NUM_ZERO, CAT_UNKNOWN, fit_imputer, apply_imputation
from Project.models.regression.train import load_data

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

paths = {
    "application_train": DATA_DIR / "application_train.csv",
    "bureau": DATA_DIR / "bureau.csv",
    "bureau_balance": DATA_DIR / "bureau_balance.csv",
    "pos": DATA_DIR / "POS_CASH_balance.csv",
    "installments": DATA_DIR / "installments_payments.csv",
    "previous_application": DATA_DIR / "previous_application.csv",
    "credit_card": DATA_DIR / "credit_card_balance.csv",
}


def evaluate(model, X, y):
    preds = model.predict(X)

    mae = np.mean(np.abs(np.expm1(preds) - np.expm1(y)))
    rmse = np.sqrt(np.mean((np.expm1(preds) - np.expm1(y)) ** 2))

    residuals = y - preds

    return mae, rmse, residuals


def plot_residuals(residuals):
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=50)
    plt.title("Residual Distribution")
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("residuals.png")
    plt.close()


if __name__ == "__main__":   
    mlflow.set_experiment("loan_amount_regression_prob")
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    df = build_all_features(paths)
    stats = fit_imputer(df)
    df = apply_imputation(df, stats)
    X, y = load_data(df)

    model = mlflow.lightgbm.load_model("models:/loan_amount_regressor@production")

    mae, rmse, residuals = evaluate(model, X, y)

    plot_residuals(residuals)

    mlflow.log_metric("eval_mae", mae)
    mlflow.log_metric("eval_rmse", rmse)
    mlflow.log_artifact("residuals.png")

    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f}")
