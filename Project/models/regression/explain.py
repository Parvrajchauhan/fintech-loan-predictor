import pandas as pd
import shap
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

if __name__ == "__main__":
    mlflow.set_experiment("loan_amount_regression_prob")
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    df = build_all_features(paths)
    stats = fit_imputer(df)
    df = apply_imputation(df, stats)
    X, y = load_data(df)

    model = mlflow.lightgbm.load_model("models:/loan_amount_regressor@production")

    X_shap = X.sample(500, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    # Summary plot
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    plt.close()

    shap.summary_plot(shap_values, X_shap, plot_type="bar")
    plt.tight_layout()
    plt.savefig("shap_summary_bar.png")
    plt.close()

    mlflow.log_artifact("shap_summary_bar.png")
    mlflow.log_artifact("shap_summary.png")

    print("SHAP explanations generated and logged.")
