
import pandas as pd
import shap
import xgboost as xgb
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt


from Project.features.build_features import build_all_features
from Project.features.imputation import NUM_MEDIAN, NUM_ZERO, CAT_UNKNOWN, fit_imputer, apply_imputation
from Project.models.classification.train import load_data

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


MODEL_URI = "models:/loan_default_classifier@production"
TARGET = "TARGET"
ID_COL = "SK_ID_CURR"



if __name__ == "__main__":
    mlflow.set_experiment("loan_default_classification")
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    df = build_all_features(paths)
    stats = fit_imputer(df)
    df = apply_imputation(df, stats)
    X, y = load_data(df)

    model = mlflow.xgboost.load_model(MODEL_URI)
    
    X_shap = X.sample(1000, random_state=42)
    booster = model.get_booster()
    dmat = xgb.DMatrix(X_shap)
    # get SHAP contributions
    shap_values = booster.predict(
        dmat,
        pred_contribs=True)
    shap_values = shap_values[:, :-1]
    shap_df = pd.DataFrame(
        shap_values,
        columns=X_shap.columns)
    
    shap_df.abs().mean().sort_values(ascending=False).plot(
        kind="bar",
        figsize=(14, 6)
        )
    plt.tight_layout()
    plt.savefig("shap_summary_classification.png")
    plt.close()

    # Local (single user)
    plt.figure(figsize=(14, 8))
    shap_df.iloc[801].sort_values().plot(kind="barh")
    plt.title(f"SHAP Values for Sample 801", fontsize=14, pad=20)
    plt.xlabel("SHAP Value", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.tight_layout()  
    plt.show()
    plt.savefig("shap_local_classification.png")
    plt.close()

    mlflow.log_artifact("shap_summary_classification.png")
    mlflow.log_artifact("shap_local_classification.png")

    print("SHAP explanations generated.")
