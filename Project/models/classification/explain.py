
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt

from Project.db.repositories import load_dataframe
from Project.models.classification.feature_list import final_features_classification
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "classification"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_URI = "models:/loan_default_classifier@production"
TARGET = "TARGET"
ID_COL = "SK_ID_CURR"



if __name__ == "__main__":
    mlflow.set_experiment("loan_default_classification")
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    X_shap= load_dataframe(
        "loan_classification_system_data",
        columns=final_features_classification,
        limit=1000)

    model = mlflow.xgboost.load_model(MODEL_URI)
    
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
    plt.savefig(ARTIFACTS_DIR / "shap_summary_classification.png")
    plt.close()

    # Local (single user)
    plt.figure(figsize=(14, 8))
    shap_df.iloc[801].sort_values().plot(kind="barh")
    plt.title(f"SHAP Values for Sample 801", fontsize=14, pad=20)
    plt.xlabel("SHAP Value", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.tight_layout()  
    plt.show()
    plt.savefig(ARTIFACTS_DIR / "shap_local_classification.png")
    plt.close()

    mlflow.log_artifact(str(ARTIFACTS_DIR / "shap_summary_classification.png"))
    mlflow.log_artifact(str(ARTIFACTS_DIR / "shap_local_classification.png"))

    print("SHAP explanations generated.")
