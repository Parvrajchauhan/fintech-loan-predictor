
import mlflow
import shap
import mlflow.lightgbm
import matplotlib.pyplot as plt
from Project.db.repositories import load_dataframe
from Project.models.regression.feature_list import final_features_regression
from Project.features.imputation import CAT_UNKNOWN
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "regression"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def explain():
    mlflow.set_experiment("loan_amount_regression_prob")
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    X_shap = load_dataframe(
        "loan_regression_system_data",
        columns=final_features_regression,
        limit=1000
    )
    cat_cols= [col for col in final_features_regression if col in CAT_UNKNOWN]
    for col in cat_cols:
        X_shap[col] = X_shap[col].fillna("Unknown")
        X_shap[col] = LabelEncoder().fit_transform(X_shap[col])
    
    model = mlflow.lightgbm.load_model("models:/loan_amount_regressor@production")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    # Summary plot
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR /"shap_summary.png")
    plt.close()

    shap.summary_plot(shap_values, X_shap, plot_type="bar")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "shap_summary_bar.png")
    plt.close()

    mlflow.log_artifact(
    str(ARTIFACTS_DIR / "shap_summary_bar.png")
)
    mlflow.log_artifact(
    str(ARTIFACTS_DIR / "shap_summary.png")
)

    print("SHAP explanations generated and logged.")
