
import pandas as pd
import shap
import mlflow
import mlflow.lightgbm
import matplotlib.pyplot as plt


MODEL_URI = "models:/loan_default_classifier@production"
TARGET = "TARGET"
ID_COL = "SK_ID_CURR"


if __name__ == "__main__":
    df = pd.read_csv("data/final_classification_table.csv")
    X = df.drop(columns=[TARGET, ID_COL])

    model = mlflow.lightgbm.load_model(MODEL_URI)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.sample(500, random_state=42))[1]

    # Global
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_classification.png")
    plt.close()

    # Local (single user)
    shap.force_plot(
        explainer.expected_value[1],
        shap_values[0],
        X.iloc[0],
        matplotlib=True,
        show=False,
    )
    plt.savefig("shap_local_classification.png")
    plt.close()

    mlflow.log_artifact("shap_summary_classification.png")
    mlflow.log_artifact("shap_local_classification.png")

    print("SHAP explanations generated.")
