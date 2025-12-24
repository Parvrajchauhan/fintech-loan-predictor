import pandas as pd
import mlflow.lightgbm
from sklearn.preprocessing import LabelEncoder
import shap
from Project.models.regression.feature_list import final_features_regression
from Project.features.imputation import CAT_UNKNOWN
from Project.db.repositories import load_dataframe

MODEL_URI = "models:/loan_amount_regressor@production"

def load_model():
    return mlflow.lightgbm.load_model(MODEL_URI)


if __name__ == "__main__":
    sample = load_dataframe(
        "loan_regression_system_data",
        columns=final_features_regression,
        limit=1000
    )
    cat_cols= [col for col in final_features_regression if col in CAT_UNKNOWN]
    for col in cat_cols:
        sample[col] = sample[col].fillna("Unknown")
        sample[col] = LabelEncoder().fit_transform(sample[col])

    model = load_model()
    preds = model.predict(sample)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    i=9
    shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[i],
        base_values=explainer.expected_value,
        data=sample.iloc[i],
        feature_names=sample.columns))

    print("Predicted loan amount:", preds[0])
