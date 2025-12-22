import pandas as pd
import mlflow.lightgbm
from sklearn.preprocessing import LabelEncoder
import shap

final_features_regression = [

    # numeric
    "AMT_INCOME_TOTAL",
    "DAYS_BIRTH",
    "DAYS_REGISTRATION",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "REGION_POPULATION_RELATIVE",

    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",

    "Annuity_to_Income_Ratio",

    # POS (light usage signals)
    "pos_mean_cnt_instalment",

    # Previous applications
    "avg_prev_amt_credit",
    "prev_num_approved",

    # Credit card
    "avg_cc_max_limit_used",

    # categorical
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
]

MODEL_URI = "models:/loan_amount_regressor@production"

def load_model():
    return mlflow.lightgbm.load_model(MODEL_URI)


if __name__ == "__main__":
    sample = pd.read_csv("data/sample_inference_row.csv")
    cat_cols= [col for col in final_features_regression if col in CAT_UNKNOWN]
    for col in cat_cols:
        sample[col] = sample[col].fillna("Unknown")
        sample[col] = LabelEncoder().fit_transform(sample[col])

    model = load_model()
    preds = model.predict(sample)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    shap.waterfall_plot(
    shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=sample.iloc,
        feature_names=sample.columns))

    print("Predicted loan amount:", preds[0])
