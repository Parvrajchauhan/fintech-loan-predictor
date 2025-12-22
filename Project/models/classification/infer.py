
import pandas as pd
import mlflow.xgboost
from sklearn.preprocessing import LabelEncoder
import json

final_features_classification = [

    # numeric 
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",

    "Credit_to_Income_Ratio",
    "Annuity_to_Income_Ratio",
    "Annuity_to_Credit_Ratio",

    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",

    "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH",
    "DAYS_REGISTRATION",
    "DAYS_LAST_PHONE_CHANGE",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",

    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",

    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_YEAR",

    # bureau engineered 
    "bureau_overdue_months",
    "max_overdue_duration",
    "num_bad_statuses",
    "total_active_bureau_loans",
    "avg_credit_utilization",

    # POS 
    "pos_num_loans",
    "pos_mean_cnt_instalment",
    "pos_late_ratio",
    "pos_max_dpd",

    # Installments
    "inst_mean_delay_days",
    "inst_max_delay_days",
    "inst_payment_delay_ratio",
    "inst_late_ratio",
    "inst_late_count",
    "inst_missed_payment_count",
    "inst_min_installment",
    "inst_mean_installment",

    # Previous applications 
    "avg_prev_amt_credit",
    "avg_prev_interest_rate",
    "prev_num_approved",
    "prev_num_rejected",
    "prev_early_repayment_count",

    # Credit card
    "max_cc_sk_dpd_def",
    "CC_PAYMENT_RATIO",
    "DELINQUENCY_SEVERITY",
    "avg_cc_max_limit_used",
    "avg_cc_mean_drawings_atm_current",

    # binary flags 
    "has_overdue",
    "has_active_bureau_loan",
    "has_late_payment",
    "has_missed_payment",
    "has_prev_rejection",
    "has_early_repayment",
    "HAS_CC_DELINQUENCY",
    "NO_CC_PAYMENT_FLAG",

    # categorical 
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",

    #Target
    "TARGET"
]


MODEL_URI = "models:/loan_default_classifier@production"
THRESHOLD = 0.07


def load_model():
    return mlflow.xgboost.load_model(MODEL_URI)



def load_risk_cutoffs():
    with open("risk_cutoffs.json", "r") as f:
        return json.load(f)


def assign_risk_segment(prob, cutoffs):
    if prob <= cutoffs["low_risk_max_pd"]:
        return "Low Risk"
    elif prob <= cutoffs["medium_risk_max_pd"]:
        return "Medium Risk"
    else:
        return "High Risk"



def predict(df: pd.DataFrame):
    model = load_model()
    cutoffs = load_risk_cutoffs()

    prob = model.predict_proba(df)[:, 1][0]
    decision = int(prob >= THRESHOLD)
    risk_segment = assign_risk_segment(prob, cutoffs)

    return {
        "default_probability": float(prob),
        "decision": decision,      
        "risk_segment": risk_segment
    }

if __name__ == "__main__":
    X = pd.read_csv("data/sample_classification_row.csv")
    cat_cols = X.select_dtypes(include=['object','category']).columns
    for col in cat_cols:
        X[col] = X[col].astype(str)
        most_freq = X[col].mode()[0]
        X[col] = X[col].fillna(most_freq)
        le = LabelEncoder()
        le.fit(X[col])
        X[col] = le.transform(X[col])
    result = predict(X)

    print(result)
