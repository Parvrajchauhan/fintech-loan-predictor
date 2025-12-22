import json
import mlflow.lightgbm
import mlflow.xgboost
from functools import lru_cache


MODEL_URI_CLASS = "models:/loan_default_classifier@production"
MODEL_URI_REG = "models:/loan_amount_regressor@production"

RISK_CUTOFF_PATH = "risk_cutoffs.json"
THRESHOLD = 0.45

@lru_cache()
def load_model():
    return mlflow.xgboost.load_model(MODEL_URI_CLASS)


@lru_cache()
def load_risk_cutoffs():
    with open(RISK_CUTOFF_PATH, "r") as f:
        return json.load(f)


def assign_risk_segment(prob: float, cutoffs: dict) -> str:
    if prob <= cutoffs["low_risk_max_pd"]:
        return "Low Risk"
    elif prob <= cutoffs["medium_risk_max_pd"]:
        return "Medium Risk"
    else:
        return "High Risk"


@lru_cache()
def load_regression_model():
    return mlflow.lightgbm.load_model(MODEL_URI_REG)