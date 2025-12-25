import json
import pandas as pd
import mlflow.xgboost
import mlflow.lightgbm
from functools import lru_cache
from typing import Dict, List
from pathlib import Path


from Project.db.repositories import load_single_row

MODEL_URI_CLASS = "models:/loan_default_classifier@production"
MODEL_URI_REG = "models:/loan_amount_regressor@production"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "classification"
RISK_CUTOFF_PATH = ARTIFACTS_DIR / "risk_cutoffs.json"
THRESHOLD = 0.45


# MLflow
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


# PostgreSQL
def fetch_and_merge_features(
    table_name: str,
    index_id: int,
    user_features: Dict
) -> Dict:
    """
    Fetch ALL system features from Postgres for a given index_id
    and override them with user-provided values.
    """

    df = load_single_row(
        table_name=table_name,
        index_id=index_id,
        columns=None  
    )

    if df.empty:
        raise ValueError(f"INDEX_ID {index_id} not found in {table_name}")

    system_features = df.iloc[0].to_dict()
    system_features.pop("index_id", None)

    # Override with user-provided fields
    final_features = {**system_features, **user_features}

    return final_features
