import json
from pathlib import Path
from typing import Dict

import pandas as pd
import mlflow.xgboost
from sklearn.preprocessing import LabelEncoder

from Project.models.classification.feature_list import final_features_classification
from Project.db.repositories import load_dataframe


MODEL_URI = "models:/loan_default_classifier@production"
THRESHOLD = 0.45

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "classification"
RISK_CUTOFF_PATH = ARTIFACTS_DIR / "risk_cutoffs.json"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    return mlflow.xgboost.load_model(MODEL_URI)


def load_risk_cutoffs() -> Dict[str, float]:
    """Load PD cutoffs for risk segmentation."""
    with open(RISK_CUTOFF_PATH, "r") as f:
        return json.load(f)


def assign_risk_segment(prob: float, cutoffs: Dict[str, float]) -> str:
    if prob <= cutoffs["low_risk_max_pd"]:
        return "Low Risk"
    if prob <= cutoffs["medium_risk_max_pd"]:
        return "Medium Risk"
    return "High Risk"


def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Basic inference-time preprocessing.
    NOTE: Ideally this should be replaced with a persisted preprocessing pipeline.
    """
    X = X.copy()

    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        X[col] = X[col].astype(str)
        X[col].fillna(X[col].mode().iloc[0], inplace=True)

        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    return X


def predict_single(model, df: pd.DataFrame, cutoffs: Dict[str, float]) -> Dict:
    """Predict for a single row."""
    prob = float(model.predict_proba(df)[0, 1])
    decision = int(prob >= THRESHOLD)
    risk_segment = assign_risk_segment(prob, cutoffs)

    return {
        "default_probability": prob,
        "decision": decision,
        "risk_segment": risk_segment,
    }

def infer():
    model = load_model()
    cutoffs = load_risk_cutoffs()

    X = load_dataframe(
        table_name="loan_classification_system_data",
        columns=final_features_classification,
        limit=1, )

    if X.empty:
        raise ValueError("No data loaded for inference.")

    X = preprocess_features(X)

    result = predict_single(model, X, cutoffs)
    print(result)


if __name__ == "__main__":
    infer()
