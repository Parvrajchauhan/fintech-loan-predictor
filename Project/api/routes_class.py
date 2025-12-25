import pandas as pd
import joblib
from fastapi import APIRouter, HTTPException
from Project.models.classification.train import encode
from Project.schemas.schemas_classification import (
    LoanDefaultPredictionRequest,
    LoanDefaultPredictionResponse,
)

from Project.api.deps import (
    load_model,
    load_risk_cutoffs,
    assign_risk_segment,
    THRESHOLD,
)

from Project.api.class_features import get_classification_features
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "classification"

router = APIRouter()

MODEL_FEATURES_CLASS = joblib.load(ARTIFACTS_DIR /"model_features_class.pkl")


@router.post(
    "/predict",
    response_model=LoanDefaultPredictionResponse,
    summary="Predict default probability and decision",
)
def predict(payload: LoanDefaultPredictionRequest):
    try:
        final_features = get_classification_features(
            table_name="loan_classification_system_data",   # feature-store table
            index_id=payload.index_id,
            user_features=payload.model_dump(exclude={"index_id"}),
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Model inference
    model = load_model()
    cutoffs = load_risk_cutoffs()

    df = pd.DataFrame([final_features])
    df = df[MODEL_FEATURES_CLASS] 
    df=encode(df)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.fillna(0.0)

    prob = float(model.predict_proba(df)[:, 1][0])

    decision = int(prob >= THRESHOLD)
    risk_segment = assign_risk_segment(prob, cutoffs)

    # Response
    return LoanDefaultPredictionResponse(
        default_probability=prob,
        decision=decision,
        risk_segment=risk_segment,
        explanation={
            "top_risk_driver": "EXT_SOURCE_2",
            "confidence": round(prob, 3),
        },
    )
