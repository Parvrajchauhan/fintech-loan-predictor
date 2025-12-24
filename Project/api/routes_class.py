import pandas as pd
import joblib
from fastapi import APIRouter, HTTPException

from schemas.schemas_classification import (
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


router = APIRouter()

MODEL_FEATURES_CLASS = joblib.load("model_features_class.pkl")


@router.post(
    "/predict",
    response_model=LoanDefaultPredictionResponse,
    summary="Predict default probability and decision",
)
def predict(payload: LoanDefaultPredictionRequest):
    try:
        final_features = get_classification_features(
            table_name="classification_features",   # feature-store table
            index_id=payload.index_id,
            user_features=payload.model_dump(exclude={"index_id"}),
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Model inference
    model = load_model()
    cutoffs = load_risk_cutoffs()

    df = pd.DataFrame([final_features])
    df = df[MODEL_FEATURES_CLASS]   # enforce training order

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
