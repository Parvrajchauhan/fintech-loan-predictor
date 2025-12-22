import pandas as pd
from fastapi import APIRouter
import joblib
from api.schemas_classification import LoanDefaultPredictionRequest, LoanDefaultPredictionResponse
from Project.api.deps import load_model, load_risk_cutoffs, assign_risk_segment,  THRESHOLD


router = APIRouter()
MODEL_FEATURES_CLASS = joblib.load("model_features_class.pkl")




@router.post(
    "/predict",
    response_model=LoanDefaultPredictionResponse,
    summary="Predict default probability and decision",
)
def predict(payload: LoanDefaultPredictionRequest):

    model = load_model()
    cutoffs = load_risk_cutoffs()

    # Convert request → DataFrame
    df = pd.DataFrame(
        [payload.model_dump()]  # Pydantic v2 safe
    )

    # Ensure column order matches training
    df = df[MODEL_FEATURES_CLASS]

    # Predict probability
    prob = float(model.predict_proba(df)[:, 1][0])

    # Decision & risk
    decision = int(prob >= THRESHOLD)
    risk_segment = assign_risk_segment(prob, cutoffs)

    # Optional: SHAP / explanation placeholder
    explanation = {
        "top_risk_driver": "EXT_SOURCE_2",
        "confidence": round(prob, 3)
    }

    return LoanDefaultPredictionResponse(
        default_probability=prob,
        decision=decision,
        risk_segment=risk_segment,
        explanation=explanation,
    )


