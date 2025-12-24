import pandas as pd
import joblib
from fastapi import APIRouter, HTTPException

from schemas.schemas_regression import (
    LoanAmountPredictionRequest,
    LoanAmountPredictionResponse,
)

from Project.api.deps import load_regression_model
from Project.api.reg_features import get_regression_features


router = APIRouter()

MODEL_FEATURES_REG = joblib.load("model_features_reg.pkl")


@router.post(
    "/predict-loan-amount",
    response_model=LoanAmountPredictionResponse,
    summary="Predict likely loan amount",
)
def predict_loan_amount(payload: LoanAmountPredictionRequest):

    # Fetch + merge system features
    try:
        final_features = get_regression_features(
            table_name="regression_features",   # feature-store table
            index_id=payload.index_id,
            user_features=payload.model_dump(exclude={"index_id"}),
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Model inference
    model = load_regression_model()

    df = pd.DataFrame([final_features])
    df = df[MODEL_FEATURES_REG]   # enforce training schema

    predicted_amount = float(model.predict(df)[0])

    # Confidence band (demo-safe)
    lower_bound = predicted_amount * 0.9
    upper_bound = predicted_amount * 1.1

    # Response
    return LoanAmountPredictionResponse(
        predicted_loan_amount=predicted_amount,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        confidence_score=0.8,
        explanation={
            "top_driver": "AMT_INCOME_TOTAL",
            "confidence": 0.8,
        },
    )
