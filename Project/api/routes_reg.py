
import pandas as pd
from fastapi import APIRouter
import joblib
from api.schemas_regression import LoanAmountPredictionRequest, LoanAmountPredictionResponse
from Project.api.deps import  load_regression_model

router = APIRouter()
MODEL_FEATURES_REG = joblib.load("model_features_reg.pkl")

@router.post(
    "/predict-loan-amount",
    response_model=LoanAmountPredictionResponse,
    summary="Predict likely loan amount",
)
def predict_loan_amount(payload: LoanAmountPredictionRequest):

    model = load_regression_model()

    # Convert request → DataFrame
    df = pd.DataFrame(
        [payload.model_dump()]  # Pydantic v2 safe
    )

    # Ensure column order matches training
    df = df[MODEL_FEATURES_REG]

    # Predict loan amount
    predicted_amount = float(model.predict(df)[0])

    # Optional confidence band (simple heuristic / placeholder)
    lower_bound = predicted_amount * 0.9
    upper_bound = predicted_amount * 1.1

    # Optional explanation placeholder
    explanation = {
        "top_driver": "AMT_INCOME_TOTAL",
        "confidence": 0.8
    }

    return LoanAmountPredictionResponse(
        predicted_loan_amount=predicted_amount,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        confidence_score=0.8,
        explanation=explanation,
    )





