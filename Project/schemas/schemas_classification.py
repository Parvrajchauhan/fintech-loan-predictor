from pydantic import BaseModel, Field
from typing import Optional,Dict

class LoanDefaultPredictionRequest(BaseModel):
    """
    Client-facing request.
    Only INDEX_ID + user-overrides are required.
    System & derived features are populated backend-side.
    """

    # ---------------------
    # Identifier (MANDATORY)
    # ---------------------
   # schema
    index_id: int = Field(..., description="Unique user/application id")

    # ---------------------
    # User-provided (optional overrides)
    # ---------------------
    code_gender: Optional[str] = None
    name_income_type: Optional[str] = None
    name_education_type: Optional[str] = None
    name_family_status: Optional[str] = None
    name_housing_type: Optional[str] = None
    occupation_type: Optional[str] = None
    organization_type: Optional[str] = None

    flag_own_car: Optional[str] = None
    flag_own_realty: Optional[str] = None
    name_contract_type: Optional[str] = None

    # User can override income/loan details if updated
    amt_income_total: Optional[float] = Field(None, gt=0)
    amt_credit: Optional[float] = Field(None, gt=0)
    amt_annuity: Optional[float] = Field(None, gt=0)

    # Optional external scores
    ext_source_1: Optional[float] = Field(None, ge=0, le=1)
    ext_source_2: Optional[float] = Field(None, ge=0, le=1)
    ext_source_3: Optional[float] = Field(None, ge=0, le=1)

    # ---------------------
    # Backend-derived (DO NOT send from client)
    # ---------------------
    credit_to_income_ratio: Optional[float] = None
    annuity_to_income_ratio: Optional[float] = None
    annuity_to_credit_ratio: Optional[float] = None

    # ---------------------
    # System-fetched (PostgreSQL)
    # ---------------------
    days_birth: Optional[int] = None
    days_registration: Optional[int] = None
    days_last_phone_change: Optional[int] = None

    cnt_fam_members: Optional[int] = Field(None, ge=1)
    region_population_relative: Optional[float] = None
    region_rating_client: Optional[int] = None

    # ---------------------
    # Bureau / behavior flags (system-generated)
    # ---------------------
    has_overdue: Optional[bool] = None
    has_active_bureau_loan: Optional[bool] = None
    has_late_payment: Optional[bool] = None
    has_missed_payment: Optional[bool] = None
    has_prev_rejection: Optional[bool] = None
    has_early_repayment: Optional[bool] = None
    has_cc_delinquency: Optional[bool] = None
    no_cc_payment_flag: Optional[bool] = None



class LoanDefaultPredictionResponse(BaseModel):
    default_probability: float = Field(..., ge=0, le=1)
    risk_segment: str = Field(..., description="Low / Medium / High Risk")
    decision: int = Field(..., description="0 = Approve, 1 = Reject")

    explanation: Dict = Field(
        ..., description="Top risk drivers (SHAP-style)"
    )

