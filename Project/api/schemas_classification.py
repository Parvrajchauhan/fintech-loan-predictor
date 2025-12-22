from pydantic import BaseModel, Field
from typing import Optional


class LoanDefaultPredictionRequest(BaseModel):
    # Employee-verified
    days_birth: int = Field(..., description="Age derived from DOB")
    cnt_fam_members: int = Field(..., ge=1)

    code_gender: str
    name_income_type: str
    name_education_type: str
    name_family_status: str
    name_housing_type: str
    occupation_type: Optional[str]
    organization_type: Optional[str]

    flag_own_car: str
    flag_own_realty: str
    name_contract_type: str

    # System-fetched
    amt_income_total: float = Field(..., gt=0)
    amt_credit: float = Field(..., gt=0)
    amt_annuity: float = Field(..., gt=0)

    ext_source_1: Optional[float] = Field(None, ge=0, le=1)
    ext_source_2: Optional[float] = Field(None, ge=0, le=1)
    ext_source_3: Optional[float] = Field(None, ge=0, le=1)

    # Derived (backend)
    credit_to_income_ratio: float
    annuity_to_income_ratio: float
    annuity_to_credit_ratio: float

    region_population_relative: float
    region_rating_client: int

    days_registration: int
    days_last_phone_change: int

    # Bureau / behavior
    has_overdue: bool
    has_active_bureau_loan: bool
    has_late_payment: bool
    has_missed_payment: bool
    has_prev_rejection: bool
    has_early_repayment: bool
    has_cc_delinquency: bool
    no_cc_payment_flag: bool


class LoanDefaultPredictionResponse(BaseModel):
    default_probability: float = Field(..., ge=0, le=1)
    risk_segment: str = Field(..., description="Low / Medium / High Risk")
    decision: int = Field(..., description="0 = Approve, 1 = Reject")

    explanation: dict = Field(
        ..., description="Top risk drivers (SHAP-style)"
    )
