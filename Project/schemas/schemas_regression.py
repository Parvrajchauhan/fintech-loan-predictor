from pydantic import BaseModel, Field
from typing import Optional, Dict


# =========================
# REQUEST
# =========================
class LoanAmountPredictionRequest(BaseModel):
    """
    Input schema for loan amount regression model.
    Used internally by bank employees + backend systems.
    """

    # ---------------------
    # Identifier (MANDATORY)
    # ---------------------
    index_id: int = Field(
        ..., description="Unique customer / application identifier"
    )

    # ---------------------
    # Employee-provided overrides (optional)
    # ---------------------
    age_years: Optional[int] = Field(
        None, ge=18, le=75, description="Customer age in years"
    )

    cnt_fam_members: Optional[int] = Field(
        None, ge=1, le=15, description="Number of family members"
    )

    name_income_type: Optional[str] = None
    name_education_type: Optional[str] = None
    occupation_type: Optional[str] = None
    organization_type: Optional[str] = None
    name_housing_type: Optional[str] = None
    name_contract_type: Optional[str] = None

    # ---------------------
    # Income / loan overrides
    # ---------------------
    amt_income_total: Optional[float] = Field(None, gt=0)

    # ---------------------
    # External scores (optional)
    # ---------------------
    ext_source_1: Optional[float] = Field(None, ge=0, le=1)
    ext_source_2: Optional[float] = Field(None, ge=0, le=1)
    ext_source_3: Optional[float] = Field(None, ge=0, le=1)

    # ---------------------
    # System-fetched behavioral features
    # ---------------------
    avg_cc_max_limit_used: Optional[float] = None
    avg_prev_amt_credit: Optional[float] = None
    prev_num_approved: Optional[int] = Field(0, ge=0)
    pos_mean_cnt_instalment: Optional[float] = None

    # ---------------------
    # Geo / region (system-fetched)
    # ---------------------
    region_rating_client: Optional[int] = Field(None, ge=1, le=3)
    region_population_relative: Optional[float] = None

    # ---------------------
    # Backend-derived (DO NOT send from client)
    # ---------------------
    annuity_to_income_ratio: Optional[float] = None


class LoanAmountPredictionResponse(BaseModel):
    """
    Output shown to bank employee
    """

    predicted_loan_amount: float = Field(
        ..., description="Predicted sustainable loan amount"
    )

    lower_bound: float = Field(
        ..., description="Lower confidence bound"
    )

    upper_bound: float = Field(
        ..., description="Upper confidence bound"
    )

    confidence_score: float = Field(
        ..., ge=0, le=1, description="Model confidence score"
    )

    explanation: Dict = Field(
        ..., description="Top feature contributions (SHAP-style)"
    )

