from pydantic import BaseModel, Field
from typing import Optional


class LoanAmountPredictionRequest(BaseModel):
    """
    Input schema for loan amount regression model.
    Used internally by bank employees + backend systems.
    """

    # Employee-verified fields
    age_years: int = Field(
        ..., ge=18, le=75, description="Customer age in years (derived from DOB)"
    )

    cnt_fam_members: int = Field(
        ..., ge=1, le=15, description="Number of family members"
    )

    name_income_type: str = Field(
        ..., description="Income type (Salaried, Self-employed, Business, etc.)"
    )

    name_education_type: str = Field(
        ..., description="Highest education level"
    )

    occupation_type: Optional[str] = Field(
        None, description="Customer occupation"
    )

    organization_type: Optional[str] = Field(
        None, description="Employer / industry type"
    )

    name_housing_type: str = Field(
        ..., description="Housing type (Owned, Rented, With parents)"
    )

    name_contract_type: str = Field(
        ..., description="Loan contract type (Cash loan / Revolving loan)"
    )
    # System-fetched fields
    amt_income_total: float = Field(
        ..., gt=0, description="Total verified monthly or annual income"
    )

    ext_source_1: Optional[float] = Field(
        None, ge=0, le=1, description="External credit bureau score 1"
    )

    ext_source_2: Optional[float] = Field(
        None, ge=0, le=1, description="External credit bureau score 2"
    )

    ext_source_3: Optional[float] = Field(
        None, ge=0, le=1, description="External credit bureau score 3"
    )

    avg_cc_max_limit_used: Optional[float] = Field(
        None, ge=0, le=1, description="Avg credit card utilization ratio"
    )

    avg_prev_amt_credit: Optional[float] = Field(
        None, ge=0, description="Average amount of previously approved loans"
    )

    prev_num_approved: int = Field(
        0, ge=0, description="Number of previously approved loans"
    )

    pos_mean_cnt_instalment: Optional[float] = Field(
        None, ge=0, description="Mean number of installments from POS loans"
    )
    # Derived / geo features
    region_rating_client: int = Field(
        ..., ge=1, le=3, description="Region rating (1 = best, 3 = worst)"
    )

    region_population_relative: float = Field(
        ..., gt=0, description="Population relative of client region"
    )

    annuity_to_income_ratio: Optional[float] = Field(
        None, ge=0, le=2, description="EMI to income ratio"
    )


class LoanAmountPredictionResponse(BaseModel):
    """
    Output shown to bank employee
    """

    predicted_loan_amount: float = Field(
        ..., description="Predicted loan amount customer is likely to request"
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

    explanation: dict = Field(
        ..., description="Top feature contributions (SHAP-style)"
    )
