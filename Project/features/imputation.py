import pandas as pd

# Feature groups
NUM_MEDIAN = [
    # Application
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",

    "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH",
    "DAYS_REGISTRATION",
    "DAYS_LAST_PHONE_CHANGE",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",

    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",

    "Credit_to_Income_Ratio",
    "Annuity_to_Income_Ratio",
    "Annuity_to_Credit_Ratio",
    # Bureau (STEP 2)
    "mean_days_credit",
    "avg_credit_utilization",
    # Installments (STEP 3)
    "inst_mean_delay_days",
    "inst_max_delay_days",
    "inst_payment_delay_ratio",
    "inst_late_ratio",
    "inst_min_installment",
    "inst_mean_installment",
    # POS (STEP 3)
    "pos_late_ratio",
    "pos_severe_late_ratio",
    "pos_max_dpd",
    # Previous applications (STEP 4)
    "avg_prev_amt_requested",
    "avg_prev_amt_credit",
    "avg_prev_interest_rate",
    # Credit card
    "avg_cc_max_limit_used",
    "avg_cc_mean_drawings_atm_current",
]
NUM_ZERO = [
    # Social / Bureau raw
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",

    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    # Bureau engineered
    "bureau_overdue_months",
    "max_overdue_duration",
    "num_bad_statuses",
    "total_active_bureau_loans",
    "has_overdue",
    "has_active_bureau_loan",
    # POS
    "pos_num_loans",
    "pos_mean_cnt_instalment",
    # Installments
    "inst_late_count",
    "inst_missed_payment_count",
    "has_late_payment",
    "has_missed_payment",
    # Previous applications
    "prev_num_approved",
    "prev_num_rejected",
    "prev_early_repayment_count",
    "has_prev_rejection",
    "has_early_repayment",
    # Credit card
    "max_cc_sk_dpd_def",
    "CC_PAYMENT_RATIO",
    "DELINQUENCY_SEVERITY",
    "HAS_CC_DELINQUENCY",
    "NO_CC_PAYMENT_FLAG",
]


CAT_UNKNOWN = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",

    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",

    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
]


def fit_imputer(df: pd.DataFrame):
    stats = {}

    for col in NUM_MEDIAN:
        if col in df.columns:
            stats[col] = df[col].median()

    return stats


def apply_imputation(df: pd.DataFrame, stats: dict):
    df = df.copy()

    # Median imputation
    for col, median in stats.items():
        if col in df.columns:
            df[col] = df[col].fillna(median)

    # Zero imputation
    for col in NUM_ZERO:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Categorical imputation
    for col in CAT_UNKNOWN:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df
