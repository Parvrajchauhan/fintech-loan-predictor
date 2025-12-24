
final_features_classification = [

    # numeric 
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",

    "Credit_to_Income_Ratio",
    "Annuity_to_Income_Ratio",
    "Annuity_to_Credit_Ratio",

    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",

    "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH",
    "DAYS_REGISTRATION",
    "DAYS_LAST_PHONE_CHANGE",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",

    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",

    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_YEAR",

    # bureau engineered 
    "bureau_overdue_months",
    "max_overdue_duration",
    "num_bad_statuses",
    "total_active_bureau_loans",
    "avg_credit_utilization",

    # POS 
    "pos_num_loans",
    "pos_mean_cnt_instalment",
    "pos_late_ratio",
    "pos_severe_late_ratio",
    "pos_max_dpd",

    # Installments
    "inst_mean_delay_days",
    "inst_max_delay_days",
    "inst_payment_delay_ratio",
    "inst_late_ratio",
    "inst_late_count",
    "inst_missed_payment_count",
    "inst_min_installment",
    "inst_mean_installment",

    # Previous applications 
    "avg_prev_amt_requested",
    "avg_prev_amt_credit",
    "avg_prev_interest_rate",
    "prev_num_approved",
    "prev_num_rejected",
    "prev_early_repayment_count",

    # Credit card
    "max_cc_sk_dpd_def",
    "CC_PAYMENT_RATIO",
    "DELINQUENCY_SEVERITY",
    "avg_cc_max_limit_used",
    "avg_cc_mean_drawings_atm_current",

    # binary flags 
    "has_overdue",
    "has_active_bureau_loan",
    "has_late_payment",
    "has_missed_payment",
    "has_prev_rejection",
    "has_early_repayment",
    "HAS_CC_DELINQUENCY",
    "NO_CC_PAYMENT_FLAG",

    # categorical 
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
