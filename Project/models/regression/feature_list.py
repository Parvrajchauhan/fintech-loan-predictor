final_features_regression = [

    # numeric
    "AMT_INCOME_TOTAL",
    "DAYS_BIRTH",
    "DAYS_REGISTRATION",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "REGION_POPULATION_RELATIVE",

    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",

    "Annuity_to_Income_Ratio",

    # POS (light usage signals)
    "pos_mean_cnt_instalment",

    # Previous applications
    "avg_prev_amt_credit",
    "prev_num_approved",

    # Credit card
    "avg_cc_max_limit_used",

    # categorical
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
]