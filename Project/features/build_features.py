import pandas as pd
import numpy as np

#application tavle
def build_application_features(app_path):
    app = pd.read_csv(app_path)

    app["Credit_to_Income_Ratio"] = app["AMT_CREDIT"] / app["AMT_INCOME_TOTAL"].replace(0, np.nan)
    app["Annuity_to_Income_Ratio"] = app["AMT_ANNUITY"] / app["AMT_INCOME_TOTAL"].replace(0, np.nan)
    app["Annuity_to_Credit_Ratio"] = app["AMT_ANNUITY"] / app["AMT_CREDIT"].replace(0, np.nan)
    app["AMT_REQ_CREDIT_BUREAU_MON"]=(app["AMT_REQ_CREDIT_BUREAU_MON"]>0).astype(int)


    selected_cols = [
        'SK_ID_CURR','TARGET','NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR',
        'FLAG_OWN_REALTY','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY',
        'NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE',
        'REGION_POPULATION_RELATIVE','DAYS_BIRTH','DAYS_REGISTRATION',
        'OCCUPATION_TYPE','CNT_FAM_MEMBERS','REGION_RATING_CLIENT','ORGANIZATION_TYPE',
        'EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','OBS_60_CNT_SOCIAL_CIRCLE',
        'DEF_60_CNT_SOCIAL_CIRCLE','DAYS_LAST_PHONE_CHANGE',
        'AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_YEAR',
        "Credit_to_Income_Ratio",
        "Annuity_to_Income_Ratio",
        "Annuity_to_Credit_Ratio","EMPLOYED_FLAG"
    ]
    app['EMPLOYED_FLAG'] = (app['DAYS_EMPLOYED'] < 365243).astype(int)
    app['DAYS_BIRTH'] = abs(app['DAYS_BIRTH']) / 365
    app['DAYS_REGISTRATION'] = abs(app['DAYS_REGISTRATION']) / 365
    for col in ['Credit_to_Income_Ratio', 'Annuity_to_Income_Ratio', 'Annuity_to_Credit_Ratio']:
        upper_limit = app[col].quantile(0.99)
        app[col] = app[col].clip(upper=upper_limit)
    return app[selected_cols]


# Bureau + Bureau Balance
def build_bureau_features(bureau_path, bureau_balance_path):
    bureau = pd.read_csv(bureau_path)
    bureau_balance = pd.read_csv(bureau_balance_path)

    # Bureau balance features
    overdue_status = ["1", "2", "3", "4", "5"]
    bureau_balance["is_overdue"] = bureau_balance["STATUS"].isin(overdue_status).astype(int)

    bb_agg = (
        bureau_balance
        .groupby("SK_ID_BUREAU")
        .agg(
            bureau_overdue_months=("is_overdue", "sum"),          # total bad months
            max_overdue_duration=("is_overdue", "max"),           # any severe delinquency
        )
        .reset_index()
    )

    # Merge with bureau
    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    bureau[["bureau_overdue_months", "max_overdue_duration"]] = (
        bureau[["bureau_overdue_months", "max_overdue_duration"]].fillna(0)
    )

    # Credit utilization
    bureau["credit_utilization"] = (
        bureau["AMT_CREDIT_SUM"] /
        bureau["AMT_CREDIT_SUM_LIMIT"]
    )

    bureau["credit_utilization"] = bureau["credit_utilization"].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    # Aggregate to customer
    bureau_final = (
        bureau
        .groupby("SK_ID_CURR")
        .agg(
            total_active_bureau_loans=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
            bureau_overdue_months=("bureau_overdue_months", "sum"),
            max_overdue_duration=("max_overdue_duration", "max"),
            mean_days_credit=("DAYS_CREDIT", "mean"),
            num_bad_statuses=("bureau_overdue_months", lambda x: (x > 0).sum()),
            avg_credit_utilization=("credit_utilization", "mean"),
        )
        .reset_index()
    )

    # Binary risk flags
    bureau_final["has_overdue"] = (bureau_final["bureau_overdue_months"] > 0).astype(int)
    bureau_final["has_active_bureau_loan"] = (
        bureau_final["total_active_bureau_loans"] > 0
    ).astype(int)

    return bureau_final


# POS Cash Balance
def build_pos_features(pos_path):
    pos = pd.read_csv(pos_path)

    pos["is_late"] = pos["SK_DPD"] > 0
    pos["is_severely_late"] = pos["SK_DPD"] > 30

    pos_agg = (
        pos
        .groupby("SK_ID_CURR")
        .agg(
            pos_num_loans=("SK_ID_PREV", "nunique"),
            pos_mean_cnt_instalment=("CNT_INSTALMENT", "mean"),

            # risk behaviour
            pos_late_ratio=("is_late", "mean"),
            pos_severe_late_ratio=("is_severely_late", "mean"),
            pos_max_dpd=("SK_DPD", "max"),
        )
        .reset_index()
    )
    return pos_agg


# Installments Payments
def build_installment_features(installments_path):
    inst = pd.read_csv(installments_path)

    inst["payment_delay_days"] = (
        inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    )

    inst["is_late"] = inst["payment_delay_days"] > 0
    inst["is_missed"] = inst["AMT_PAYMENT"].isna() | (inst["AMT_PAYMENT"] == 0)

    # delay severity ratio
    inst["delay_ratio"] = (
        inst["payment_delay_days"] / inst["DAYS_INSTALMENT"].abs()
    )

    inst["delay_ratio"] = inst["delay_ratio"].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    inst_agg = (
        inst
        .groupby("SK_ID_CURR")
        .agg(
            # delay behavior
            inst_mean_delay_days=("payment_delay_days", "mean"),
            inst_max_delay_days=("payment_delay_days", "max"),
            inst_payment_delay_ratio=("delay_ratio", "mean"),

            # late behavior
            inst_late_ratio=("is_late", "mean"),
            inst_late_count=("is_late", "sum"),

            # missed payments
            inst_missed_payment_count=("is_missed", "sum"),

            # payment size
            inst_min_installment=("AMT_INSTALMENT", "min"),
            inst_mean_installment=("AMT_INSTALMENT", "mean"),
        )
        .reset_index()
    )

    inst_agg["has_late_payment"] = (inst_agg["inst_late_count"] > 0).astype(int)
    inst_agg["has_missed_payment"] = (inst_agg["inst_missed_payment_count"] > 0).astype(int)

    return inst_agg


# Previous Applications

def build_previous_application_features(prev_path):
    prev = pd.read_csv(prev_path)

    # normalize status
    prev["status"] = prev["NAME_CONTRACT_STATUS"].astype(str).str.lower()

    prev["is_prev_approved"] = prev["status"].str.contains("approved", na=False).astype(int)
    prev["is_prev_rejected"] = prev["status"].str.contains("refus|reject|canceled", na=False).astype(int)

    # early repayment: closed before planned end
    prev["is_early_repayment"] = (
        (prev["DAYS_TERMINATION"] < prev["DAYS_DECISION"]) &
        prev["DAYS_TERMINATION"].notna()
    ).astype(int)

    # interest proxy (rate ≈ annuity / credit)
    prev["interest_rate_proxy"] = (
        prev["AMT_ANNUITY"] / prev["AMT_CREDIT"]
    )

    prev["interest_rate_proxy"] = prev["interest_rate_proxy"].replace(
        [np.inf, -np.inf], np.nan
    )

    prev_agg = (
        prev
        .groupby("SK_ID_CURR")
        .agg(
            # approval behaviour
            prev_num_approved=("is_prev_approved", "sum"),
            prev_num_rejected=("is_prev_rejected", "sum"),

            # repayment behaviour
            prev_early_repayment_count=("is_early_repayment", "sum"),

            # loan sizing
            avg_prev_amt_requested=("AMT_APPLICATION", "mean"),
            avg_prev_amt_credit=("AMT_CREDIT", "mean"),

            # pricing / risk
            avg_prev_interest_rate=("interest_rate_proxy", "mean"),
        )
        .reset_index()
    )

    # binary risk flags (very helpful for tree models)
    prev_agg["has_prev_rejection"] = (prev_agg["prev_num_rejected"] > 0).astype(int)
    prev_agg["has_early_repayment"] = (prev_agg["prev_early_repayment_count"] > 0).astype(int)

    return prev_agg

#Credit Card Balance
def build_credit_card_features(cc_path):
    cc = pd.read_csv(cc_path)

    cc_agg = (
        cc
        .groupby("SK_ID_CURR")
        .agg(
            avg_cc_mean_balance=("AMT_BALANCE", "mean"),
            avg_cc_max_limit_used=("AMT_CREDIT_LIMIT_ACTUAL", "max"),
            avg_cc_mean_drawings_atm_current=("AMT_DRAWINGS_ATM_CURRENT", "mean"),
            avg_cc_amt_payment_total_current=("AMT_PAYMENT_TOTAL_CURRENT", "mean"),
            max_cc_sk_dpd=("SK_DPD", "max"),
            max_cc_sk_dpd_def=("SK_DPD_DEF", "max"),
        )
        .reset_index()
    )
    cc_agg['max_cc_sk_dpd_def']=(cc_agg['max_cc_sk_dpd_def']>0).astype(int)
    cc_agg['HAS_CC_DELINQUENCY'] = (cc_agg['max_cc_sk_dpd'] > 0).astype(int)

    cc_agg['CC_PAYMENT_RATIO'] = (
    cc_agg['avg_cc_amt_payment_total_current'] /(cc_agg['avg_cc_mean_balance'] + 1))

    cc_agg['NO_CC_PAYMENT_FLAG'] = (cc_agg['avg_cc_amt_payment_total_current'] == 0).astype(int)
    cc_agg.drop(columns=['max_cc_sk_dpd','avg_cc_amt_payment_total_current', 'avg_cc_mean_balance'], inplace=True, errors='ignore')
    return cc_agg


# Master Feature Builder
def build_all_features(paths: dict):

    app_f= build_application_features(paths["application_train"])
    bureau_f = build_bureau_features(paths["bureau"], paths["bureau_balance"])
    pos_f = build_pos_features(paths["pos"])
    inst_f = build_installment_features(paths["installments"])
    prev_f = build_previous_application_features(paths["previous_application"])
    credit_f= build_credit_card_features(paths["credit_card"])

    features = (
        app_f
        .merge(bureau_f, on="SK_ID_CURR", how="left")
        .merge(pos_f, on="SK_ID_CURR", how="left")
        .merge(inst_f, on="SK_ID_CURR", how="left")
        .merge(prev_f, on="SK_ID_CURR", how="left")
        .merge(credit_f, on="SK_ID_CURR", how="left")
    )

    features['DELINQUENCY_SEVERITY'] = features['HAS_CC_DELINQUENCY'] * features['inst_mean_delay_days']
    return features