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

    overdue_status = ["1", "2", "3", "4", "5"]
    bureau_balance["is_overdue"] = bureau_balance["STATUS"].isin(overdue_status)

    bb_agg = (
        bureau_balance
        .groupby("SK_ID_BUREAU")["is_overdue"]
        .sum()
        .reset_index()
        .rename(columns={"is_overdue": "bureau_overdue_months"})
    )

    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    bureau["bureau_overdue_months"] = bureau["bureau_overdue_months"].fillna(0)

    bureau_final = (
        bureau
        .groupby("SK_ID_CURR")
        .agg(
            total_active_bureau_loans=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
            bureau_overdue_months=("bureau_overdue_months", "sum"),
        )
        .reset_index()
    )
    bureau_final['bureau_overdue_months']=(bureau_final['bureau_overdue_months']>0).astype(int)
    bureau_final['total_active_bureau_loans']=(bureau_final['total_active_bureau_loans']>0).astype(int)

    return bureau_final


# POS Cash Balance
def build_pos_features(pos_path):
    pos = pd.read_csv(pos_path)

    pos_agg = (
        pos
        .groupby("SK_ID_CURR")
        .agg(
            pos_num_loans=("SK_ID_PREV", "nunique"),
            pos_mean_cnt_instalment=("CNT_INSTALMENT", "mean"),
        )
        .reset_index()
    )

    return pos_agg

# Installments Payments
def build_installment_features(installments_path):
    inst = pd.read_csv(installments_path)

    inst["payment_diff"] = inst["AMT_PAYMENT"] - inst["AMT_INSTALMENT"]
    inst["is_late"] = inst["DAYS_ENTRY_PAYMENT"] > inst["DAYS_INSTALMENT"]

    inst_agg = (
        inst
        .groupby("SK_ID_CURR")
        .agg(
            inst_mean_payment_delay=("payment_diff", "mean"),
            inst_mean_payment_rate=("is_late", "mean"),
        )
        .reset_index()
    )

    return inst_agg

# Previous Applications

def build_previous_application_features(prev_path):
    prev = pd.read_csv(prev_path)
    prev["status"] = prev["NAME_CONTRACT_STATUS"].astype(str).str.lower()
    prev["is_prev_approved"] = prev["status"].str.contains("approved", na=False).astype(int)
    prev["is_prev_rejected"] = prev["status"].str.contains("refus|reject|canceled", na=False).astype(int)

    prev_agg = (
        prev
        .groupby("SK_ID_CURR")
        .agg(
            prev_num_approved=("is_prev_approved", "sum"),
            prev_num_rejected=("is_prev_rejected", "sum"),
            avg_prev_amt_requested=("AMT_APPLICATION", "mean"),
        )
        .reset_index()
    )

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
    cc_agg.drop(columns=['max_cc_sk_dpd','avg_cc_amt_payment_total_current', 'avg_cc_amt_balance'], inplace=True, errors='ignore')
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

    features['DELINQUENCY_SEVERITY'] = features['HAS_CC_DELINQUENCY'] * features['inst_mean_payment_delay']
    return features