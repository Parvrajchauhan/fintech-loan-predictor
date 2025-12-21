# models/classification/train.py

import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_curve,
    precision_recall_curve
)

from Project.features.build_features import build_all_features
from Project.features.imputation import NUM_MEDIAN, NUM_ZERO, CAT_UNKNOWN, fit_imputer, apply_imputation

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

paths = {
    "application_train": DATA_DIR / "application_train.csv",
    "bureau": DATA_DIR / "bureau.csv",
    "bureau_balance": DATA_DIR / "bureau_balance.csv",
    "pos": DATA_DIR / "POS_CASH_balance.csv",
    "installments": DATA_DIR / "installments_payments.csv",
    "previous_application": DATA_DIR / "previous_application.csv",
    "credit_card": DATA_DIR / "credit_card_balance.csv",
}


from mlflow.tracking import MlflowClient

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



TARGET = "TARGET"
ID_COL = "SK_ID_CURR"
MODEL_NAME = "loan_default_classifier"
THRESHOLD = 0.45


def load_data(df:pd.DataFrame):
    X = df[final_features_classification]
    y =df[TARGET]
    cat_cols= [col for col in final_features_classification if col in CAT_UNKNOWN]
    for col in cat_cols:
        X[col] = X[col].astype(str)
        most_freq = X[col].mode()[0]
        X[col] = X[col].fillna(most_freq)
        le = LabelEncoder()
        le.fit(X[col])
        X[col] = le.transform(X[col])
    return X, y


def build_model(scale_pos_weight):
    return XGBClassifier(
        colsample_bytree=0.8,
        learning_rate=0.018,
        max_depth=4,
        min_child_weight=5,
        n_estimators=2300,
        reg_alpha=1.0,
        reg_lambda=2.0,
        subsample=0.8,
        objective='binary:logistic',
        eval_metric='auc',
        base_score=0.5000,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )


if __name__ == "__main__":
    mlflow.set_experiment("loan_default_classification")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    with mlflow.start_run():
        df = build_all_features(paths)
        stats = fit_imputer(df)
        df = apply_imputation(df, stats)
        X, y = load_data(df)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = build_model(scale_pos_weight)
        model.fit(X_train, y_train,eval_set=[(X_val, y_val)],verbose=100)

        val_probs = model.predict_proba(X_val)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_val, val_probs)
        ks_scores = tpr - fpr
        threshold = 0.45

        y_pred = (val_probs >= threshold).astype(int)

        roc_auc = roc_auc_score(y_val, val_probs)
        pr_auc = average_precision_score(y_val, val_probs)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        accuracy = accuracy_score(y_val, y_pred)

        mlflow.log_param("n_estimators", model.n_estimators)
        for param_name, param_value in model.get_params().items():
            if param_value is not None:
                mlflow.log_param(param_name, param_value)

        mlflow.log_param("features_count", X.shape[1])

        # log metrics
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("ks_statistic", ks_scores.max())

        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        # promote to staging (alias-based)
        client = MlflowClient()
        latest_version = client.get_latest_versions(MODEL_NAME)[0].version

        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="production",
            version=latest_version,
        )

        print(f"Training complete |ROC-AUC: {roc_auc:.4f} | recall: {recall:.4f} | precision: {precision:.4f} | ks: {ks_scores.max():.4f}")
