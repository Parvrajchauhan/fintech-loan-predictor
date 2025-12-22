# models/classification/evaluate.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
import mlflow
import mlflow.xgboost
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)
from Project.features.build_features import build_all_features
from Project.features.imputation import NUM_MEDIAN, NUM_ZERO, CAT_UNKNOWN, fit_imputer, apply_imputation
from Project.models.classification.train import load_data

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

TARGET = "TARGET"
ID_COL = "SK_ID_CURR"
THRESHOLDS = np.arange(0.45, 0.3, 0.4)
MODEL_URI = "models:/loan_default_classifier@production"


def ks_statistic(y_true, y_prob):
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df = df.sort_values("p")
    df["cum_good"] = (df["y"] == 0).cumsum() / (df["y"] == 0).sum()
    df["cum_bad"] = (df["y"] == 1).cumsum() / (df["y"] == 1).sum()
    return np.max(np.abs(df["cum_bad"] - df["cum_good"]))




if __name__ == "__main__":
    
    mlflow.set_experiment("loan_default_classification")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    df = build_all_features(paths)
    stats = fit_imputer(df)
    df = apply_imputation(df, stats)
    X, y = load_data(df)

    X, X_val, y, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = mlflow.xgboost.load_model(MODEL_URI)
    probs = model.predict_proba(X)[:, 1]

    roc_auc = roc_auc_score(y, probs)
    pr_auc = average_precision_score(y, probs)
    ks = ks_statistic(y, probs)

    THRESHOLD = 0.45
    preds = (probs >= THRESHOLD).astype(int)

    recall = recall_score(y, preds)
    precision = precision_score(y, preds)
    f1 = f1_score(y, preds)

    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("f1", f1)

    mlflow.log_metric("eval_roc_auc", roc_auc)
    mlflow.log_metric("eval_pr_auc", pr_auc)
    mlflow.log_metric("eval_ks", ks)
    
    low_cutoff = np.quantile(probs, 0.30)
    high_cutoff = np.quantile(probs, 0.70)

    cutoffs = {
        "low_risk_max_pd": float(low_cutoff),
        "medium_risk_max_pd": float(high_cutoff),
    }
    
    pd.Series(cutoffs).to_json("risk_cutoffs.json")
    mlflow.log_artifact("risk_cutoffs.json")

    print("\n=== MODEL EVALUATION SUMMARY ===")
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}")
    print(f"KS      : {ks:.4f}")
    print(f"Recall@{THRESHOLD}: {recall:.4f}")

    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()

    approved = preds == 0
    rejected = preds == 1

    approval_rate = approved.mean()
    bad_rate_approved = y[approved].mean()

    print(approval_rate)
    print(bad_rate_approved)

