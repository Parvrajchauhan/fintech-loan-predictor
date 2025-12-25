# models/classification/train.py

import pandas as pd
import numpy as np
import mlflow
import joblib
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
from Project.db.repositories import save_dataframe
from Project.models.classification.feature_list import final_features_classification
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "classification"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


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

joblib.dump(final_features_classification, ARTIFACTS_DIR /"model_features_class.pkl")


TARGET = "TARGET"
ID_COL = "SK_ID_CURR"
MODEL_NAME = "loan_default_classifier"
THRESHOLD = 0.45

def encode(X):
    cat_cols= [col for col in final_features_classification if col in CAT_UNKNOWN]
    for col in cat_cols:
        X[col] = X[col].astype(str)
        most_freq = X[col].mode()[0]
        X[col] = X[col].fillna(most_freq)
        le = LabelEncoder()
        le.fit(X[col])
        X[col] = le.transform(X[col])
    return X


def load_data(df:pd.DataFrame):
    X = df[final_features_classification]
    y =df[TARGET]
    X=encode(X)
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


def build():
    df = build_all_features(paths)
    stats = fit_imputer(df)
    df = apply_imputation(df, stats)
    feature_snapshot=df[final_features_classification].sample(2000, random_state=42)
    save_dataframe(
        feature_snapshot,
        table_name="loan_classification_system_data"
    )
    return df


def train():
    mlflow.set_experiment("loan_default_classification")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    with mlflow.start_run():
        mlflow.log_artifact(str(ARTIFACTS_DIR /"model_features_class.pkl"))
        
        df=build()
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
        return X_val,y_val
