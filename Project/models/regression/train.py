# models/regression/train.py

import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from mlflow.tracking import MlflowClient

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

    "Credit_to_Income_Ratio",
    "Annuity_to_Income_Ratio",

    # POS (light usage signals)
    "pos_num_loans",
    "pos_mean_cnt_instalment",

    # Previous applications
    "avg_prev_amt_requested",
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


TARGET = "AMT_CREDIT"

def load_data(df:pd.DataFrame):
    X = df[final_features_regression]
    y = np.log1p(df[TARGET])
    cat_cols= [col for col in final_features_regression if col in CAT_UNKNOWN]
    for col in cat_cols:
        X[col] = X[col].fillna("Unknown")
        X[col] = LabelEncoder().fit_transform(X[col])
    return X, y


def build_model():
    model = LGBMRegressor(
        objective="regression",      
        metric="mae",
        learning_rate=0.06,
        num_leaves=128,
        max_depth=-1,
        n_estimators=800,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
    )
    return model


def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = build_model()
    model.fit(X_train, y_train,eval_set=[(X_val, y_val)])

    val_preds = model.predict(X_val)
    mae = np.mean(np.abs(np.expm1(val_preds) - np.expm1(y_val)))
    rmse = np.sqrt(np.mean((np.expm1(val_preds) - np.expm1(y_val)) ** 2))

    return model, mae, rmse




if __name__ == "__main__":
    mlflow.set_experiment("loan_amount_regression_prob")
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")

    with mlflow.start_run():
        df = build_all_features(paths)
        stats = fit_imputer(df)
        df = apply_imputation(df, stats)
        X, y = load_data(df)

        model, val_mae, val_rmse = train_model(X, y)

        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_param("model_type", "LightGBM_Regressor")
        mlflow.log_param("n_estimators", model.n_estimators)
        for param_name, param_value in model.get_params().items():
            if param_value is not None:
                mlflow.log_param(param_name, param_value)

        mlflow.log_param("features_count", X.shape[1])
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name="loan_amount_regressor",
        )

        print(f"Training complete | MAE: {val_mae:.2f} | RMSE: {val_rmse:.2f}")
        client = MlflowClient()

        model_name = "loan_amount_regressor"
        latest_version = client.get_latest_versions(
            model_name, stages=["None"])[0].version
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            alias="production",

        )
