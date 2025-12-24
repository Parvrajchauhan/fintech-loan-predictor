# models/regression/train.py

import pandas as pd
import numpy as np
import mlflow
import joblib
import mlflow.lightgbm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from mlflow.tracking import MlflowClient
from Project.models.regression.feature_list import final_features_regression
from Project.features.build_features import build_all_features
from Project.db.repositories import save_dataframe
from Project.features.imputation import NUM_MEDIAN, NUM_ZERO, CAT_UNKNOWN, fit_imputer, apply_imputation

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "regression"
DATA_DIR = PROJECT_ROOT / "data" / "raw"
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


joblib.dump(
    final_features_regression,
    ARTIFACTS_DIR / "model_features_reg.pkl"
)

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
        n_estimators=820,
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
    mape = np.mean(np.abs((np.expm1(y_val) - np.expm1(val_preds)) / np.expm1(y_val))) * 100

    return model, mae, rmse,mape




if __name__ == "__main__":
    mlflow.set_experiment("loan_amount_regression_prob")
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")

    with mlflow.start_run():
        mlflow.log_artifact(str(ARTIFACTS_DIR / "model_features_reg.pkl"))
        df = build_all_features(paths)
        stats = fit_imputer(df)
        df = apply_imputation(df, stats)
        X, y = load_data(df)
        
        feature_snapshot=df.drop(columns=[TARGET]).sample(2000, random_state=42)
        save_dataframe(
            feature_snapshot,
            table_name="loan_regression_system_data"
        )

        model, val_mae, val_rmse,val_mape= train_model(X, y)

        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_rmse", val_rmse)
        
        mlflow.log_metric("MAPE",val_mape)
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

        print(f"Training complete | MAE: {val_mae:.2f} | RMSE: {val_rmse:.2f}| MAPE:{val_mape:.2f}")
        
        client = MlflowClient()

        model_name = "loan_amount_regressor"
        latest_version = client.get_latest_versions(
            model_name, stages=["None"])[0].version
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            alias="production",
        )
