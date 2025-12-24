import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

from Project.models.regression.feature_list import final_features_regression
from Project.db.repositories import load_dataframe


EXPERIMENT_NAME = "loan_amount_regression_prob"
FEATURE_TABLE = "loan_regression_system_data"
MODEL_NAME = "loan_amount_regressor"


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    encoders = {}

    for col in cat_cols:
        df[col] = df[col].astype(str)
        df[col] = df[col].fillna(df[col].mode()[0])
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    joblib.dump(encoders, "label_encoders_reg.pkl")
    mlflow.log_artifact("label_encoders_reg.pkl")

    return df


def main():

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():

        # -------------------------
        # Load feature snapshot
        # -------------------------
        df = load_dataframe(
            table_name=FEATURE_TABLE,
            columns=final_features_regression + ["target_loan_amount"],
        )

        X = df[final_features_regression]
        y = df["target_loan_amount"]

        # -------------------------
        # Encode categoricals
        # -------------------------
        X = encode_categoricals(X)

        # -------------------------
        # Train / validation split
        # -------------------------
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # -------------------------
        # Model
        # -------------------------
        model = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )

        model.fit(X_train, y_train)

        # -------------------------
        # Evaluation
        # -------------------------
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)

        mlflow.log_metric("val_mae", mae)
        mlflow.log_metric("val_r2", r2)

        # -------------------------
        # Log model + features
        # -------------------------
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        joblib.dump(final_features_regression, "model_features_reg.pkl")
        mlflow.log_artifact("model_features_reg.pkl")

        print("Regression training completed successfully.")


if __name__ == "__main__":
    main()
