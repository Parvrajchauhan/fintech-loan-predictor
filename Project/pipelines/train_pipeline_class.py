import pandas as pd
import joblib
import mlflow
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from Project.models.classification.feature_list import final_features_classification
from Project.db.repositories import save_dataframe
from Project.db.repositories import load_dataframe


EXPERIMENT_NAME = "loan_default_classification"
FEATURE_TABLE = "loan_classification_system_data"
MODEL_NAME = "loan_default_classifier"


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    encoders = {}

    for col in cat_cols:
        df[col] = df[col].astype(str)
        df[col] = df[col].fillna(df[col].mode()[0])
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    joblib.dump(encoders, "label_encoders.pkl")
    mlflow.log_artifact("label_encoders.pkl")

    return df


def main():

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():

        # -------------------------
        # Load feature snapshot
        # -------------------------
        df = load_dataframe(
            table_name=FEATURE_TABLE,
            columns=final_features_classification + ["target_default"],
        )

        X = df[final_features_classification]
        y = df["target_default"]

        # -------------------------
        # Encode categoricals
        # -------------------------
        X = encode_categoricals(X)

        # -------------------------
        # Train / validation split
        # -------------------------
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # -------------------------
        # Model
        # -------------------------
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )

        model.fit(X_train, y_train)

        # -------------------------
        # Log model + features
        # -------------------------
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        joblib.dump(final_features_classification, "model_features_class.pkl")
        mlflow.log_artifact("model_features_class.pkl")

        # -------------------------
        # Metrics
        # -------------------------
        val_auc = model.score(X_val, y_val)
        mlflow.log_metric("val_accuracy", val_auc)

        print("Training completed successfully.")


if __name__ == "__main__":
    main()
