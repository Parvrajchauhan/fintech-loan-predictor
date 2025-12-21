
import pandas as pd
import mlflow.lightgbm


MODEL_URI = "models:/loan_default_classifier@production"
THRESHOLD = 0.07


def load_model():
    return mlflow.lightgbm.load_model(MODEL_URI)


def predict(df: pd.DataFrame):
    model = load_model()
    prob = model.predict_proba(df)[:, 1][0]
    decision = int(prob >= THRESHOLD)

    return {
        "default_probability": float(prob),
        "decision": decision,  # 1 = reject / risky, 0 = accept
    }


if __name__ == "__main__":
    sample = pd.read_csv("data/sample_classification_row.csv")
    result = predict(sample)
    print(result)
