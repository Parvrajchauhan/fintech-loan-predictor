
import pandas as pd
import mlflow.xgboost
from sklearn.preprocessing import LabelEncoder
import json
from Project.models.classification.feature_list import final_features_classification
from Project.db.repositories import load_dataframe
MODEL_URI = "models:/loan_default_classifier@production"
THRESHOLD = 0.07


def load_model():
    return mlflow.xgboost.load_model(MODEL_URI)



def load_risk_cutoffs():
    with open("risk_cutoffs.json", "r") as f:
        return json.load(f)


def assign_risk_segment(prob, cutoffs):
    if prob <= cutoffs["low_risk_max_pd"]:
        return "Low Risk"
    elif prob <= cutoffs["medium_risk_max_pd"]:
        return "Medium Risk"
    else:
        return "High Risk"



def predict(df: pd.DataFrame):
    model = load_model()
    cutoffs = load_risk_cutoffs()

    prob = model.predict_proba(df)[:, 1][0]
    decision = int(prob >= THRESHOLD)
    risk_segment = assign_risk_segment(prob, cutoffs)

    return {
        "default_probability": float(prob),
        "decision": decision,      
        "risk_segment": risk_segment
    }

if __name__ == "__main__":
    X= load_dataframe(
        "loan_classification_system_data",
        columns=final_features_classification,
        limit=1000
    )
    cat_cols = X.select_dtypes(include=['object','category']).columns
    for col in cat_cols:
        X[col] = X[col].astype(str)
        most_freq = X[col].mode()[0]
        X[col] = X[col].fillna(most_freq)
        le = LabelEncoder()
        le.fit(X[col])
        X[col] = le.transform(X[col])
    result = predict(X)

    print(result)
