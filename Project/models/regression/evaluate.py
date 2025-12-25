import numpy as np
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.lightgbm
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "regression"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)



def evaluate(model, X, y):
    preds = model.predict(X)

    mae = np.mean(np.abs(np.expm1(preds) - np.expm1(y)))
    rmse = np.sqrt(np.mean((np.expm1(preds) - np.expm1(y)) ** 2))
    mape = np.mean(np.abs((np.expm1(y) - np.expm1(preds)) / np.expm1(y))) * 100

    residuals = y - preds

    return mae, rmse, residuals,mape


def segment_analysis(model,X,y,mae):
    # Segment analysis
    loan_sizes = np.expm1(y) 
    list=[]
    preds=model.predict(X)
    errors = np.abs(np.expm1(preds) - np.expm1(y))
    quartiles = np.percentile(loan_sizes, [25, 50, 75])
    print("MAPE by loan size quartile:")
    for i, (low, high) in enumerate([(0, quartiles[0]),(quartiles[0], quartiles[1]),
                                     (quartiles[1], quartiles[2]),(quartiles[2], np.inf)]):
            mask = (loan_sizes >= low) & (loan_sizes < high)
            if mask.sum() > 0:
                 segment_mape = np.mean(errors[mask] / loan_sizes[mask]) * 100
                 list.append(segment_mape)
                 print(f"Q{i+1} (${low:,.0f}-${high:,.0f}): {segment_mape:.1f}%")
    mlflow.log_metric("Q1",list[0])
    mlflow.log_metric("Q2",list[1])
    mlflow.log_metric("Q3",list[2])
    mlflow.log_metric("Q4",list[3])
    
    # Cost of errors
    average_loan = np.mean(loan_sizes)
    print(f"\nBusiness Impact:")
    print(f"Average loan: ${average_loan:,.0f}")
    print(f"Average error: ${mae:,.0f}")
    print(f"Error as % of loan: {mae/average_loan*100:.1f}%")
    mlflow.log_metric("Error as percentageof loan",mae/average_loan*100)
    portfolio_value = 10_000_000
    avg_error_rate = mae / average_loan
    print(f"\nFor ${portfolio_value:,.0f} portfolio:")
    print(f"Expected prediction error: ${portfolio_value * avg_error_rate:,.0f}")

def plot_residuals(residuals):
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=50)
    plt.title("Residual Distribution")
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "residuals.png")
    plt.close()

def eval(X,y):
    mlflow.set_experiment("loan_amount_regression_prob")
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")

    model = mlflow.lightgbm.load_model("models:/loan_amount_regressor@production")

    mae, rmse, residuals,mape = evaluate(model, X, y)

    plot_residuals(residuals)

    mlflow.log_metric("eval_mae", mae)
    mlflow.log_metric("eval_rmse", rmse)
    mlflow.log_metric("eval_mape", mape)
    mlflow.log_artifact(
    str(ARTIFACTS_DIR / "residuals.png")
)
    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")
    segment_analysis(model,X,y,mae)

    
