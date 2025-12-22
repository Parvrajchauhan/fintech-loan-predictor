from fastapi import FastAPI
from api.routes_class import router as classification_router
from api.routes_reg import router as regression_router

app = FastAPI(
    title="Loan Risk & Amount Prediction API",
    version="1.0.0",
    description=(
        "Provides default risk classification and loan amount regression "
        "using MLflow-registered models"
    ),
)

# Routers
app.include_router(
    classification_router,
    prefix="/classification",
    tags=["Default Risk"]
)

app.include_router(
    regression_router,
    prefix="/regression",
    tags=["Loan Amount"]
)

# Health check
@app.get("/health")
def health_check():
    return {"status": "ok"}
