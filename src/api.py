from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.predict import predict_transaction


app = FastAPI(
    title="Fraud Detection API",
    description="API for predicting fraudulent transactions",
    version="1.0.0",
)


class Transaction(BaseModel):
    transaction_amount: float = Field(gt=0)
    transaction_type: str
    payment_mode: str
    device_type: str
    device_location: str
    account_age_days: int = Field(ge=0)
    transaction_hour: int = Field(ge=0, le=23)
    previous_failed_attempts: int = Field(ge=0)
    avg_transaction_amount: float = Field(ge=0)
    is_international: int = Field(ge=0, le=1)
    ip_risk_score: float = Field(ge=0, le=100)
    login_attempts_last_24h: int = Field(ge=0)
    hist_mean: float = Field(ge=0)
    hist_std: float = Field(ge=0)
    amount_deviation: float
    user_transaction_count: int = Field(ge=0)
    amount_ratio: float = Field(ge=0)
    risk_login_interaction: float = Field(ge=0)

@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running"
    }


@app.post("/predict")
def predict(transaction: Transaction):
    result = predict_transaction(
        transaction.model_dump()
    )

    result["status"] = (
        "fraud"
        if result["prediction"] == 1
        else "legitimate"
    )

    return result
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "logistic_regression",
    }