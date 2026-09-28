from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_transaction


app = FastAPI(
    title="Fraud Detection API",
    description="API for predicting fraudulent transactions",
    version="1.0.0",
)


class Transaction(BaseModel):
    transaction_amount: float
    transaction_type: str
    payment_mode: str
    device_type: str
    device_location: str
    account_age_days: int
    transaction_hour: int
    previous_failed_attempts: int
    avg_transaction_amount: float
    is_international: int
    ip_risk_score: float
    login_attempts_last_24h: int
    hist_mean: float
    hist_std: float
    amount_deviation: float
    user_transaction_count: int
    amount_ratio: float
    risk_login_interaction: float


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

    return result