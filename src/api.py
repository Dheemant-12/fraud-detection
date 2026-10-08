import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.prediction_service import predict
from src.config import MODEL_NAME, MODEL_VERSION

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=LOG_DIR / "predictions.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)
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
    result = predict(
        transaction.model_dump()
    )

    result["status"] = (
        "fraud"
        if result["prediction"] == 1
        else "legitimate"
    )

    result["model_version"] = MODEL_VERSION

    result["timestamp"] = datetime.now(
        timezone.utc
    ).isoformat()

    logger.info(
        "Prediction made: status=%s probability=%.4f",
        result["status"],
        result["fraud_probability"],
    )

    return result


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
    }
@app.get("/model-info")
def model_info():
    return {
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "task": "fraud_detection",
    }
    return {
        "model": "logistic_regression",
        "version": "1.0.0",
        "task": "fraud_detection",
    }
@app.post("/predict")
def predict_endpoint(transaction: Transaction):
    try:
        result = predict(transaction.model_dump())

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    result["status"] = (
        "fraud"
        if result["prediction"] == 1
        else "legitimate"
    )

    result["model_version"] = MODEL_VERSION
    result["timestamp"] = datetime.now(timezone.utc).isoformat()

    logger.info(
        "Prediction made: status=%s probability=%.4f",
        result["status"],
        result["fraud_probability"],
    )

    return result