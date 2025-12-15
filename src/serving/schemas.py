# src/serving/schemas.py
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    model_used: str
    label: str
    confidence: float | None = None
