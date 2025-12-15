# src/serving/__init__.py
from .api import app
from .inference import load_artifacts, predict_with_model
from .schemas import PredictionResponse

__all__ = ["app", "load_artifacts", "predict_with_model", "PredictionResponse"]

