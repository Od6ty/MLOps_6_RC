# src/serving/api.py
from __future__ import annotations

import os
import random
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from .schemas import PredictionResponse
from .inference import MODELS_DIR, load_artifacts, predict_with_model
from src.monitoring.drift import log_prediction, compute_drift  # src/monitoring/drift.py

app = FastAPI(title="Flower Classification API (SVM & Naive Bayes)")

# Konfigurasi canary (misal 0.1 = 10% request juga dievaluasi di canary)
CANARY_RATIO = float(os.getenv("CANARY_RATIO", "0.1"))

PROD_DIR = MODELS_DIR / "production"
CANARY_DIR = MODELS_DIR / "canary"

# Load artifacts saat startup
prod_artifacts = None
canary_artifacts = None


@app.on_event("startup")
def load_models_on_startup():
    global prod_artifacts, canary_artifacts
    if PROD_DIR.exists():
        prod_artifacts = load_artifacts(PROD_DIR)
        print("[startup] Loaded PRODUCTION models from", PROD_DIR)
    else:
        print("[startup] PRODUCTION dir not found:", PROD_DIR)

    if CANARY_DIR.exists():
        canary_artifacts = load_artifacts(CANARY_DIR)
        print("[startup] Loaded CANARY models from", CANARY_DIR)
    else:
        print("[startup] CANARY dir not found:", CANARY_DIR)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Endpoint utama prediksi:
      - Production: SVM (models/production/)
      - Canary (shadow): Naive Bayes (models/canary/) dengan CANARY_RATIO
    """
    if prod_artifacts is None:
        return JSONResponse(
            status_code=500,
            content={"detail": "Production model not loaded"},
        )

    image_bytes = await file.read()

    # --- PRODUCTION: SVM ---
    label_prod, conf_prod = predict_with_model(
        artifacts=prod_artifacts,
        image_bytes=image_bytes,
        model_type="svm",
    )

    # Log ke monitoring (y_true None, karena belum ada label ground truth)
    log_prediction(
        model_name="svm_prod",
        y_pred=label_prod,
        y_true=None,
        confidence=conf_prod,
    )

    # --- CANARY (SHADOW) ---
    if canary_artifacts is not None and random.random() < CANARY_RATIO:
        label_canary, conf_canary = predict_with_model(
            artifacts=canary_artifacts,
            image_bytes=image_bytes,
            model_type="nb",
        )
        log_prediction(
            model_name="nb_canary",
            y_pred=label_canary,
            y_true=None,
            confidence=conf_canary,
        )

    return PredictionResponse(
        model_used="svm_prod",
        label=label_prod,
        confidence=conf_prod,
    )


@app.get("/monitoring/drift")
def get_drift_metrics():
    """
    Endpoint untuk mengecek metrik drift (baca dari metrics_store.json). [web:122][web:130]
    """
    metrics = compute_drift()
    return metrics
