# src/serving/inference.py
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Tuple, Any, Dict

import joblib
import numpy as np
from PIL import Image


MODELS_DIR = Path(__file__).resolve().parents[1].parents[0] / "models"


def load_artifacts(model_dir: Path) -> Dict[str, Any]:
    """
    Load scaler, classes, SVM & Naive Bayes dari folder:
      models/production/ atau models/canary/
    """
    scaler = joblib.load(model_dir / "scaler.joblib")
    classes = joblib.load(model_dir / "classes.joblib")
    svm_clf = joblib.load(model_dir / "svm_canary.joblib")  # nama file sama, tapi lokasinya beda
    nb_clf = joblib.load(model_dir / "nb_canary.joblib")
    return {
        "scaler": scaler,
        "classes": classes,
        "svm": svm_clf,
        "nb": nb_clf,
    }


def preprocess_image_bytes(
    image_bytes: bytes,
    image_size: Tuple[int, int] = (128, 128),
) -> np.ndarray:
    """
    Preprocess gambar dari bytes -> array 1D (flatten) sama seperti di training.[file:4]
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize(image_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(1, -1)  # (1, D)


def predict_with_model(
    artifacts: Dict[str, Any],
    image_bytes: bytes,
    model_type: str = "svm",
) -> Tuple[str, float | None]:
    """
    Jalankan inference dengan SVM atau Naive Bayes.
    Return: (label_prediksi, confidence)
    """
    X = preprocess_image_bytes(image_bytes)
    X_scaled = artifacts["scaler"].transform(X)

    if model_type == "svm":
        clf = artifacts["svm"]
    elif model_type == "nb":
        clf = artifacts["nb"]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Prediksi label
    y_proba = None
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_scaled)[0]
        idx = int(np.argmax(y_proba))
        label = str(clf.classes_[idx])
        conf = float(y_proba[idx])
    else:
        # fallback: hanya predict label
        y_pred = clf.predict(X_scaled)[0]
        label = str(y_pred)
        conf = None

    return label, conf
