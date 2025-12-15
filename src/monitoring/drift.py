# src/monitoring/drift.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
from sklearn.metrics import accuracy_score


# Lokasi file metrics/log
THIS_DIR = Path(__file__).resolve().parent
METRICS_STORE = THIS_DIR / "metrics_store.json"
PREDICTIONS_LOG = THIS_DIR / "predictions_log.jsonl"


# ========= UTIL: I/O JSON =========

def _load_metrics() -> Dict[str, Any]:
    if METRICS_STORE.exists():
        return json.loads(METRICS_STORE.read_text())
    # Struktur default
    return {
        "reference": {
            "class_distribution": {},
            "mean_confidence": None,
        },
        "production": {
            "last_window_size": 0,
            "class_distribution": {},
            "mean_confidence": None,
            "accuracy": None,
        },
        "drift": {
            "class_dist_diff": None,
            "confidence_drop": None,
        },
    }


def _save_metrics(metrics: Dict[str, Any]) -> None:
    METRICS_STORE.write_text(json.dumps(metrics, indent=2))


# ========= PUBLIC API UNTUK FASTAPI =========

def log_prediction(
    *,
    model_name: str,
    y_pred: str,
    y_true: Optional[str],
    confidence: Optional[float],
) -> None:
    """
    Dipanggil dari FastAPI setiap kali ada prediksi.
    Simpan log sebagai JSON lines.
    y_true boleh None (kalau belum ada label ground truth).
    """
    record = {
        "model_name": model_name,
        "y_pred": y_pred,
        "y_true": y_true,
        "confidence": confidence,
    }
    with PREDICTIONS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def compute_drift(window_size: int = 200) -> Dict[str, Any]:
    """
    Baca log prediksi terbaru dan hitung:
      - distribusi kelas prediksi
      - rata-rata confidence
      - jika y_true tersedia, hitung akurasi
      - bandingkan dengan reference (jika sudah di-set)

    Ini versi sangat sederhana dari konsep monitoring data/model drift.[web:123][web:128]
    """
    metrics = _load_metrics()

    if not PREDICTIONS_LOG.exists():
        _save_metrics(metrics)
        return metrics

    # Ambil window terakhir dari log
    lines: List[str] = PREDICTIONS_LOG.read_text().strip().split("\n")
    if not lines or lines == [""]:
        _save_metrics(metrics)
        return metrics

    window_records = [json.loads(l) for l in lines[-window_size:]]
    y_preds = [r["y_pred"] for r in window_records]
    confidences = [r["confidence"] for r in window_records if r["confidence"] is not None]
    y_trues = [r["y_true"] for r in window_records if r["y_true"] is not None]

    # Distribusi kelas sederhana
    class_dist: Dict[str, float] = {}
    total = len(y_preds)
    for c in y_preds:
        class_dist[c] = class_dist.get(c, 0) + 1
    if total > 0:
        for k in class_dist:
            class_dist[k] /= total

    mean_conf = float(np.mean(confidences)) if confidences else None
    acc = accuracy_score(y_trues, y_preds[: len(y_trues)]) if y_trues else None

    # Update bagian production
    metrics["production"]["last_window_size"] = total
    metrics["production"]["class_distribution"] = class_dist
    metrics["production"]["mean_confidence"] = mean_conf
    metrics["production"]["accuracy"] = float(acc) if acc is not None else None

    # Hitung indikasi drift sederhana:
    #  - beda L1 antara distribusi kelas production vs reference
    #  - penurunan mean confidence vs reference
    ref_dist = metrics["reference"].get("class_distribution") or {}
    ref_conf = metrics["reference"].get("mean_confidence")

    if ref_dist and class_dist:
        # normalisasi gabungan key
        all_classes = set(ref_dist.keys()) | set(class_dist.keys())
        diff = 0.0
        for c in all_classes:
            p_ref = ref_dist.get(c, 0.0)
            p_cur = class_dist.get(c, 0.0)
            diff += abs(p_ref - p_cur)
        metrics["drift"]["class_dist_diff"] = diff  # semakin besar, indikasi drift[web:129]
    else:
        metrics["drift"]["class_dist_diff"] = None

    if ref_conf is not None and mean_conf is not None:
        metrics["drift"]["confidence_drop"] = float(ref_conf - mean_conf)
    else:
        metrics["drift"]["confidence_drop"] = None

    _save_metrics(metrics)
    return metrics


def set_reference_from_training(
    class_distribution: Dict[str, float],
    mean_confidence: Optional[float] = None,
) -> None:
    """
    Dipanggil sekali dari pipeline training (offline) untuk menyimpan baseline reference.[web:122][web:131]
    Misalnya class_distribution dihitung dari data validasi.
    """
    metrics = _load_metrics()
    metrics["reference"]["class_distribution"] = class_distribution
    metrics["reference"]["mean_confidence"] = mean_confidence
    _save_metrics(metrics)


def print_latest_metrics() -> None:
    """
    Dipakai CLI: python cli/main.py monitor show
    """
    metrics = _load_metrics()
    print(json.dumps(metrics, indent=2))
