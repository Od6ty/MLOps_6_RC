# src/models/registry.py
from pathlib import Path
import shutil
import json

ROOT_MODELS_DIR = Path(__file__).resolve().parents[1].parents[0] / "models"
PROD_DIR = ROOT_MODELS_DIR / "production"
CANARY_DIR = ROOT_MODELS_DIR / "canary"
REGISTRY_META = ROOT_MODELS_DIR / "registry_meta.json"


def _load_meta():
    if REGISTRY_META.exists():
        return json.loads(REGISTRY_META.read_text())
    return {
        "production_version": None,
        "canary_version": None,
    }


def _save_meta(meta):
    REGISTRY_META.write_text(json.dumps(meta, indent=2))


def set_canary_from_run(run_id: str) -> None:
    """
    Placeholder jika nanti pakai MLflow.
    Untuk sekarang, anggap model hasil train_new_model sudah ada di models/canary/.
    Di sini hanya update metadata version.
    """
    meta = _load_meta()
    meta["canary_version"] = run_id
    _save_meta(meta)
    print(f"[registry] Set canary version to run_id={run_id}")


def promote_canary_to_production() -> None:
    """
    Copy isi folder models/canary -> models/production.
    """
    PROD_DIR.mkdir(parents=True, exist_ok=True)
    CANARY_DIR.mkdir(parents=True, exist_ok=True)

    # Bersihkan production lama
    for p in PROD_DIR.glob("*"):
        if p.is_file():
            p.unlink()

    # Copy semua file dari canary
    for p in CANARY_DIR.glob("*"):
        if p.is_file():
            shutil.copy2(p, PROD_DIR / p.name)

    meta = _load_meta()
    meta["production_version"] = meta.get("canary_version", None)
    _save_meta(meta)

    print("[registry] Promoted CANARY -> PRODUCTION")


def print_status() -> None:
    meta = _load_meta()
    print("[registry] Status:")
    print("  production_version:", meta.get("production_version"))
    print("  canary_version    :", meta.get("canary_version"))
