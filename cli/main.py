# cli/main.py
import argparse
import subprocess
import sys
from pathlib import Path

# ------- Helper paths -------
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
MODELS_DIR = ROOT_DIR / "models"
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"


# ------- TRAIN COMMANDS -------

def cmd_train(args: argparse.Namespace) -> None:
    """
    Entry untuk: python cli/main.py train new
    """
    sys.path.append(str(SRC_DIR))
    from models import train as train_module  # src/models/train.py

    if args.mode == "new":
        train_module.train_new_model(
            data_raw_dir=DATA_RAW_DIR,
            data_processed_dir=DATA_PROCESSED_DIR,
            output_dir=MODELS_DIR / "canary",
        )
    elif args.mode == "pipeline":
        train_module.train_pipeline_with_registry(
            data_raw_dir=DATA_RAW_DIR,
            data_processed_dir=DATA_PROCESSED_DIR,
        )
    else:
        print(f"[train] Mode tidak dikenal: {args.mode}")
        sys.exit(1)


# ------- REGISTRY COMMANDS -------

def cmd_registry(args: argparse.Namespace) -> None:
    """
    Registry sederhana: set model canary / promote ke production.
    Implementasi detail di src/models/registry.py
    """
    sys.path.append(str(SRC_DIR))
    from models import registry as reg  # src/models/registry.py

    if args.action == "set-canary":
        reg.set_canary_from_run(run_id=args.run_id)
    elif args.action == "promote-canary":
        reg.promote_canary_to_production()
    elif args.action == "status":
        reg.print_status()
    else:
        print(f"[registry] Action tidak dikenal: {args.action}")
        sys.exit(1)


# ------- SERVE COMMANDS -------

def cmd_serve(args: argparse.Namespace) -> None:
    """
    Jalankan FastAPI + Uvicorn.
    Contoh:
      python cli/main.py serve start
      python cli/main.py serve start --port 9000
    """
    host = args.host
    port = args.port

    # Panggil uvicorn sebagai subprocess
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.serving.api:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    print(f"[serve] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ------- MONITORING COMMANDS -------

def cmd_monitor(args: argparse.Namespace) -> None:
    """
    Placeholder untuk operasi monitoring (drift, metrics, dsb).
    Implementasi detail di src/monitoring/drift.py.
    """
    sys.path.append(str(SRC_DIR))
    from monitoring import drift  # src/monitoring/drift.py

    if args.action == "compute-drift":
        drift.compute_drift()
    elif args.action == "show":
        drift.print_latest_metrics()
    else:
        print(f"[monitor] Action tidak dikenal: {args.action}")
        sys.exit(1)


# ------- MAIN ARGPARSE SETUP -------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI untuk MLOps klasifikasi bunga (training, registry, serving, monitoring)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    p_train = subparsers.add_parser("train", help="Training model")
    p_train.add_argument(
        "mode",
        choices=["new", "pipeline"],
        help="new: train canary baru, pipeline: train + registry",
    )
    p_train.set_defaults(func=cmd_train)

    # --- registry ---
    p_reg = subparsers.add_parser("registry", help="Kelola model registry (canary/production)")
    p_reg_sub = p_reg.add_subparsers(dest="action", required=True)

    p_reg_set = p_reg_sub.add_parser("set-canary", help="Set model canary dari MLflow run")
    p_reg_set.add_argument("run_id", type=str, help="MLflow run_id")
    p_reg_set.set_defaults(func=cmd_registry)

    p_reg_promote = p_reg_sub.add_parser("promote-canary", help="Promosikan canary -> production")
    p_reg_promote.set_defaults(func=cmd_registry)

    p_reg_status = p_reg_sub.add_parser("status", help="Lihat status registry")
    p_reg_status.set_defaults(func=cmd_registry)

    # --- serve ---
    p_serve = subparsers.add_parser("serve", help="Jalankan API FastAPI")
    p_serve.add_argument("action", choices=["start"], help="start: jalankan server")
    p_serve.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    p_serve.set_defaults(func=cmd_serve)

    # --- monitor ---
    p_monitor = subparsers.add_parser("monitor", help="Operasi monitoring")
    p_monitor.add_argument(
        "action",
        choices=["compute-drift", "show"],
        help="compute-drift: hitung drift; show: tampilkan metrik terakhir",
    )
    p_monitor.set_defaults(func=cmd_monitor)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
