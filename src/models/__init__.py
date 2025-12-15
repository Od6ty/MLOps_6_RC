# src/models/__init__.py
from .train import train_new_model, train_pipeline_with_registry
from .registry import (
    set_canary_from_run,
    promote_canary_to_production,
    print_status,
)

__all__ = [
    "train_new_model",
    "train_pipeline_with_registry",
    "set_canary_from_run",
    "promote_canary_to_production",
    "print_status",
]

