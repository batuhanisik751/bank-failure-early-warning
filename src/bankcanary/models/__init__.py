"""Baseline models on the fixed out-of-time split (spec section 8, contract section 8)."""

from bankcanary.models.baselines import (
    MODEL_NAMES,
    SMALL_FEATURES,
    TexasRatioScorer,
    coefficient_table,
    make_model,
    model_features,
    score_pipeline,
)
from bankcanary.models.preprocess import Winsorizer, make_pipeline
from bankcanary.models.train import (
    TrainResult,
    evaluate_model,
    load_model,
    model_dir,
    train_model,
    write_baselines_report,
)

__all__ = [
    "MODEL_NAMES",
    "SMALL_FEATURES",
    "TexasRatioScorer",
    "TrainResult",
    "Winsorizer",
    "coefficient_table",
    "evaluate_model",
    "load_model",
    "make_model",
    "make_pipeline",
    "model_dir",
    "model_features",
    "score_pipeline",
    "train_model",
    "write_baselines_report",
]
