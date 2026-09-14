"""Scores a model version against a fixed, seeded holdout split of the
evaluation dataset so metrics are directly comparable across versions
(the champion/challenger comparison in the promotion gate).

The full eval CSV is the training distribution (see scripts/fetch_data.py);
we carve out a deterministic holdout slice (last 20%, fixed seed) rather than
reusing the whole file, since these are the original labeled rows — using a
held-out slice consistently is the honest way to compare within this
take-home's constraints (no independent physician-labeled set is available).
This limitation is called out explicitly in REPORT.md / DESIGN.md.
"""
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.ml.errors import ModelLoadError
from app.ml.predictor import predict_batch
from app.ml.schema import FEATURE_ORDER, TARGET_COLUMN

logger = logging.getLogger("manas.evaluate")

HOLDOUT_FRACTION = 0.2
RANDOM_SEED = 42


class EvalDataUnavailable(Exception):
    pass


def load_holdout(data_path: str | Path) -> pd.DataFrame:
    p = Path(data_path)
    if not p.exists():
        raise EvalDataUnavailable(f"evaluation dataset not found at {data_path}")
    try:
        df = pd.read_csv(p)
    except Exception as exc:  # noqa: BLE001
        raise EvalDataUnavailable(f"failed to read evaluation dataset: {exc}") from exc

    missing = set(FEATURE_ORDER + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise EvalDataUnavailable(f"evaluation dataset missing columns: {sorted(missing)}")

    # Deterministic shuffle + fixed holdout slice, stable across processes/runs.
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    n_holdout = int(len(df) * HOLDOUT_FRACTION)
    return df.iloc[-n_holdout:].reset_index(drop=True)


def evaluate_model(
    *,
    model: Any,
    scaler: Any,
    framework: str,
    data_path: str | Path,
    threshold: float = 0.5,
    batch_size: int = 2000,
) -> dict:
    """Returns a metrics dict: accuracy, precision, recall, f1, roc_auc,
    n_samples, positive_rate. Raises EvalDataUnavailable / ModelLoadError /
    InferenceError on failure — never a bare exception."""
    holdout = load_holdout(data_path)
    y_true = holdout[TARGET_COLUMN].astype(int).to_numpy()
    records = holdout[FEATURE_ORDER].to_dict(orient="records")

    y_pred = np.empty(len(records), dtype=int)
    y_prob = np.empty(len(records), dtype=float)

    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        results = predict_batch(chunk, model=model, scaler=scaler, framework=framework, threshold=threshold)
        for i, r in enumerate(results):
            y_pred[start + i] = r.prediction
            y_prob[start + i] = r.probability

    metrics = {
        "n_samples": int(len(y_true)),
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_pred": float(np.mean(y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = None  # only one class present in holdout, degenerate case

    return metrics
