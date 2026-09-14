"""Population Stability Index (PSI) drift detection: compares the feature
distribution of recent live inference inputs against a reference baseline
computed once from the training/eval dataset.

PSI is a standard, simple, interpretable drift metric:
  PSI < 0.1  -> no significant shift
  0.1 - 0.25 -> moderate shift, worth investigating
  > 0.25     -> significant shift, model may need re-evaluation

This is monitoring only — it never triggers retraining automatically (see
DESIGN.md). It surfaces on the dashboard as a signal for a human to act on.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.schema import FEATURE_ORDER

N_BINS = 10
EPSILON = 1e-4


def compute_reference_bins(data_path: str | Path) -> dict[str, np.ndarray]:
    """Quantile bin edges per feature, computed once from the reference
    (training/eval) distribution. Cached by the caller (drift router)."""
    df = pd.read_csv(data_path)
    bins = {}
    for name in FEATURE_ORDER:
        col = df[name].astype(float)
        try:
            edges = np.unique(np.quantile(col, np.linspace(0, 1, N_BINS + 1)))
        except Exception:  # noqa: BLE001
            edges = np.array([col.min(), col.max()])
        if len(edges) < 2:
            edges = np.array([col.min() - 0.5, col.max() + 0.5])
        bins[name] = edges
    return bins


def _psi_for_feature(reference: np.ndarray, current: np.ndarray, edges: np.ndarray) -> float:
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1) + EPSILON
    cur_pct = cur_counts / max(cur_counts.sum(), 1) + EPSILON

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    bins: dict[str, np.ndarray],
) -> dict[str, float]:
    """Returns {feature_name: psi_score} for every feature present in both
    frames. Missing/empty current_df yields an empty dict (not an error) —
    drift simply can't be assessed yet with no recent traffic."""
    if current_df.empty:
        return {}
    result = {}
    for name in FEATURE_ORDER:
        if name not in reference_df.columns or name not in current_df.columns:
            continue
        edges = bins.get(name)
        if edges is None or len(edges) < 2:
            continue
        result[name] = _psi_for_feature(
            reference_df[name].astype(float).to_numpy(),
            current_df[name].astype(float).to_numpy(),
            edges,
        )
    return result


def classify_psi(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "moderate_shift"
    return "significant_shift"
