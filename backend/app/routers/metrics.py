"""Basic monitoring: latency percentiles, error rate, prediction distribution,
and feature drift (PSI) vs the training baseline. Read-only, feeds the
dashboard. Monitoring surfaces signals for a human to act on — it never
triggers retraining or promotion itself (see DESIGN.md)."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import get_db
from app.ml.drift import classify_psi, compute_drift, compute_reference_bins
from app.ml.schema import FEATURE_ORDER
from app.models_db import InferenceLog, ModelStatus, ModelVersion, RequestLog, User
from app.security import get_current_user

router = APIRouter(prefix="/metrics", tags=["metrics"])

_drift_bins_cache: dict | None = None


@router.get("/summary")
def metrics_summary(
    window_minutes: int = Query(default=60, ge=1, le=10080),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    since = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)

    req_logs = db.query(RequestLog).filter(RequestLog.created_at >= since).all()
    total_requests = len(req_logs)
    error_requests = sum(1 for r in req_logs if r.status_code >= 400)
    latencies = sorted(r.latency_ms for r in req_logs)

    def _pct(p: float) -> float | None:
        if not latencies:
            return None
        idx = min(int(len(latencies) * p), len(latencies) - 1)
        return latencies[idx]

    inf_logs = db.query(InferenceLog).filter(InferenceLog.created_at >= since).all()
    ok_logs = [l for l in inf_logs if l.status == "ok" and l.prediction is not None]
    positive_rate = (sum(1 for l in ok_logs if l.prediction == 1) / len(ok_logs)) if ok_logs else None

    active = db.query(ModelVersion).filter(ModelVersion.status == ModelStatus.production).first()

    return {
        "window_minutes": window_minutes,
        "total_requests": total_requests,
        "error_requests": error_requests,
        "error_rate": (error_requests / total_requests) if total_requests else None,
        "latency_ms": {"p50": _pct(0.5), "p95": _pct(0.95), "p99": _pct(0.99)},
        "inference_count": len(inf_logs),
        "inference_error_count": sum(1 for l in inf_logs if l.status == "error"),
        "prediction_positive_rate": positive_rate,
        "active_model": {
            "id": active.id, "name": active.name, "version": active.version,
        } if active else None,
    }


@router.get("/drift")
def drift_report(
    window_minutes: int = Query(default=1440, ge=1, le=43200),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> dict:
    global _drift_bins_cache

    data_path = Path(settings.data_dir) / "brfss2015_5050split.csv"
    if not data_path.exists():
        raise HTTPException(status_code=503, detail="reference dataset unavailable; cannot compute drift")

    if _drift_bins_cache is None:
        _drift_bins_cache = {
            "bins": compute_reference_bins(data_path),
            "reference_df": pd.read_csv(data_path)[FEATURE_ORDER],
        }

    since = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
    recent = (
        db.query(InferenceLog)
        .filter(InferenceLog.created_at >= since, InferenceLog.status == "ok")
        .all()
    )
    if not recent:
        return {"window_minutes": window_minutes, "n_samples": 0, "features": {}, "message": "no recent inferences to assess"}

    current_df = pd.DataFrame([l.input_echo for l in recent])[FEATURE_ORDER]
    psi_by_feature = compute_drift(_drift_bins_cache["reference_df"], current_df, _drift_bins_cache["bins"])

    return {
        "window_minutes": window_minutes,
        "n_samples": len(recent),
        "features": {
            name: {"psi": round(psi, 4), "status": classify_psi(psi)}
            for name, psi in sorted(psi_by_feature.items(), key=lambda kv: -kv[1])
        },
        "overall_status": classify_psi(max(psi_by_feature.values())) if psi_by_feature else "stable",
    }
