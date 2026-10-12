"""Prediction-service (Project 3b).

The model-serving side. On a request it:
  1. fetches point-in-time features from the feature-service OVER HTTP,
  2. loads the Production model from the registry,
  3. scores churn probability and computes signed per-feature contributions,
  4. returns a typed ChurnPrediction.

Step 1 is the "API transfer" / service-to-service contract that keeps model
serving decoupled from feature computation.

Run: uvicorn novacart.serving.prediction_service:app --port 8002
"""
from __future__ import annotations

import time

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException

from novacart.common.config import settings
from novacart.common.metrics import add_metrics_endpoint, metrics
from novacart.common.registry import Registry
from novacart.common.schemas import (
    ChurnPrediction,
    CustomerFeatures,
    FEATURE_COLUMNS,
    FeatureContribution,
    HealthResponse,
)

app = FastAPI(title="NovaCart Prediction Service", version="1.0.0")

# Loaded once at import; refreshed via /model/reload after a new promotion.
_STATE: dict = {"model": None, "meta": None}
# Pooled, keep-alive client reused across requests (no per-call connection setup).
_HTTP = httpx.Client(timeout=settings.http_timeout_seconds)


def _load_model() -> None:
    model, meta = Registry().load_production_model()
    _STATE["model"], _STATE["meta"] = model, meta


try:
    _load_model()
except Exception as exc:  # serving can boot before training; /health reflects it
    _STATE["error"] = str(exc)


def _risk_band(p: float) -> str:
    return ("high" if p >= settings.risk_band_high
            else "medium" if p >= settings.risk_band_medium else "low")


def _contributions(feats: CustomerFeatures) -> list[FeatureContribution]:
    """Exact log-odds contributions: coef_i * (x_i - mean_i) / scale_i."""
    pipe = _STATE["model"]
    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]
    x = np.array(feats.to_vector(), dtype=float)
    z = (x - scaler.mean_) / scaler.scale_
    contribs = clf.coef_[0] * z
    order = np.argsort(-np.abs(contribs))[:5]
    return [
        FeatureContribution(
            feature=FEATURE_COLUMNS[i],
            value=round(float(x[i]), 4),
            contribution=round(float(contribs[i]), 4),
        )
        for i in order
    ]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ready = _STATE.get("model") is not None
    return HealthResponse(
        status="ok" if ready else "degraded",
        service="prediction-service",
        detail=None if ready else _STATE.get("error", "no Production model"),
    )


@app.get("/model/info")
def model_info() -> dict:
    meta = _STATE.get("meta")
    if meta is None:
        raise HTTPException(status_code=503, detail="no Production model loaded")
    return {
        "model_name": settings.model_name,
        "version": meta.version,
        "stage": meta.stage,
        "metrics": meta.metrics,
        "feature_columns": meta.feature_columns,
        "sklearn_version": meta.sklearn_version,
        "created_at": meta.created_at,
    }


@app.post("/model/reload")
def model_reload() -> dict:
    _load_model()
    return {"reloaded": True, "version": _STATE["meta"].version}


@app.get("/predict/{customer_id}", response_model=ChurnPrediction)
def predict(customer_id: int) -> ChurnPrediction:
    if _STATE.get("model") is None:
        raise HTTPException(status_code=503, detail="model not loaded — train first")
    t0 = time.perf_counter()

    # 1) API transfer: fetch features from the feature-service.
    url = f"{settings.feature_service_url}/features/{customer_id}"
    try:
        resp = _HTTP.get(url)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"feature-service unreachable: {exc}")
    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail=f"customer {customer_id} not found")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"feature-service error {resp.status_code}")

    feats = CustomerFeatures(**resp.json())

    # 2+3) score.
    pipe = _STATE["model"]
    prob = float(pipe.predict_proba([feats.to_vector()])[0, 1])
    meta = _STATE["meta"]
    band = _risk_band(prob)

    metrics.inc("predictions_total", {"band": band})
    metrics.observe("prediction_latency_ms", (time.perf_counter() - t0) * 1000)
    metrics.observe("churn_probability", prob)

    return ChurnPrediction(
        customer_id=customer_id,
        churn_probability=round(prob, 4),
        risk_band=band,
        model_name=settings.model_name,
        model_version=meta.version,
        as_of_date=feats.as_of_date,
        top_contributions=_contributions(feats),
    )


add_metrics_endpoint(app)
