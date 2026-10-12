"""Feature-service (Project 3a).

Serves online, point-in-time features for a single customer by calling the SAME
compute module the offline pipeline uses. This is the "feature store" side of
the two-service split, kept deliberately so the service contract is explicit.

Run: uvicorn novacart.serving.feature_service:app --port 8001
"""
from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException

from novacart.common.config import settings
from novacart.common import storage
from novacart.common.metrics import add_metrics_endpoint, metrics
from novacart.common.schemas import CustomerFeatures, HealthResponse
from novacart.feature_pipeline.compute import compute_customer_features

app = FastAPI(title="NovaCart Feature Service", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ok = storage.table_exists("customers")
    return HealthResponse(
        status="ok" if ok else "degraded",
        service="feature-service",
        detail=None if ok else "raw tables missing — run the data engine",
    )


@app.get("/features/{customer_id}", response_model=CustomerFeatures)
def get_features(customer_id: int) -> CustomerFeatures:
    t0 = time.perf_counter()
    cust = storage.read_sql(
        "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
    )
    if cust.empty:
        metrics.inc("feature_requests_total", {"status": "404"})
        raise HTTPException(status_code=404, detail=f"customer {customer_id} not found")

    orders = storage.read_sql(
        "SELECT * FROM orders WHERE customer_id = ?", (customer_id,)
    )
    events = storage.read_sql(
        "SELECT * FROM events WHERE customer_id = ?", (customer_id,)
    )
    feats = compute_customer_features(
        cust.iloc[0].to_dict(), orders, events, settings.reference_date
    )
    metrics.inc("feature_requests_total", {"status": "200"})
    metrics.observe("feature_latency_ms", (time.perf_counter() - t0) * 1000)
    return feats


add_metrics_endpoint(app)
