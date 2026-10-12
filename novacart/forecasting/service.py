"""Demand forecast serving.

Serves the versioned 365-day forecast + on-demand replenishment suggestions.

Run: uvicorn novacart.forecasting.service:app --port 8104
"""
from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from novacart.common.config import settings
from novacart.common.metrics import add_metrics_endpoint
from novacart.common.registry import Registry

app = FastAPI(title="NovaCart Demand Forecast Service", version="1.0.0")
FORECAST_PATH = settings.features_dir / "demand_forecast_365d.parquet"


def _forecast_df() -> pd.DataFrame:
    if not FORECAST_PATH.exists():
        raise HTTPException(status_code=503, detail="no forecast — run forecasting.train first")
    return pd.read_parquet(FORECAST_PATH)


@app.get("/health")
def health() -> dict:
    reg = Registry(model_name="novacart_demand")
    mv = reg.get_production()
    return {"status": "ok" if (mv and FORECAST_PATH.exists()) else "degraded",
            "service": "forecast-service",
            "model_version": mv.version if mv else None,
            "best_model": (mv.metrics.get("best_model") if mv else None)}


@app.get("/forecast")
def forecast(horizon: int = Query(30, ge=1, le=365)) -> dict:
    df = _forecast_df().head(horizon)
    return {"horizon": horizon, "total_forecast": round(float(df["forecast"].sum()), 1),
            "daily": df.assign(date=df["date"].astype(str)).to_dict(orient="records")}


@app.get("/replenishment")
def replenishment(safety_stock_days: int = 7, lead_time_days: int = 3) -> dict:
    """Convert the forecast into a reorder quantity (demand over lead+safety window)."""
    df = _forecast_df().head(lead_time_days + safety_stock_days)
    reorder_qty = float(df["forecast"].sum())
    return {"lead_time_days": lead_time_days, "safety_stock_days": safety_stock_days,
            "reorder_point_units": round(reorder_qty, 1),
            "avg_daily_demand": round(float(df["forecast"].mean()), 1)}


@app.get("/model/info")
def model_info() -> dict:
    mv = Registry(model_name="novacart_demand").get_production()
    if mv is None:
        raise HTTPException(status_code=503, detail="no Production forecast model")
    return {"model_name": "novacart_demand", "version": mv.version,
            "stage": mv.stage, "metrics": mv.metrics, "created_at": mv.created_at}


add_metrics_endpoint(app)
