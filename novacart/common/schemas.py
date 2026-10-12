"""Pydantic data contracts shared between the pipeline and the API layer.

These are the single source of truth for what a "feature row" and a
"prediction" look like. Both the feature-service and prediction-service import
them, so the service-to-service contract cannot silently drift.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

# The exact, ordered list of model input features. Training and serving both
# import FEATURE_COLUMNS so the vector fed to the model is identical in both
# paths (this is how we prevent train/serve skew).
FEATURE_COLUMNS: list[str] = [
    "recency_days",
    "frequency_90d",
    "monetary_90d",
    "avg_order_value",
    "tenure_days",
    "sessions_30d",
    "support_tickets_90d",
    "distinct_categories_90d",
    "discount_rate",
    "days_between_orders",
]


class CustomerFeatures(BaseModel):
    """Point-in-time feature vector for one customer, as-of the reference date."""

    customer_id: int
    as_of_date: str

    recency_days: float = Field(..., description="Days since last purchase")
    frequency_90d: float = Field(..., description="Orders in trailing 90 days")
    monetary_90d: float = Field(..., description="Spend in trailing 90 days")
    avg_order_value: float
    tenure_days: float = Field(..., description="Days since signup")
    sessions_30d: float = Field(..., description="Web sessions in trailing 30 days")
    support_tickets_90d: float
    distinct_categories_90d: float
    discount_rate: float = Field(..., description="Share of spend that was discounted")
    days_between_orders: float = Field(..., description="Avg gap between orders")

    def to_vector(self) -> list[float]:
        """Return features in FEATURE_COLUMNS order for the model."""
        return [float(getattr(self, c)) for c in FEATURE_COLUMNS]


class FeatureContribution(BaseModel):
    feature: str
    value: float
    contribution: float = Field(..., description="Signed impact on churn log-odds")


class ChurnPrediction(BaseModel):
    customer_id: int
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    risk_band: str = Field(..., description="low | medium | high")
    model_name: str
    model_version: int
    as_of_date: str
    top_contributions: list[FeatureContribution]


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str
    detail: Optional[str] = None
