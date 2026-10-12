"""Customer Lifetime Value + RFM segmentation.

Combines transactional RFM with graph-derived features (centrality, community).
CLV uses the standard predictive formula:
    CLV = avg_order_value * annual_purchase_frequency * expected_lifespan_years * margin
Segmentation is RFM-quantile based (Champions / Loyal / At-Risk / Hibernating),
the standard scheme marketing teams already use, so segments need no translation.
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from novacart.common import storage
from novacart.common.config import settings
from novacart.common.dates import as_date

_CACHE: dict[str, pd.DataFrame] = {}


def compute_clv_table(use_cache: bool = True) -> pd.DataFrame:
    """CLV + RFM per customer. Cached at module level — the served endpoints hit
    this on every request, so we compute the full table once and reuse it."""
    if use_cache and "clv" in _CACHE:
        return _CACHE["clv"]
    orders = storage.read_sql("SELECT * FROM orders")
    customers = storage.read_sql("SELECT * FROM customers")
    orders["order_dt"] = pd.to_datetime(orders["order_ts"]).dt.date
    ref = as_date(settings.reference_date)
    past = orders[orders["order_dt"] < ref]

    agg = past.groupby("customer_id").agg(
        frequency=("order_id", "count"),
        monetary=("order_total", "sum"),
        avg_order_value=("order_total", "mean"),
        first_order=("order_dt", "min"),
        last_order=("order_dt", "max"),
    ).reset_index()

    agg["recency_days"] = agg["last_order"].apply(lambda d: (ref - d).days)
    agg["tenure_days"] = agg["first_order"].apply(lambda d: max((ref - d).days, 1))
    agg["annual_freq"] = agg["frequency"] / (agg["tenure_days"] / 365.0)
    agg["predicted_clv"] = (
        agg["avg_order_value"] * agg["annual_freq"]
        * settings.clv_lifespan_years * settings.clv_margin
    ).round(2)
    agg["historical_value"] = agg["monetary"].round(2)

    agg["segment_rfm"] = _rfm_segment(agg)
    agg = agg.merge(customers[["customer_id", "segment"]], on="customer_id", how="left")
    out = agg.sort_values("predicted_clv", ascending=False).reset_index(drop=True)
    _CACHE["clv"] = out
    return out


def _rfm_segment(df: pd.DataFrame) -> pd.Series:
    # 1..4 quantile scores; recency reversed (lower recency = better).
    r = pd.qcut(df["recency_days"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    f = pd.qcut(df["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    m = pd.qcut(df["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)

    def label(row_r, row_f, row_m):
        if row_r >= 3 and row_f >= 3:
            return "Champions"
        if row_f >= 3:
            return "Loyal"
        if row_r >= 3:
            return "Promising"
        if row_r <= 2 and row_m >= 3:
            return "At-Risk (high value)"
        return "Hibernating"

    return pd.Series([label(a, b, c) for a, b, c in zip(r, f, m)], index=df.index)


def top_clv(n: int = 10) -> list[dict]:
    df = compute_clv_table().head(n)
    cols = ["customer_id", "predicted_clv", "historical_value", "frequency",
            "avg_order_value", "segment_rfm", "segment"]
    return df[cols].to_dict(orient="records")


def customer_clv(customer_id: int) -> dict | None:
    df = compute_clv_table()
    row = df[df["customer_id"] == customer_id]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "customer_id": int(customer_id),
        "predicted_clv": float(r["predicted_clv"]),
        "historical_value": float(r["historical_value"]),
        "frequency": int(r["frequency"]),
        "avg_order_value": round(float(r["avg_order_value"]), 2),
        "recency_days": int(r["recency_days"]),
        "segment_rfm": r["segment_rfm"],
    }


if __name__ == "__main__":
    df = compute_clv_table()
    print(f"CLV computed for {len(df):,} customers")
    print(df["segment_rfm"].value_counts().to_string())
    print("\nTop 5 by predicted CLV:")
    print(df.head(5)[["customer_id", "predicted_clv", "segment_rfm"]].to_string(index=False))
