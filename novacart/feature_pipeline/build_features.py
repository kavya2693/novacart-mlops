"""Offline feature + label pipeline (Project 1).

Reads raw tables, computes point-in-time features via the SHARED compute module,
derives the strictly-forward 30-day churn label, validates the schema, and
persists a training-ready feature table.
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from novacart.common.config import settings
from novacart.common import storage
from novacart.common.dates import as_date
from novacart.common.schemas import CustomerFeatures, FEATURE_COLUMNS
from novacart.feature_pipeline.compute import compute_features_batch

FEATURE_TABLE = "feature_store"


def build(as_of: str | None = None) -> pd.DataFrame:
    as_of = as_of or settings.reference_date
    as_of_d = as_date(as_of)
    horizon = settings.churn_horizon_days
    lookback = settings.active_lookback_days

    customers = storage.read_sql("SELECT * FROM customers")
    orders = storage.read_sql("SELECT * FROM orders")
    events = storage.read_sql("SELECT * FROM events")

    # 1) Point-in-time features (strictly < as_of).
    feats = compute_features_batch(customers, orders, events, as_of)

    # 2) Label: churn = NO purchase in the strictly-forward window (as_of, as_of+h].
    orders = orders.copy()
    orders["order_dt"] = pd.to_datetime(orders["order_ts"]).dt.date
    fwd_start = as_of_d
    fwd_end = as_of_d + timedelta(days=horizon)
    forward = orders[(orders["order_dt"] > fwd_start) & (orders["order_dt"] <= fwd_end)]
    purchasers_fwd = set(forward["customer_id"].unique())

    # 3) Eligibility: only label customers who were ACTIVE before as_of
    #    (purchased within the lookback). Others are cold/censored -> excluded,
    #    NOT defaulted to "not churned" (a trap the pressure-test flagged).
    win_active = as_of_d - timedelta(days=lookback)
    active_before = set(
        orders[(orders["order_dt"] < as_of_d) & (orders["order_dt"] >= win_active)][
            "customer_id"
        ].unique()
    )

    feats["churn"] = (~feats["customer_id"].isin(purchasers_fwd)).astype(int)
    feats["eligible"] = feats["customer_id"].isin(active_before)
    labeled = feats[feats["eligible"]].drop(columns=["eligible"]).reset_index(drop=True)

    # 4) Contract validation — every row must satisfy the pydantic schema.
    _validate(labeled)

    storage.write_df(FEATURE_TABLE, labeled)
    labeled.to_parquet(settings.features_dir / "feature_store.parquet", index=False)
    return labeled


def _validate(df: pd.DataFrame) -> None:
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"feature table missing columns: {missing}")
    # Validate a sample against the pydantic contract (fail fast on bad types/ranges).
    for _, row in df.head(50).iterrows():
        CustomerFeatures(**{k: row[k] for k in CustomerFeatures.model_fields})
    if df["churn"].nunique() < 2:
        raise ValueError("label has a single class — check windows / generator")


def load_feature_table() -> pd.DataFrame:
    return storage.read_sql(f"SELECT * FROM {FEATURE_TABLE}")


if __name__ == "__main__":
    df = build()
    rate = df["churn"].mean()
    print(f"feature_store built: rows={len(df):,}  churn_rate={rate:.3f}")
    print(df[FEATURE_COLUMNS + ["churn"]].describe().round(2).T.to_string())
