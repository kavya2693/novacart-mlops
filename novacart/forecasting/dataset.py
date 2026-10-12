"""Demand time-series construction.

Turns the NovaCart order stream into daily demand series at two hierarchy
levels — total company and per-category — and builds the supervised feature
matrix (calendar + lag + rolling) that the global LightGBM model consumes.
SKU-level and store-level use the identical code with a different group key.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from novacart.common import storage

LAGS = [1, 7, 14, 28]
ROLLS = [7, 28]


def daily_demand(level: str = "total") -> pd.DataFrame:
    """Daily order-count + revenue demand. level in {'total','category'}."""
    orders = storage.read_sql("SELECT * FROM orders")
    orders["date"] = pd.to_datetime(orders["order_ts"]).dt.normalize()

    if level == "total":
        g = orders.groupby("date").agg(
            demand=("order_id", "count"), revenue=("order_total", "sum")
        ).reset_index()
        g["series"] = "total"
        return _fill_calendar(g, ["series"])

    if level == "category":
        items = storage.read_sql("SELECT * FROM order_items")
        oid_date = dict(zip(orders["order_id"], orders["date"]))
        items["date"] = items["order_id"].map(oid_date)
        g = items.groupby(["date", "category"]).agg(
            demand=("product_id", "count"), revenue=("unit_price", "sum")
        ).reset_index().rename(columns={"category": "series"})
        return _fill_calendar(g, ["series"])

    raise ValueError(f"unknown level {level}")


def _fill_calendar(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Reindex each series to a gap-free daily calendar (0 demand on empty days)."""
    out = []
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    for key_vals, grp in df.groupby(keys):
        grp = grp.set_index("date").reindex(full).rename_axis("date").reset_index()
        for k, v in zip(keys, key_vals if isinstance(key_vals, tuple) else (key_vals,)):
            grp[k] = v
        grp["demand"] = grp["demand"].fillna(0.0)
        grp["revenue"] = grp["revenue"].fillna(0.0)
        out.append(grp)
    return pd.concat(out, ignore_index=True)


def make_supervised(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar + lag + rolling features per series. Returns model-ready frame."""
    frames = []
    for _, grp in df.groupby("series"):
        grp = grp.sort_values("date").copy()
        d = grp["date"].dt
        grp["dow"] = d.dayofweek
        grp["month"] = d.month
        grp["day"] = d.day
        grp["week"] = d.isocalendar().week.astype(int)
        grp["is_weekend"] = (grp["dow"] >= 5).astype(int)
        for lag in LAGS:
            grp[f"lag_{lag}"] = grp["demand"].shift(lag)
        for r in ROLLS:
            grp[f"roll_mean_{r}"] = grp["demand"].shift(1).rolling(r).mean()
            grp[f"roll_std_{r}"] = grp["demand"].shift(1).rolling(r).std()
        frames.append(grp)
    out = pd.concat(frames, ignore_index=True)
    return out.dropna().reset_index(drop=True)


FEATURE_COLS = (
    ["dow", "month", "day", "week", "is_weekend"]
    + [f"lag_{l}" for l in LAGS]
    + [f"roll_mean_{r}" for r in ROLLS]
    + [f"roll_std_{r}" for r in ROLLS]
)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom * 100) if denom else float("nan")
