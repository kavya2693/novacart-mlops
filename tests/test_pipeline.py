"""End-to-end + unit tests for the NovaCart platform.

Assumes the pipeline has been run (data engine -> features -> train -> graph
-> forecast). The Makefile target `make test` runs the pipeline first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from novacart.common import storage
from novacart.common.config import settings
from novacart.common.registry import Registry
from novacart.common.schemas import FEATURE_COLUMNS, CustomerFeatures


# --- data engine -----------------------------------------------------------
def test_raw_tables_exist():
    for t in ["customers", "products", "orders", "order_items", "events"]:
        assert storage.table_exists(t), f"missing table {t}"


def test_generator_deterministic():
    from novacart.data_engine.generate import generate_customers
    import numpy as np
    from datetime import date
    a = generate_customers(np.random.default_rng(1), 50, date(2026, 6, 1))
    b = generate_customers(np.random.default_rng(1), 50, date(2026, 6, 1))
    pd.testing.assert_frame_equal(a, b)


# --- feature pipeline ------------------------------------------------------
def test_feature_store_built():
    df = storage.read_sql("SELECT * FROM feature_store")
    assert len(df) > 0
    assert set(FEATURE_COLUMNS).issubset(df.columns)
    assert df["churn"].nunique() == 2, "label must have both classes"


def test_point_in_time_no_future_leak():
    """No feature row may reference activity on/after the reference date."""
    df = storage.read_sql("SELECT * FROM feature_store")
    # recency is days-since-last-purchase as-of ref; must be > 0 (strictly past).
    assert (df["recency_days"] > 0).all()


def test_train_serve_feature_parity():
    """The SAME customer through the offline batch and the single-customer online
    path must produce identical features (kills train/serve skew)."""
    from novacart.feature_pipeline.compute import (
        compute_features_batch, compute_customer_features,
    )
    customers = storage.read_sql("SELECT * FROM customers LIMIT 5")
    orders = storage.read_sql("SELECT * FROM orders")
    events = storage.read_sql("SELECT * FROM events")
    batch = compute_features_batch(customers, orders, events, settings.reference_date)
    for _, crow in customers.iterrows():
        cid = int(crow["customer_id"])
        online = compute_customer_features(
            crow.to_dict(),
            orders[orders["customer_id"] == cid],
            events[events["customer_id"] == cid],
            settings.reference_date,
        )
        brow = batch[batch["customer_id"] == cid].iloc[0]
        for col in FEATURE_COLUMNS:
            assert getattr(online, col) == pytest.approx(float(brow[col]), rel=1e-6), col


# --- registry / training ---------------------------------------------------
def test_churn_model_registered_production():
    reg = Registry(model_name="novacart_churn")
    mv = reg.get_production()
    assert mv is not None and mv.stage == "Production"
    assert mv.metrics["roc_auc"] > 0.7, "churn model should clearly beat random"


def test_demand_model_registered():
    reg = Registry(model_name="novacart_demand")
    mv = reg.get_production()
    assert mv is not None
    assert mv.metrics["lightgbm_mape"] < 25.0, "must beat a 25% MAPE baseline"


def test_only_one_production_version():
    for name in ["novacart_churn", "novacart_demand"]:
        prods = [v for v in Registry(model_name=name).list_versions() if v.stage == "Production"]
        assert len(prods) <= 1


# --- graph -----------------------------------------------------------------
def test_graph_and_rag():
    from novacart.graph.build_graph import load_graph
    from novacart.graph.rag.engine import GraphRAG
    G = load_graph()
    assert G.number_of_nodes() > 4000
    rag = GraphRAG(G)
    res = rag.ask("Who are the most influential products?")
    assert res["intent"] == "influential_products"
    assert "MATCH" in res["generated_query"]
    assert len(res["citations"]) > 0


def test_clv_segments():
    from novacart.graph.clv import compute_clv_table
    df = compute_clv_table()
    assert (df["predicted_clv"] >= 0).all()
    assert df["segment_rfm"].nunique() >= 3


# --- forecasting -----------------------------------------------------------
def test_mape_metric():
    from novacart.forecasting.dataset import mape
    assert mape(np.array([100, 100]), np.array([100, 100])) == 0.0
    assert mape(np.array([100.0]), np.array([90.0])) == pytest.approx(10.0)
