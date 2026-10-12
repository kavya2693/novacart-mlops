"""Feature drift monitoring via PSI (Project 4).

Population Stability Index per feature vs a FIXED training baseline. Bin edges
are frozen from the baseline (quantile bins) so drift is measured on consistent
buckets — the trap the architecture review flagged. PSI reading:
  < 0.1  stable | 0.1-0.2 moderate shift | > 0.2 significant drift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from novacart.common.schemas import FEATURE_COLUMNS
from novacart.feature_pipeline.build_features import load_feature_table

MODERATE, SIGNIFICANT = 0.1, 0.2


def _psi(expected: np.ndarray, actual: np.ndarray, edges: np.ndarray) -> float:
    e_hist = np.histogram(expected, bins=edges)[0] / max(len(expected), 1)
    a_hist = np.histogram(actual, bins=edges)[0] / max(len(actual), 1)
    e_hist = np.clip(e_hist, 1e-6, None)
    a_hist = np.clip(a_hist, 1e-6, None)
    return float(np.sum((a_hist - e_hist) * np.log(a_hist / e_hist)))


def psi_report(baseline: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in FEATURE_COLUMNS:
        edges = np.unique(np.quantile(baseline[col], np.linspace(0, 1, 11)))
        if len(edges) < 3:
            edges = np.linspace(baseline[col].min(), baseline[col].max() + 1e-6, 4)
        edges[0], edges[-1] = -np.inf, np.inf
        val = _psi(baseline[col].to_numpy(), current[col].to_numpy(), edges)
        flag = "SIGNIFICANT" if val > SIGNIFICANT else "moderate" if val > MODERATE else "stable"
        rows.append({"feature": col, "psi": round(val, 4), "status": flag})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def run(current: pd.DataFrame | None = None) -> pd.DataFrame:
    baseline = load_feature_table()
    if current is None:
        # No live batch: simulate a shifted population (recency +40%, sessions -30%)
        # to demonstrate the detector firing. Swap for real serving logs in prod.
        current = baseline.copy()
        current["recency_days"] = current["recency_days"] * 1.4
        current["sessions_30d"] = current["sessions_30d"] * 0.7
    report = psi_report(baseline, current)
    print("PSI drift report (baseline = training feature_store):")
    print(report.to_string(index=False))
    drifted = report[report["status"] == "SIGNIFICANT"]["feature"].tolist()
    print(f"\n{len(drifted)} feature(s) with significant drift: {drifted or 'none'}")
    return report


if __name__ == "__main__":
    run()
