#!/usr/bin/env bash
# Runs the NovaCart platform ONE STEP AT A TIME (Projects 0 -> 4).
# Each step is independent and idempotent; re-running is safe.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-./.venv/bin/python}"

step() { echo; echo "════════ $1 ════════"; }

step "0/4  Data engine — simulate the NovaCart company"
$PY -m novacart.data_engine.generate

step "1/4  Feature pipeline — point-in-time features + churn label"
$PY -m novacart.feature_pipeline.build_features

step "2/4  Train churn model + register -> Production"
$PY -m novacart.training.train

step "2b   Build Knowledge Graph + CLV + graph analytics"
$PY -m novacart.graph.build_graph
$PY -m novacart.graph.clv

step "3/4  Demand forecasting — LightGBM/ARIMA/seasonal-naive backtest + 365d forecast"
$PY -m novacart.forecasting.train

step "4/4  Drift report"
$PY -m novacart.monitoring.drift || echo "(drift module optional)"

echo; echo "✅ Pipeline complete. Start the APIs with:  make serve"
