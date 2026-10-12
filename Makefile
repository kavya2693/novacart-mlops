PY ?= ./.venv/bin/python
.DEFAULT_GOAL := help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv: ## Create venv + install deps
	python3 -m venv .venv && $(PY) -m pip install -U pip && $(PY) -m pip install -r requirements.txt

data: ## Project 0 — simulate NovaCart company
	$(PY) -m novacart.data_engine.generate

features: ## Project 1 — point-in-time features + label
	$(PY) -m novacart.feature_pipeline.build_features

train: ## Project 2 — train churn model + register
	$(PY) -m novacart.training.train

graph: ## Build KG + CLV + graph analytics
	$(PY) -m novacart.graph.build_graph && $(PY) -m novacart.graph.clv

forecast: ## Demand forecasting backtest + 365d forecast
	$(PY) -m novacart.forecasting.train

drift: ## Project 4 — PSI drift report
	$(PY) -m novacart.monitoring.drift

pipeline: ## Run the whole pipeline one step at a time
	bash scripts/run_pipeline.sh

serve: ## Start all four APIs (feature 8101, prediction 8102, graph-rag 8103, forecast 8104)
	$(PY) -m uvicorn novacart.serving.feature_service:app --port 8101 & \
	$(PY) -m uvicorn novacart.serving.prediction_service:app --port 8102 & \
	$(PY) -m uvicorn novacart.graph.service:app --port 8103 & \
	$(PY) -m uvicorn novacart.forecasting.service:app --port 8104 & \
	echo "APIs starting on 8101-8104 — Ctrl-C to stop"; wait

test: ## Run the pipeline then the test suite
	bash scripts/run_pipeline.sh && $(PY) -m pytest -q tests/

reproduce: venv pipeline ## Full clean reproduce from scratch

.PHONY: help venv data features train graph forecast drift pipeline serve test reproduce
