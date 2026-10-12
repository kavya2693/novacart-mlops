# Model Cards — NovaCart

Three registered models share the simulated NovaCart company data. Cards below
include the limitations.

---

## 1. `novacart_churn` — 30-day customer churn

| | |
|---|---|
| **Task** | Binary classification: will an *active* customer make **no** purchase in the next 30 days? |
| **Label** | `churn = 1` if no order in the strictly-forward window `(t_ref, t_ref + 30d]`. |
| **Eligibility** | Only customers active in the trailing 90 days are labeled. Cold/censored customers are **excluded**, not defaulted to "not churned." |
| **Features** | 10 point-in-time features (RFM + behavioral + tenure), all computed strictly `< t_ref`. |
| **Model** | `StandardScaler → LogisticRegression(class_weight="balanced")`. Chosen for exact, signed log-odds contributions (explainability) — the gradient-boosted benchmark scored lower (ROC-AUC 0.772). |
| **Metrics** | ROC-AUC 0.795, PR-AUC 0.744 (25% time-agnostic holdout, base rate ≈ 0.44). |
| **Serving** | `prediction-service` fetches features from `feature-service` over HTTP, loads the Production model, returns probability + top-5 contributions + model version. |

**Limitations:**
- **Synthetic data.** Labels come from a known generative process (a renewal
  process with dormancy). The model recovers a *noisy* approximation of that
  process — that is why AUC is a realistic 0.80, not a suspicious 0.99. On real
  data I would expect lower, noisier performance and would validate against a
  true holdout period.
- **Single snapshot.** One reference date, so the split is stratified rather
  than rolling-origin. Production uses multiple as-of snapshots (backtesting
  across time) — the feature code already supports any `as_of` date.
- **No skew:** offline and online features come from **one** shared module
  (`feature_pipeline/compute.py`); `test_train_serve_feature_parity` in `tests/test_pipeline.py`
  checks the two paths agree to 1e-6 on every feature.

---

## 2. `novacart_demand` — daily demand forecast

| | |
|---|---|
| **Task** | Forecast daily company-wide order demand; 365-day horizon (per-category series are built but not trained). |
| **Models** | Seasonal-naïve (baseline), **LightGBM** (global, feature-based), ARIMA(2,1,2). |
| **Backtest** | Time-based: last 60 days held out. LightGBM MAPE 6.28% vs seasonal-naïve 7.50%; ARIMA is effectively tied (MAPE 6.29%, and better on WAPE: 6.20% vs 6.31%). |
| **Serving** | `forecast-service` serves the versioned 365-day forecast + a replenishment reorder-point calc (lead-time + safety-stock demand). |

**Limitations:** aggregate demand over 4,000 customers is smooth (central-limit),
so MAPE is optimistic vs. spiky SKU-level series; the same code runs per-SKU/
store by changing the group key (`daily_demand(level=...)`). The 365-day forecast
is a recursive roll — error compounds with horizon; production would refit daily.

---

## 3. Graph-RAG Customer Intelligence

| | |
|---|---|
| **KG** | 4,308 nodes (Customer/Product/Category/Order), 225k edges, built from orders. |
| **Analytics** | PageRank (influential products), Louvain communities (affinity segments), degree centrality, SVD embeddings (node2vec stand-in), RFM CLV segmentation. |
| **Assistant** | NL question → intent + entity parse → ontology-aware query (Cypher surfaced) → deterministic graph retrieval → cited answer. LLM synthesis is a pluggable hook. |

**Limitations:** intent routing is rule-based (deterministic + testable), not an
LLM — swap in an LLM router without touching retrieval/synthesis contracts.
Runs on networkx only. The Cypher strings are generated for transparency but
nothing here loads the graph into Neo4j or Neptune.
