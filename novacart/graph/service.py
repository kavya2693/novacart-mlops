"""Graph-RAG service.

Exposes the question-answering assistant + graph analytics as HTTP endpoints so
any client (a dashboard, a CRM panel, a notebook) can query the same graph.

Run: uvicorn novacart.graph.service:app --port 8103
"""
from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from novacart.common.metrics import add_metrics_endpoint, metrics
from novacart.graph import analytics, clv
from novacart.graph.build_graph import load_graph
from novacart.graph.ontology import NodeType, describe, node_id
from novacart.graph.rag.engine import GraphRAG

app = FastAPI(title="NovaCart Graph-RAG Customer Intelligence", version="1.0.0")

_G = None
_RAG: GraphRAG | None = None


def _rag() -> GraphRAG:
    global _G, _RAG
    if _RAG is None:
        _G = load_graph()
        _RAG = GraphRAG(_G)
    return _RAG


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health() -> dict:
    try:
        g = _rag().G
        return {"status": "ok", "service": "graph-rag",
                "nodes": g.number_of_nodes(), "edges": g.number_of_edges()}
    except Exception as exc:  # pragma: no cover
        return {"status": "degraded", "service": "graph-rag", "detail": str(exc)}


@app.get("/ontology", response_class=PlainTextResponse)
def ontology() -> str:
    return describe()


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    t0 = time.perf_counter()
    result = _rag().ask(req.question)
    metrics.inc("graphrag_questions_total", {"intent": result["intent"]})
    metrics.observe("graphrag_latency_ms", (time.perf_counter() - t0) * 1000)
    return result


@app.get("/analytics/influential-products")
def influential(top_n: int = 10) -> list[dict]:
    return analytics.influential_products(_rag().G, top_n=top_n)


@app.get("/analytics/communities")
def communities() -> dict:
    return analytics.product_communities(_rag().G)


@app.get("/analytics/top-clv")
def top_clv(top_n: int = 10) -> list[dict]:
    return clv.top_clv(top_n)


@app.get("/customer/{customer_id}")
def customer(customer_id: int) -> dict:
    info = clv.customer_clv(customer_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"customer {customer_id} not found")
    recs, _ = _rag()._recommend(customer_id)
    info["next_best_products"] = recs
    return info


@app.on_event("startup")
def _warm() -> None:
    """Pay graph-load cost at boot, not on the first user request."""
    _rag()


add_metrics_endpoint(app)
