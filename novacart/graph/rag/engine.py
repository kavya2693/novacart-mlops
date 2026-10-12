"""Graph-RAG query engine: NL business questions -> cited answers.

Pipeline (classic Graph-RAG):
  question -> parse intent+entities -> generate ontology-aware query (Cypher)
           -> RETRIEVE from the KG / analytics -> assemble grounded context
           -> SYNTHESIZE a natural-language answer with citations.

Retrieval is deterministic graph traversal (fast, no hallucination). Synthesis
is template-based by default but exposes an `llm` hook: pass any callable
(any hosted or local LLM client) to generate the prose while keeping the
citations grounded in retrieved nodes/edges.
"""
from __future__ import annotations

from typing import Callable, Optional

import networkx as nx

from novacart.graph import analytics, clv
from novacart.graph.build_graph import load_graph
from novacart.graph.ontology import EdgeType, NodeType, node_id
from novacart.graph.rag import intents
from novacart.graph.rag.intents import ParsedQuery, cypher_for


class GraphRAG:
    def __init__(self, G: Optional[nx.MultiDiGraph] = None,
                 llm: Optional[Callable[[str, str], str]] = None):
        self.G = G if G is not None else load_graph()
        self.llm = llm  # optional (question, context) -> answer

    # -- public API ---------------------------------------------------------
    def ask(self, question: str) -> dict:
        pq = intents.parse(question)
        cypher = cypher_for(pq)
        records, citations = self._retrieve(pq)
        answer = self._synthesize(question, pq, records, citations)
        return {
            "question": question,
            "intent": pq.intent,
            "generated_query": cypher,
            "answer": answer,
            "citations": citations,
            "records": records,
        }

    # -- retrieval ----------------------------------------------------------
    def _retrieve(self, pq: ParsedQuery) -> tuple[list[dict], list[str]]:
        i, e = pq.intent, pq.entities
        if i == intents.INFLUENTIAL_PRODUCTS:
            recs = analytics.influential_products(self.G, top_n=e.get("top_n", 10))
            return recs, [r["product"] for r in recs]
        if i == intents.SIMILAR_PRODUCTS:
            pid = e.get("product_id")
            node = node_id(NodeType.PRODUCT, pid) if pid is not None else None
            recs = analytics.similar_products(self.G, node) if node else []
            return recs, [r["product"] for r in recs]
        if i == intents.RECOMMEND_FOR_CUSTOMER:
            return self._recommend(e.get("customer_id"))
        if i == intents.TOP_CLV:
            recs = clv.top_clv(e.get("top_n", 10))
            return recs, [node_id(NodeType.CUSTOMER, r["customer_id"]) for r in recs]
        if i == intents.CUSTOMER_PROFILE:
            return self._customer_profile(e.get("customer_id"))
        if i == intents.SEGMENTS:
            comm = analytics.product_communities(self.G)
            return comm["communities"], [f"community:{c['community_id']}" for c in comm["communities"]]
        if i == intents.CATEGORY_LEADERS:
            recs = [r for r in analytics.influential_products(self.G, top_n=200)
                    if r.get("category") == e.get("category")][: e.get("top_n", 10)]
            return recs, [r["product"] for r in recs]
        return [], []

    def _recommend(self, customer_id) -> tuple[list[dict], list[str]]:
        if customer_id is None:
            return [], []
        cnode = node_id(NodeType.CUSTOMER, customer_id)
        if not self.G.has_node(cnode):
            return [], []
        owned = {v for _, v, d in self.G.out_edges(cnode, data=True)
                 if d.get("etype") == EdgeType.BOUGHT}
        scores: dict[str, float] = {}
        for p in owned:
            for _, q, d in self.G.out_edges(p, data=True):
                if d.get("etype") == EdgeType.CO_PURCHASED and q not in owned:
                    scores[q] = scores.get(q, 0.0) + float(d.get("weight", 1))
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:5]
        recs = [{"product": n, "affinity_score": round(s, 2),
                 "category": self.G.nodes[n].get("category")} for n, s in ranked]
        return recs, [cnode] + [r["product"] for r in recs]

    def _customer_profile(self, customer_id) -> tuple[list[dict], list[str]]:
        if customer_id is None:
            return [], []
        info = clv.customer_clv(customer_id)
        if info is None:
            return [], []
        cnode = node_id(NodeType.CUSTOMER, customer_id)
        return [info], [cnode]

    # -- synthesis ----------------------------------------------------------
    def _synthesize(self, question, pq, records, citations) -> str:
        if self.llm is not None:
            context = self._format_context(pq, records)
            return self.llm(question, context)  # grounded generation hook
        return self._template_answer(pq, records)

    def _format_context(self, pq: ParsedQuery, records: list[dict]) -> str:
        head = f"Intent={pq.intent}; retrieved {len(records)} grounded records:\n"
        return head + "\n".join(f"- {r}" for r in records[:10])

    def _template_answer(self, pq: ParsedQuery, records: list[dict]) -> str:
        i = pq.intent
        if not records and i != intents.UNKNOWN:
            return "No matching entities were found in the knowledge graph for that question."
        if i == intents.INFLUENTIAL_PRODUCTS:
            top = ", ".join(f"{r['product']} (PR={r['pagerank']}, {r['category']})" for r in records[:5])
            return f"The most network-influential products by co-purchase PageRank are: {top}."
        if i == intents.SIMILAR_PRODUCTS:
            top = ", ".join(f"{r['product']} (sim={r['similarity']})" for r in records[:5])
            return f"Products most similar in embedding space: {top}."
        if i == intents.RECOMMEND_FOR_CUSTOMER:
            top = ", ".join(f"{r['product']} (affinity={r['affinity_score']}, {r['category']})" for r in records)
            return f"Next-best products via co-purchase affinity: {top}."
        if i == intents.TOP_CLV:
            top = ", ".join(f"cust {r['customer_id']} (CLV=${r['predicted_clv']:,.0f}, {r['segment_rfm']})" for r in records[:5])
            return f"Highest predicted-CLV customers: {top}."
        if i == intents.CUSTOMER_PROFILE:
            r = records[0]
            return (f"Customer {r['customer_id']}: predicted CLV ${r['predicted_clv']:,.0f}, "
                    f"historical ${r['historical_value']:,.0f}, {r['frequency']} orders, "
                    f"AOV ${r['avg_order_value']:,.0f}, recency {r['recency_days']}d, "
                    f"RFM segment '{r['segment_rfm']}'.")
        if i == intents.SEGMENTS:
            return (f"Detected {len(records)} product-affinity communities. "
                    + "; ".join(f"community {r['community_id']} (n={r['size']}, {r['dominant_category']})" for r in records[:5]) + ".")
        if i == intents.CATEGORY_LEADERS:
            top = ", ".join(f"{r['product']} (PR={r['pagerank']})" for r in records[:5])
            return f"Leading products in that category by influence: {top}."
        return ("I couldn't map that to a supported query. Try: influential products, "
                "products similar to <id>, recommend for customer <id>, top CLV customers, "
                "customer <id> profile, or customer segments.")


if __name__ == "__main__":
    rag = GraphRAG()
    for q in [
        "Who are the most influential products?",
        "What products are frequently bought with product 5?",
        "Recommend next best products for customer 42",
        "Show me the top high-value customers",
        "Tell me about customer 42",
        "What customer segments exist?",
        "Top products in electronics",
    ]:
        res = rag.ask(q)
        print(f"\nQ: {q}\n  intent={res['intent']}\n  cypher={res['generated_query']}\n  A: {res['answer']}")
