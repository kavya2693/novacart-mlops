"""Intent + entity parsing for the Graph-RAG assistant.

This is the NL->ontology-aware-query layer. A question is
matched to one intent; each intent knows (a) which graph operation answers it
and (b) the equivalent Cypher, which we surface for transparency ("here is the
query I ran"). Deterministic and unit-testable; an LLM can replace this router
later without changing the retrieval or synthesis contracts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from novacart.common.config import CATEGORIES


@dataclass
class ParsedQuery:
    intent: str
    entities: dict = field(default_factory=dict)
    matched_on: str = ""


# Intent names
INFLUENTIAL_PRODUCTS = "influential_products"
SIMILAR_PRODUCTS = "similar_products"
RECOMMEND_FOR_CUSTOMER = "recommend_for_customer"
TOP_CLV = "top_clv"
CUSTOMER_PROFILE = "customer_profile"
SEGMENTS = "segments"
CATEGORY_LEADERS = "category_leaders"
UNKNOWN = "unknown"


def _first_int(text: str) -> int | None:
    m = re.search(r"\b(\d{1,7})\b", text)
    return int(m.group(1)) if m else None


def _find_category(text: str) -> str | None:
    t = text.lower()
    for c in CATEGORIES:
        if c in t:
            return c
    return None


def parse(question: str) -> ParsedQuery:
    q = question.lower().strip()
    cat = _find_category(q)
    num = _first_int(q)

    # order matters: most specific first
    if re.search(r"recommend|next best|next-best|what should .* buy", q) and "customer" in q:
        return ParsedQuery(RECOMMEND_FOR_CUSTOMER, {"customer_id": num}, "recommend+customer")

    if re.search(r"similar|bought with|frequently bought|goes with|pairs with", q):
        return ParsedQuery(SIMILAR_PRODUCTS, {"product_id": num}, "similarity/co-purchase")

    if re.search(r"segment|communit|cluster|group", q):
        return ParsedQuery(SEGMENTS, {}, "segmentation")

    if cat and re.search(r"top|best|leading|popular|selling", q):
        return ParsedQuery(CATEGORY_LEADERS, {"category": cat, "top_n": num or 10}, "category-leaders")

    if re.search(r"top|high[- ]?value|best|valuable|vip|biggest", q) and re.search(r"customer|clv|lifetime|spender", q):
        return ParsedQuery(TOP_CLV, {"top_n": num or 10}, "top-clv")

    if re.search(r"customer|clv|lifetime value|profile|tell me about", q) and num is not None:
        return ParsedQuery(CUSTOMER_PROFILE, {"customer_id": num}, "customer-profile")

    if re.search(r"influential|bestsell|best-sell|central|top product|popular product|most bought", q):
        return ParsedQuery(INFLUENTIAL_PRODUCTS, {"top_n": num or 10}, "pagerank")

    return ParsedQuery(UNKNOWN, {"raw": question}, "no-match")


def cypher_for(pq: ParsedQuery) -> str:
    """The Cypher we would run on Neo4j/Neptune — surfaced for transparency."""
    e = pq.entities
    if pq.intent == INFLUENTIAL_PRODUCTS:
        return ("MATCH (p:Product)-[:CO_PURCHASED]-(:Product) "
                f"RETURN p ORDER BY pagerank(p) DESC LIMIT {e.get('top_n', 10)}")
    if pq.intent == SIMILAR_PRODUCTS:
        return (f"MATCH (p:Product {{id:{e.get('product_id')}}})-[r:CO_PURCHASED]-(q:Product) "
                "RETURN q ORDER BY r.weight DESC LIMIT 5")
    if pq.intent == RECOMMEND_FOR_CUSTOMER:
        return (f"MATCH (c:Customer {{id:{e.get('customer_id')}}})-[:BOUGHT]->(p:Product)"
                "-[:CO_PURCHASED]-(rec:Product) "
                "WHERE NOT (c)-[:BOUGHT]->(rec) RETURN rec ORDER BY sum(weight) DESC LIMIT 5")
    if pq.intent == TOP_CLV:
        return (f"MATCH (c:Customer) RETURN c ORDER BY c.predicted_clv DESC LIMIT {e.get('top_n', 10)}")
    if pq.intent == CUSTOMER_PROFILE:
        return (f"MATCH (c:Customer {{id:{e.get('customer_id')}}})-[:BOUGHT]->(p:Product) "
                "RETURN c, collect(p)")
    if pq.intent == SEGMENTS:
        return "CALL gds.louvain.stream('product_copurchase') YIELD nodeId, communityId RETURN *"
    if pq.intent == CATEGORY_LEADERS:
        return (f"MATCH (p:Product)-[:IN_CATEGORY]->(:Category {{name:'{e.get('category')}'}}) "
                f"RETURN p ORDER BY pagerank(p) DESC LIMIT {e.get('top_n', 10)}")
    return "// no query — intent not recognized"
