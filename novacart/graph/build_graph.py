"""Build the NovaCart Knowledge Graph from raw tables.

Reads customers / products / orders / order_items from the store and assembles
a networkx property graph following the ontology. Also derives the two
analytics-critical projections:
  - Customer -BOUGHT-> Product   (weighted purchase bipartite edges)
  - Product <-CO_PURCHASED-> Product (basket co-occurrence, the recommender base)

Persists the graph to GraphML so the analytics + RAG services load it fast.
Loading the same graph into Neo4j/Neptune is not implemented here; GraphML
is the interchange format such a loader would start from.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd

from novacart.common import storage
from novacart.common.config import settings
from novacart.graph.ontology import EdgeType, NodeType, node_id

GRAPH_PATH = settings.data_dir / "graph" / "novacart_kg.graphml"


def build_graph() -> nx.MultiDiGraph:
    customers = storage.read_sql("SELECT * FROM customers")
    products = storage.read_sql("SELECT * FROM products")
    orders = storage.read_sql("SELECT * FROM orders")
    items = storage.read_sql("SELECT * FROM order_items")

    G = nx.MultiDiGraph()

    # Entities -----------------------------------------------------------
    for _, c in customers.iterrows():
        G.add_node(node_id(NodeType.CUSTOMER, int(c["customer_id"])),
                   ntype=NodeType.CUSTOMER, segment=c["segment"],
                   signup_date=c["signup_date"], label=f"Customer {int(c['customer_id'])}")
    for _, p in products.iterrows():
        G.add_node(node_id(NodeType.PRODUCT, int(p["product_id"])),
                   ntype=NodeType.PRODUCT, category=p["category"],
                   price=float(p["price"]), label=f"Product {int(p['product_id'])}")
    for cat in products["category"].unique():
        G.add_node(node_id(NodeType.CATEGORY, cat), ntype=NodeType.CATEGORY, label=str(cat))

    # Product -> Category
    for _, p in products.iterrows():
        G.add_edge(node_id(NodeType.PRODUCT, int(p["product_id"])),
                   node_id(NodeType.CATEGORY, p["category"]), etype=EdgeType.IN_CATEGORY)

    # Orders + line items -> BOUGHT + CO_PURCHASED ----------------------
    order_to_cust = dict(zip(orders["order_id"], orders["customer_id"]))
    bought_weight: dict[tuple[int, int], int] = {}
    copurchase: dict[tuple[int, int], int] = {}

    for oid, grp in items.groupby("order_id"):
        cust = order_to_cust.get(oid)
        if cust is None:
            continue
        prods = grp["product_id"].astype(int).tolist()
        for pid in prods:
            bought_weight[(int(cust), pid)] = bought_weight.get((int(cust), pid), 0) + 1
        for a, b in combinations(sorted(set(prods)), 2):
            copurchase[(a, b)] = copurchase.get((a, b), 0) + 1

    for (cust, pid), w in bought_weight.items():
        G.add_edge(node_id(NodeType.CUSTOMER, cust), node_id(NodeType.PRODUCT, pid),
                   etype=EdgeType.BOUGHT, weight=w)
    for (a, b), w in copurchase.items():
        # store both directions for undirected traversal convenience
        G.add_edge(node_id(NodeType.PRODUCT, a), node_id(NodeType.PRODUCT, b),
                   etype=EdgeType.CO_PURCHASED, weight=w)
        G.add_edge(node_id(NodeType.PRODUCT, b), node_id(NodeType.PRODUCT, a),
                   etype=EdgeType.CO_PURCHASED, weight=w)

    return G


def save_graph(G: nx.MultiDiGraph) -> Path:
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, GRAPH_PATH)
    return GRAPH_PATH


def load_graph() -> nx.MultiDiGraph:
    if not GRAPH_PATH.exists():
        G = build_graph()
        save_graph(G)
        return G
    return nx.read_graphml(GRAPH_PATH)


if __name__ == "__main__":
    G = build_graph()
    path = save_graph(G)
    ntypes = pd.Series([d["ntype"] for _, d in G.nodes(data=True)]).value_counts()
    etypes = pd.Series([d["etype"] for _, _, d in G.edges(data=True)]).value_counts()
    print(f"NovaCart KG built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print("  nodes by type:\n" + ntypes.to_string())
    print("  edges by type:\n" + etypes.to_string())
    print(f"  saved -> {path}")
