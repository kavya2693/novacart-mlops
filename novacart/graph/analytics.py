"""Graph analytics: PageRank, centrality, community detection, embeddings.

Operates on projections of the NovaCart KG:
  - product co-purchase graph  -> PageRank (influential/bestseller products),
                                  community detection (product affinity segments),
                                  SVD embeddings (similarity / recommendation).
  - customer projection         -> degree centrality (connected/active buyers).

All algorithms are standard networkx; embeddings use truncated SVD on the
normalized adjacency (a defensible, dependency-light stand-in for node2vec).
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from novacart.graph.ontology import EdgeType, NodeType


def product_copurchase_graph(G: nx.MultiDiGraph) -> nx.Graph:
    """Undirected weighted product-product graph from CO_PURCHASED edges."""
    H = nx.Graph()
    for n, d in G.nodes(data=True):
        if d.get("ntype") == NodeType.PRODUCT:
            H.add_node(n, **d)
    for u, v, d in G.edges(data=True):
        if d.get("etype") == EdgeType.CO_PURCHASED and H.has_node(u) and H.has_node(v):
            w = float(d.get("weight", 1))
            if H.has_edge(u, v):
                H[u][v]["weight"] = max(H[u][v]["weight"], w)
            else:
                H.add_edge(u, v, weight=w)
    return H


def influential_products(G: nx.MultiDiGraph, top_n: int = 10) -> list[dict]:
    """PageRank over the product co-purchase graph = network-influential products."""
    H = product_copurchase_graph(G)
    if H.number_of_edges() == 0:
        return []
    pr = nx.pagerank(H, weight="weight")
    ranked = sorted(pr.items(), key=lambda kv: -kv[1])[:top_n]
    return [
        {"product": n, "pagerank": round(score, 6),
         "category": H.nodes[n].get("category"), "price": H.nodes[n].get("price")}
        for n, score in ranked
    ]


def product_communities(G: nx.MultiDiGraph, resolution: float = 1.0) -> dict:
    """Louvain community detection = product affinity segments."""
    H = product_copurchase_graph(G)
    if H.number_of_edges() == 0:
        return {"n_communities": 0, "communities": []}
    comms = nx.community.louvain_communities(H, weight="weight", seed=42, resolution=resolution)
    out = []
    for i, c in enumerate(sorted(comms, key=len, reverse=True)):
        cats: dict[str, int] = {}
        for n in c:
            cat = H.nodes[n].get("category", "?")
            cats[cat] = cats.get(cat, 0) + 1
        dominant = max(cats, key=cats.get) if cats else "?"
        out.append({"community_id": i, "size": len(c),
                    "dominant_category": dominant, "members_sample": list(c)[:8]})
    return {"n_communities": len(out), "communities": out}


def customer_centrality(G: nx.MultiDiGraph, top_n: int = 10) -> list[dict]:
    """Degree centrality on the customer->product buy graph (breadth of buying)."""
    scores = []
    for n, d in G.nodes(data=True):
        if d.get("ntype") == NodeType.CUSTOMER:
            deg = sum(1 for _ in G.out_edges(n))
            scores.append((n, deg, d.get("segment")))
    scores.sort(key=lambda t: -t[1])
    return [{"customer": n, "distinct_products": deg, "segment": seg}
            for n, deg, seg in scores[:top_n]]


def product_embeddings(G: nx.MultiDiGraph, dim: int = 16) -> tuple[list[str], np.ndarray]:
    """SVD embeddings of the product co-purchase graph (node2vec stand-in)."""
    H = product_copurchase_graph(G)
    nodes = list(H.nodes())
    if len(nodes) < 2:
        return nodes, np.zeros((len(nodes), dim))
    A = nx.to_numpy_array(H, nodelist=nodes, weight="weight")
    # Row-normalize then truncated SVD.
    row = A.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    P = A / row
    k = min(dim, len(nodes) - 1)
    U, S, _ = np.linalg.svd(P, full_matrices=False)
    emb = U[:, :k] * S[:k]
    return nodes, emb


def similar_products(G: nx.MultiDiGraph, product_node: str, top_n: int = 5) -> list[dict]:
    """Cosine similarity in embedding space = 'products like this one'."""
    nodes, emb = product_embeddings(G)
    if product_node not in nodes:
        return []
    idx = nodes.index(product_node)
    v = emb[idx]
    norms = np.linalg.norm(emb, axis=1) * (np.linalg.norm(v) + 1e-9)
    sims = (emb @ v) / (norms + 1e-9)
    order = np.argsort(-sims)
    out = []
    for j in order:
        if nodes[j] == product_node:
            continue
        out.append({"product": nodes[j], "similarity": round(float(sims[j]), 4)})
        if len(out) >= top_n:
            break
    return out
