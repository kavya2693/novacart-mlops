"""NovaCart Knowledge-Graph ontology.

The explicit schema the Graph-RAG assistant reasons over. This is the
"ontology-aware query" pattern: NL questions are resolved against these entity
and relationship types before any graph traversal happens.

The same types would map to RDF/OWL classes (Neptune) or labels (Neo4j); here
it is the contract for the networkx property graph.
"""
from __future__ import annotations

from dataclasses import dataclass


# --- Node (entity) types ---------------------------------------------------
class NodeType:
    CUSTOMER = "Customer"
    PRODUCT = "Product"
    CATEGORY = "Category"
    ORDER = "Order"


# --- Edge (relationship) types --------------------------------------------
class EdgeType:
    PLACED = "PLACED"            # Customer -> Order
    CONTAINS = "CONTAINS"        # Order -> Product
    IN_CATEGORY = "IN_CATEGORY"  # Product -> Category
    BOUGHT = "BOUGHT"            # Customer -> Product (aggregated, weighted)
    CO_PURCHASED = "CO_PURCHASED"  # Product <-> Product (basket co-occurrence)


@dataclass(frozen=True)
class RelationSpec:
    edge_type: str
    source: str
    target: str
    directed: bool
    description: str


ONTOLOGY: list[RelationSpec] = [
    RelationSpec(EdgeType.PLACED, NodeType.CUSTOMER, NodeType.ORDER, True,
                 "A customer placed an order"),
    RelationSpec(EdgeType.CONTAINS, NodeType.ORDER, NodeType.PRODUCT, True,
                 "An order contains a product line item"),
    RelationSpec(EdgeType.IN_CATEGORY, NodeType.PRODUCT, NodeType.CATEGORY, True,
                 "A product belongs to a category"),
    RelationSpec(EdgeType.BOUGHT, NodeType.CUSTOMER, NodeType.PRODUCT, True,
                 "Aggregated customer->product purchases with weight=qty"),
    RelationSpec(EdgeType.CO_PURCHASED, NodeType.PRODUCT, NodeType.PRODUCT, False,
                 "Two products appeared in the same basket; weight=co-count"),
]


def node_id(node_type: str, key) -> str:
    """Stable, typed node id, e.g. 'Customer:42' or 'Category:electronics'."""
    return f"{node_type}:{key}"


def describe() -> str:
    lines = ["NovaCart KG ontology:", "  Entities: " + ", ".join(
        [NodeType.CUSTOMER, NodeType.PRODUCT, NodeType.CATEGORY, NodeType.ORDER])]
    lines.append("  Relationships:")
    for r in ONTOLOGY:
        arrow = "->" if r.directed else "<->"
        lines.append(f"    ({r.source}) -[{r.edge_type}]{arrow} ({r.target}): {r.description}")
    return "\n".join(lines)
