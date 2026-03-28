"""neo4j_service.py
====================
Neo4j integration for Diagnotix.  Mirrors the graph_service.py API so
graph_service can dispatch to Neo4j when USE_NEO4J=true.

Provides:
  - sync_graph_to_neo4j(G)       — full-replace sync from a NetworkX DiGraph
  - load_test_nodes_neo4j()      — mirrors graph_service.load_test_nodes()
  - load_graph_json_neo4j(...)   — mirrors graph_service.load_graph_json()
  - get_existing_node_ids_neo4j()— mirrors graph_service.get_existing_node_ids()
  - close()                      — close the driver on app shutdown

Required env vars: NEO4J_URI, NEO4J_PASSWORD
Optional env vars: NEO4J_USER (default: "neo4j")
"""

import os

from neo4j import GraphDatabase

from backend.services.utils import _clean

# ─────────────────────────────────────────────────────────────────────────────
# Lazy singleton driver
# ─────────────────────────────────────────────────────────────────────────────
_DRIVER = None


def _get_driver():
    global _DRIVER
    if _DRIVER is None:
        uri = os.environ["NEO4J_URI"]
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ["NEO4J_PASSWORD"]
        _DRIVER = GraphDatabase.driver(uri, auth=(user, password))
    return _DRIVER


def close() -> None:
    """Close the driver — call on app shutdown if USE_NEO4J=true."""
    global _DRIVER
    if _DRIVER is not None:
        _DRIVER.close()
        _DRIVER = None


# ─────────────────────────────────────────────────────────────────────────────
# Sync helpers
# ─────────────────────────────────────────────────────────────────────────────
_BATCH_SIZE = 500

# Cypher templates — label/rel_type are interpolated from the schema (not user input)
_NODE_MERGE_TMPL = """
UNWIND $batch AS row
MERGE (n:{label} {{node_id: row.node_id}})
SET n += row.props
"""

_EDGE_MERGE_TMPL = """
UNWIND $batch AS row
MATCH (a {{node_id: row.src}}), (b {{node_id: row.tgt}})
MERGE (a)-[r:{rel_type}]->(b)
SET r += row.props
"""


def _node_props(attrs: dict) -> dict:
    """Convert NetworkX node attributes to Neo4j-safe property dict."""
    result = {}
    for k, v in attrs.items():
        cleaned = _clean(v)
        if cleaned is None or isinstance(cleaned, (str, int, float, bool)):
            result[k] = cleaned
        elif isinstance(cleaned, list):
            # Flatten to list of strings/numbers (Neo4j supports homogeneous lists)
            result[k] = [
                x if isinstance(x, (int, float, bool)) else str(x)
                for x in cleaned
                if x is not None
            ]
        else:
            result[k] = str(cleaned)
    return result


def _edge_props(attrs: dict) -> dict:
    """Convert NetworkX edge attributes to Neo4j-safe property dict."""
    result = {}
    for k, v in attrs.items():
        cleaned = _clean(v)
        if cleaned is None or isinstance(cleaned, (str, int, float, bool)):
            result[k] = cleaned
        else:
            result[k] = str(cleaned)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Sync
# ─────────────────────────────────────────────────────────────────────────────

def sync_graph_to_neo4j(G) -> None:
    """Full-replace sync: clears all nodes/edges then re-creates from G.

    Nodes are grouped by their "type" attribute so each group gets the right
    Neo4j label without requiring APOC.  Edges are grouped by "relationship".
    """
    driver = _get_driver()

    # 1. Clear the entire graph
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

    # 2. Upsert nodes — group by label so we can use a static label in Cypher
    nodes_by_label: dict[str, list[dict]] = {}
    for node_id, attrs in G.nodes(data=True):
        label = attrs.get("type", "Unknown")
        # Ensure label is a valid Neo4j identifier (replace spaces, hyphens)
        label = label.replace(" ", "_").replace("-", "_")
        if label not in nodes_by_label:
            nodes_by_label[label] = []
        nodes_by_label[label].append({"node_id": node_id, "props": _node_props(attrs)})

    with driver.session() as session:
        for label, nodes in nodes_by_label.items():
            cypher = _NODE_MERGE_TMPL.format(label=label)
            for i in range(0, len(nodes), _BATCH_SIZE):
                session.run(cypher, batch=nodes[i : i + _BATCH_SIZE])

    # 3. Upsert edges — group by relationship type
    edges_by_rel: dict[str, list[dict]] = {}
    for src, tgt, attrs in G.edges(data=True):
        rel_type = attrs.get("relationship", "RELATED_TO")
        rel_type = rel_type.upper().replace(" ", "_").replace("-", "_")
        if rel_type not in edges_by_rel:
            edges_by_rel[rel_type] = []
        edges_by_rel[rel_type].append({"src": src, "tgt": tgt, "props": _edge_props(attrs)})

    with driver.session() as session:
        for rel_type, edges in edges_by_rel.items():
            cypher = _EDGE_MERGE_TMPL.format(rel_type=rel_type)
            for i in range(0, len(edges), _BATCH_SIZE):
                session.run(cypher, batch=edges[i : i + _BATCH_SIZE])

    print(
        f"[neo4j] Synced {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Query functions (mirror graph_service.py API)
# ─────────────────────────────────────────────────────────────────────────────

def load_test_nodes_neo4j() -> list[dict]:
    """Return sorted list of {"id", "label"} for all Diagnostic_Test nodes."""
    driver = _get_driver()
    with driver.session() as session:
        records = session.run(
            "MATCH (n:Diagnostic_Test) RETURN n.node_id AS node_id"
        ).data()
    tests = []
    for row in records:
        node_id = row["node_id"]
        label = node_id.split(": ", 1)[1] if ": " in node_id else node_id
        tests.append({"id": node_id, "label": label})
    return sorted(tests, key=lambda x: x["label"])


def load_graph_json_neo4j(pathway: str | None = None) -> dict:
    """Return {"nodes": [...], "edges": [...]} from Neo4j.

    If *pathway* is provided (and not "All Pathways"), returns the 1-hop
    subgraph centred on that Diagnostic_Test node — same semantics as the
    PKL-based load_graph_json().
    """
    driver = _get_driver()

    with driver.session() as session:
        if pathway and pathway.lower() != "all pathways":
            test_node_id = f"Test: {pathway}"

            # Collect the 1-hop cluster node IDs
            cluster_result = session.run(
                """
                MATCH (test {node_id: $node_id})
                OPTIONAL MATCH (pred)-[]->(test)
                OPTIONAL MATCH (test)-[]->(succ)
                RETURN
                    collect(DISTINCT pred.node_id) +
                    [$node_id] +
                    collect(DISTINCT succ.node_id) AS cluster_ids
                """,
                node_id=test_node_id,
            ).single()

            if cluster_result is None:
                raise ValueError(f"Pathway '{pathway}' not found in Neo4j graph.")

            cluster_ids = [cid for cid in cluster_result["cluster_ids"] if cid]

            nodes_data = session.run(
                "MATCH (n) WHERE n.node_id IN $ids RETURN n.node_id AS node_id, properties(n) AS props",
                ids=cluster_ids,
            ).data()
            edges_data = session.run(
                """
                MATCH (a)-[r]->(b)
                WHERE a.node_id IN $ids AND b.node_id IN $ids
                RETURN a.node_id AS src, b.node_id AS tgt, properties(r) AS props
                """,
                ids=cluster_ids,
            ).data()

        else:
            nodes_data = session.run(
                "MATCH (n) RETURN n.node_id AS node_id, properties(n) AS props"
            ).data()
            edges_data = session.run(
                "MATCH (a)-[r]->(b) RETURN a.node_id AS src, b.node_id AS tgt, properties(r) AS props"
            ).data()

    # Deserialise nodes — properties() always returns a plain Python dict
    nodes = []
    for row in nodes_data:
        node_id = row["node_id"] or ""
        props = dict(row["props"])
        props.pop("node_id", None)
        node: dict = {"id": node_id}
        for k, v in props.items():
            node[k] = _clean(v)
        if "type" in node:
            node["node_type"] = node["type"]
        nodes.append(node)

    # Deserialise edges — properties() always returns a plain Python dict
    edges = []
    for row in edges_data:
        edge: dict = {"source": row["src"], "target": row["tgt"]}
        for k, v in row["props"].items():
            key = "guideline_source" if k == "source" else k
            edge[key] = _clean(v)
        edges.append(edge)

    return {"nodes": nodes, "edges": edges}


def get_existing_node_ids_neo4j() -> set[str]:
    """Return the set of node IDs currently in Neo4j."""
    driver = _get_driver()
    with driver.session() as session:
        records = session.run("MATCH (n) RETURN n.node_id AS node_id").data()
    return {row["node_id"] for row in records if row["node_id"]}
